from collections import deque
from typing import List, Deque, Tuple
import wandb

import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv

from luxai21.agent.lux_agent import LuxAgent
from luxai21.models.gnn.utils import get_board_edge_index


def compute_gae(
        next_value: list,
        rewards: list,
        masks: list,
        values: list,
        gamma: float,
        tau: float
) -> List:
    """Compute gae."""
    values = values + [next_value]
    gae = 0
    returns: Deque[float] = deque()

    for step in reversed(range(len(rewards))):
        delta = (
                rewards[step]
                + gamma * values[step + 1] * masks[step]
                - values[step]
        )
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])

    return list(returns)


def ppo_iter(
        epoch: int,
        mini_batch_size: int,
        map_states: Tensor,
        piece_states: Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
):
    """Yield mini-batches."""
    batch_size = map_states.size(0)

    #  TODO fix this
    mini_batch_size = batch_size
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield map_states[rand_ids, :], piece_states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids], returns[rand_ids], advantages[rand_ids]


class Coach(nn.Module):
    """
    Perform Graph convolution on map and output "strategy" tensor
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Coach, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.activation = nn.ReLU()

    def forward(self, map: Tensor, edge_index: torch.Tensor):
        x = self.conv1(map, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        return x


class PieceActor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(PieceActor, self).__init__()
        self.hidden_dim = hidden_dim
        self.map_gnn = Coach(in_dim, hidden_dim, hidden_dim)
        self.edge_index = get_board_edge_index(12, 12, False)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, map_tensor: Tensor, piece_tensor: Tensor):
        """
        Args:
            map_tensor: (80, 12, 12,19)
        map_flat -> (80, 144, 19)
        map_emb_flat -> (80, 144, 24)
        map_emb -> (80, 12, 12 ,24)
        cell -> (80, 80, 24)
        """
        batches = map_tensor.size()[0]
        map_flat = torch.reshape(map_tensor, (batches, -1, 19))
        map_emb_flat = self.map_gnn(map_flat, self.edge_index)

        p_type, pos, action_mask = split_piece_tensor(piece_tensor)

        """
        Alternative:
        map_emb = torch.reshape(map_emb_flat, (-1, 12, 12, self.hidden_dim))

        cell_state = torch.cat([map_emb[i, pos[i, 0], pos[i, 1], :].unsqueeze(0) for i in range(80)])
        """

        j_h = torch.Tensor([12, 1]).unsqueeze(0).repeat(batches, 1)
        j = torch.sum(pos * j_h, 1).long()
        indices = j[..., None, None].expand(-1, 1, map_emb_flat.size(2))
        cell_state = torch.gather(map_emb_flat, dim=1, index=indices).squeeze(1)

        piece_state = torch.cat([cell_state, p_type], 1)

        logits = self.model(piece_state)
        mask_value = torch.finfo(logits.dtype).min
        logits_masked = logits.masked_fill(~action_mask, mask_value)
        dist = Categorical(logits=logits_masked)
        action = dist.sample()  # TODO check dimension
        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim):
        super(Critic, self).__init__()
        self.map_emb_dim = 8
        self.map_model = Coach(in_dim, 16, self.map_emb_dim)
        self.edge_index = get_board_edge_index(12, 12, False)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 1)
        )

    def forward(self, map_tensor: Tensor):
        batches = map_tensor.size(0)
        map_flat = torch.reshape(torch.FloatTensor(map_tensor), (batches, -1, 19))
        map_emb_flat = self.map_model(map_flat, self.edge_index)
        map_emb = torch.reshape(map_emb_flat, (-1, 12, 12, self.map_emb_dim))
        if len(map_emb.size()) == 3:
            map_emb = torch.unsqueeze(map_emb, 0)
        map_emb = torch.permute(map_emb, (0, 3, 1, 2))
        value = self.model(map_emb)
        return value


def extract_actions(map_strategy: Tensor, pieces: dict, model: nn.Module):
    """
    Args:
        map_strategy: Tensor (strategy_dim, WIDTH, HEIGHT)
        pieces: Dict with piece information: type, pos, action_mask
    """
    actions = {}
    for piece_id, piece in pieces.items():
        pos_x = piece["pos"][0]
        pos_y = piece["pos"][1]
        strategy = map_strategy[:, pos_x, pos_y]
        type = torch.Tensor(piece["type"])
        x = torch.cat([strategy, type])
        action = model(x)

        actions[piece_id] = action


def piece_to_tensor(piece: dict):
    p_type = torch.IntTensor([piece["type"]])
    pos = torch.IntTensor(piece["pos"])
    action_mask = torch.IntTensor(piece["action_mask"])
    piece_tensor = torch.cat([p_type, pos, action_mask]).unsqueeze(0)
    return piece_tensor


def split_piece_tensor(piece_tensor: Tensor):
    p_type = torch.narrow(piece_tensor, 1, 0, 1)
    pos = torch.narrow(piece_tensor, 1, 1, 2).long()
    action_mask = torch.narrow(piece_tensor, 1, 3, 13).bool()
    return p_type, pos, action_mask


class LuxPPOAgent(LuxAgent):
    """
    TODO implement setting random seed
    if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """

    def __init__(self, learning_rate, gamma, tau, batch_size, epsilon, epoch, entropy_weight):
        super(LuxPPOAgent, self).__init__()

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.entropy_weight = entropy_weight

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        self.strategy_dim = 32
        self.piece_dim = 1

        # networks
        self.edge_index = get_board_edge_index(12, 12, with_meta_node=False)  # TODO get sizes dynamically
        self.actor = PieceActor(in_dim=19, hidden_dim=24, out_dim=13).to(self.device)

        # TODO implement critic on strategy map
        self.critic = Critic(in_dim=19).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # memory for training
        self.map_states: List[Tensor] = []
        self.piece_states: List[Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        self.last_returned_actions_length = 0

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def generate_actions(self, observation: dict):
        actions = {}
        for piece_id, piece in observation.items():
            if piece_id.startswith("_"):
                continue
            piece_tensor = piece_to_tensor(piece)
            _map = torch.FloatTensor(observation["_map"]).unsqueeze(0)
            action = self.select_action(_map, piece_tensor)
            actions[piece_id] = int(action)
        self.last_returned_actions_length = len(actions.keys())
        return actions

    def receive_reward(self, reward: float, done: int):
        length = self.last_returned_actions_length
        rewards = torch.FloatTensor([reward / length]).repeat(length).to(self.device)
        masks = torch.FloatTensor([1 - done]).repeat(length).to(self.device)
        self.masks.extend(masks)
        self.rewards.extend(rewards)

    def select_action(self, map_tensor: Tensor, piece_tensor: Tensor):
        """Select a action for a pre-computed strategy vector from the map and piece

        TODO incorporate piece history
        """
        action, dist = self.actor(map_tensor, piece_tensor)
        selected_action = torch.argmax(dist.logits) if self.is_test else action  # get most probable action

        if not self.is_test:
            # in training mode
            value = self.critic(map_tensor)
            self.map_states.append(map_tensor)
            self.piece_states.append(piece_tensor)
            self.actions.append(torch.unsqueeze(selected_action, 0))
            self.values.append(value)
            self.log_probs.append(torch.unsqueeze(torch.Tensor([dist.log_prob(selected_action)]), 0))

        return selected_action.cpu().detach().numpy()

    def update_model(self, last_obs):
        device = self.device

        _map = torch.FloatTensor(last_obs["_map"]).unsqueeze(0).to(device)
        next_value = self.critic(_map)

        returns = compute_gae(next_value,
                              self.rewards,
                              self.masks,
                              self.values,
                              self.gamma,
                              self.tau)

        map_states = torch.cat(self.map_states)
        piece_states = torch.cat(self.piece_states)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []

        for map_tensor, piece_tensor, action, old_value, old_log_prob, return_, adv in ppo_iter(
                epoch=self.epoch,
                mini_batch_size=self.batch_size,
                map_states=map_states,
                piece_states=piece_states,
                actions=actions,
                values=values,
                log_probs=log_probs,
                returns=returns,
                advantages=advantages
        ):
            _, dist = self.actor(map_tensor, piece_tensor)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor loss
            surr_loss = ratio * adv
            clipped_surr_loss = (
                    torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            )

            entropy = dist.entropy().mean()
            wandb.log({'entropy': entropy})

            actor_loss = (
                    - torch.min(surr_loss, clipped_surr_loss).mean()
                    - entropy * self.entropy_weight
            )

            # critic loss
            value = self.critic(map_tensor)
            critic_loss = (return_ - value).pow(2).mean()

            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.piece_states, self.map_states, self.actions, self.rewards = [], [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        wandb.log({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss
        })

        return actor_loss, critic_loss

    def load(self, path):
        self.actor = PieceActor(in_dim=19, hidden_dim=24, out_dim=13)
        # TODO implement critic on strategy map
        self.critic = Critic(in_dim=19)

        checkpoint = torch.load(path)
        self.learning_rate = checkpoint["learning_rate"]
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.batch_size = checkpoint["batch_size"]
        self.epsilon = checkpoint["epsilon"]
        self.epoch = checkpoint["epoch"]
        self.entropy_weight = checkpoint["entropy_weight"]

        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])

        self.critic.to(device)
        self.actor.to(device)

    def save(self, target="models"):
        torch.save({
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "epsilon": self.epsilon,
            "epoch": self.epoch,
            "entropy_weight": self.entropy_weight,
            "critic_state_dict": self.critic.to('cpu').state_dict(),
            "actor_state_dict": self.actor.to('cpu').state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict()
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
        },os.path.join(target, 'complete_PPOmodel_checkpoint'))
