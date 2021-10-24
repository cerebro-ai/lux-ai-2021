from __future__ import annotations

from collections import deque
from typing import List, Deque, Any
import wandb
import os

import numpy as np
import torch
from torch import nn, optim, Tensor
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv

from luxai21.agent.lux_agent import LuxAgent
from luxai21.models.gnn.map_embedding import MapEmbeddingTower
from luxai21.models.gnn.utils import get_board_edge_index, batches_to_large_graph, large_graph_to_batches


def compute_gae(
        next_value: Tensor,
        rewards: List[Tensor],
        masks: List[Any],
        values: List[Tensor],
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
        batch_size: int,
        map_states: Tensor,
        piece_states: Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
):
    """Yield mini-batches."""
    rollout_length = map_states.size()[0]

    for _ in range(epoch):
        for _ in range(rollout_length // batch_size):
            rand_ids = np.random.choice(rollout_length, batch_size)
            yield map_states[rand_ids, :], piece_states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids], returns[rand_ids], advantages[rand_ids]


class PieceActor(nn.Module):
    """
    TODO implement to use past actions (LSTM, Transformer)
    """

    def __init__(self, mlp_hidden_dim, out_dim, embedding_model):
        super(PieceActor, self).__init__()
        self.mlp_hidden_dim = mlp_hidden_dim

        self.map_gnn = embedding_model

        self.model = nn.Sequential(
            nn.Linear(embedding_model.output_dim + 3, mlp_hidden_dim),
            nn.ELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ELU(),
            nn.Linear(mlp_hidden_dim, out_dim)
        )

    def forward(self, map_tensor: Tensor, piece_tensor: Tensor, edge_index):
        """
        Args:
            map_tensor: batch of map_states. size:(80, 12, 12, 19)
            piece_tensor
        """
        assert map_tensor.dim() == 4  # batch, x, y, features
        batches = map_tensor.size()[0]
        features = map_tensor.size()[2]
        map_flat = torch.reshape(map_tensor, (batches, -1, features))  # batch, nodes, features

        x, large_edge_index, _ = batches_to_large_graph(map_flat, edge_index)
        large_map_emb_flat = self.map_gnn(x, large_edge_index)

        map_emb_flat = large_graph_to_batches(large_map_emb_flat, None, None)

        p_type, pos, action_mask = split_piece_tensor(piece_tensor)
        p_type_encoding = nn.functional.one_hot(p_type, 3)

        """
        Alternative:
        map_emb = torch.reshape(map_emb_flat, (-1, 12, 12, self.hidden_dim))

        cell_state = torch.cat([map_emb[i, pos[i, 0], pos[i, 1], :].unsqueeze(0) for i in range(80)])
        """

        j_h = torch.Tensor([12, 1]).unsqueeze(0).repeat(batches, 1)
        j = torch.sum(pos * j_h, 1).long()
        indices = j[..., None, None].expand(-1, 1, map_emb_flat.size(2))
        cell_state = torch.gather(map_emb_flat, dim=1, index=indices).squeeze(1)

        piece_state = torch.cat([cell_state, p_type_encoding], 1)

        logits = self.model(piece_state)
        mask_value = torch.finfo(logits.dtype).min
        logits_masked = logits.masked_fill(~action_mask, mask_value)
        dist = Categorical(logits=logits_masked)
        action = dist.sample()
        return action, dist


class Critic(nn.Module):
    def __init__(self, embedding_model, hidden_dim):
        super(Critic, self).__init__()
        self.embedding_model = embedding_model

        self.model = nn.Sequential(
            nn.Linear(in_features=embedding_model.output_dim,
                      out_features=hidden_dim),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ELU(),
            nn.Linear(in_features=hidden_dim, out_features=1),
            nn.Tanh()
        )

    def forward(self, map_tensor: Tensor, edge_index: Tensor):
        batches = map_tensor.size()[0]
        features = map_tensor.size()[2]
        map_flat = torch.reshape(map_tensor, (batches, -1, features))  # batch, nodes, features

        x, large_edge_index, _ = batches_to_large_graph(map_flat, edge_index)
        large_map_emb_flat = self.map_gnn(x, large_edge_index)

        map_emb_flat = large_graph_to_batches(large_map_emb_flat, None, None)

        # (10, 145, 128)

        meta_node_state = map_emb_flat[:, -1, :]
        value = self.model(meta_node_state)
        return value


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
    MAP_FEATURES = 18
    ACTION_SPACE = 13

    def __init__(self, learning_rate, gamma, tau, batch_size, epsilon, epochs, entropy_weight, **kwargs):
        super(LuxPPOAgent, self).__init__()

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epochs = epochs
        self.entropy_weight = entropy_weight

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        self.piece_dim = 1
        self.map_size = None

        # networks
        self.edge_index = None
        self.edge_index_cache = {}  # this will cache the edge_indices

        self.tower = MapEmbeddingTower(
            input_dim=19,
            hidden_dim=128,
            output_dim=128
        )

        self.actor_config = dict(
            mlp_hidden_dim=64,
            out_dim=13,
            embedding_model=self.tower
        )

        self.critic_config = dict(
            embedding_model=self.tower,
            hidden_dim=64,
        )

        self.actor = PieceActor(**self.actor_config).to(self.device)

        self.critic = Critic(**self.critic_config).to(self.device)

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

    def update_edge_index(self, _map: np.ndarray):
        """
        Between games the map can change in size, but since this does not happen to often,
        we check that here and update only if we need to. Already called edge_indices are stored in the cache
        """
        map_size = _map.shape[1]
        if self.map_size is None or map_size != self.map_size:
            self.map_size = map_size
            if map_size in self.edge_index_cache:
                self.edge_index = self.edge_index_cache[map_size]
            else:
                self.edge_index = get_board_edge_index(map_size, map_size, with_meta_node=False)
                self.edge_index_cache[map_size] = self.edge_index

        self.actor.edge_index = self.edge_index
        self.critic.edge_index = self.edge_index

    def generate_actions(self, observation: dict):
        # check if map is still of the same size
        self.update_edge_index(observation["_map"])

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
        """Select a action for from the map and piece
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
                epoch=self.epochs,
                batch_size=self.batch_size,
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

    def extend_replay_data(self, agent: LuxPPOAgent):
        """Copy the replay data from the given agent
        """
        self.piece_states.extend(agent.piece_states)
        self.map_states.extend(agent.map_states)
        self.actions.extend(agent.actions)
        self.rewards.extend(agent.rewards)
        self.values.extend(agent.values)
        self.masks.extend(agent.masks)
        self.log_probs.extend(agent.log_probs)

    def load(self, path):
        self.actor = PieceActor(**self.actor_config)
        self.critic = Critic(**self.critic_config)

        checkpoint = torch.load(path)
        self.learning_rate = checkpoint["learning_rate"]
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.batch_size = checkpoint["batch_size"]
        self.epsilon = checkpoint["epsilon"]
        self.epochs = checkpoint["epoch"]
        self.entropy_weight = checkpoint["entropy_weight"]

        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])

        self.critic.to(self.device)
        self.actor.to(self.device)

    def save(self, target="models", name=None):
        if name is not None:
            target = os.path.join(target, f'{name}_complete_PPOmodel_checkpoint')
        else:
            target = os.path.join(target, f'complete_PPOmodel_checkpoint_epoch_{self.epochs}')
        torch.save({
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "epsilon": self.epsilon,
            "epoch": self.epochs,
            "entropy_weight": self.entropy_weight,
            "critic_state_dict": self.critic.to('cpu').state_dict(),
            "actor_state_dict": self.actor.to('cpu').state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
        },
            target)
