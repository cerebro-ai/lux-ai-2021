from __future__ import annotations

from collections import deque
from pathlib import Path
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
        epochs: int,
        batch_size: int,
        map_states: Tensor,
        piece_states: Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
):
    """Yield mini-batches."""
    rollout_length = map_states.size()[0]

    for _ in range(epochs):
        for _ in range(rollout_length // batch_size):
            rand_ids = np.random.choice(rollout_length, batch_size)
            yield map_states[rand_ids, :], \
                  piece_states[rand_ids, :], \
                  actions[rand_ids], \
                  values[rand_ids], \
                  log_probs[rand_ids], \
                  returns[rand_ids], \
                  advantages[rand_ids]


class ActorCritic(nn.Module):
    def __init__(self, num_actions, policy_hidden_dim, value_hidden_dim, with_meta_node, embedding, device):
        super(ActorCritic, self).__init__()

        self.device = device
        self.with_meta_node = with_meta_node

        self.embedding_model = MapEmbeddingTower(
            **embedding
        ).to(device)

        self.policy_head_network = nn.Sequential(
            nn.Linear(in_features=embedding["output_dim"] + 3,
                      out_features=policy_hidden_dim),
            nn.ELU(),
            nn.Linear(in_features=policy_hidden_dim,
                      out_features=policy_hidden_dim),
            nn.ELU(),
            nn.Linear(in_features=policy_hidden_dim,
                      out_features=num_actions)
        ).to(device)

        self.value_head_network = nn.Sequential(
            nn.Linear(in_features=self.embedding_model.output_dim,
                      out_features=value_hidden_dim),
            nn.ELU(),
            nn.Linear(in_features=value_hidden_dim,
                      out_features=value_hidden_dim),
            nn.ELU(),
            nn.Linear(in_features=value_hidden_dim,
                      out_features=1),
            nn.Tanh()
        ).to(device)

        self.edge_index_cache = {}

    def forward(self, map_tensor, piece_tensor):
        """

        Returns:
            action, dist, value
        """
        assert map_tensor.dim() == 4  # batch, x, y, features
        map_emb_flat = self.embed_map(map_tensor)

        action, dist = self.get_action(map_emb_flat, piece_tensor)
        value = self.value(map_emb_flat)

        return action, dist, value

    def get_action(self, map_emb_flat, piece_tensor):
        p_type, pos, action_mask = split_piece_tensor(piece_tensor)

        p_type_encoding = nn.functional.one_hot(p_type, 3).squeeze(1)
        batches = map_emb_flat.size()[0]

        j_h = torch.Tensor([12, 1]).unsqueeze(0).repeat(batches, 1).to(self.device)
        j = torch.sum(pos * j_h, 1).long()
        indices = j[..., None, None].expand(-1, 1, map_emb_flat.size(2))
        cell_state = torch.gather(map_emb_flat, dim=1, index=indices).squeeze(1)

        piece_state = torch.cat([cell_state, p_type_encoding], 1)

        logits = self.policy_head_network(piece_state)
        mask_value = torch.finfo(logits.dtype).min
        logits_masked = logits.masked_fill(~action_mask, mask_value)
        dist = Categorical(logits=logits_masked)
        action = dist.sample()
        return action, dist

    def embed_map(self, map_tensor):
        """

        Returns:
            flat_map_emb: Tensor of size [batches, nodes, features]
        """
        assert map_tensor.dim() == 4

        batches = map_tensor.size()[0]
        map_size_x = map_tensor.size()[1]
        map_size_y = map_tensor.size()[2]
        features = map_tensor.size()[3]

        assert map_size_x == map_size_y, f"Map is not quadratic: {map_tensor.size()}"

        map_flat = torch.reshape(map_tensor, (batches, -1, features))  # batch, nodes, features

        # get edge_index from cache or compute new and cache
        if map_size_x in self.edge_index_cache:
            edge_index = self.edge_index_cache[map_size_x].to(self.device)
        else:
            edge_index = get_board_edge_index(map_size_x, map_size_y, self.with_meta_node).to(self.device)
            self.edge_index_cache[map_size_x] = edge_index

        x, large_edge_index, _ = batches_to_large_graph(map_flat, edge_index)
        x, large_edge_index = x.to(self.device), large_edge_index.to(self.device)
        large_map_emb_flat = self.embedding_model(x, large_edge_index)

        map_emb_flat, _ = large_graph_to_batches(large_map_emb_flat, None, batches)
        return map_emb_flat

    def value(self, map_emb_flat):
        # extract meta node
        meta_node_state = map_emb_flat[:, -1, :]
        value = self.value_head_network(meta_node_state)
        return value


def piece_to_tensor(piece: dict):
    p_type = torch.IntTensor([piece["type"]])
    pos = torch.IntTensor(piece["pos"])
    action_mask = torch.IntTensor(piece["action_mask"])
    piece_tensor = torch.cat([p_type, pos, action_mask]).unsqueeze(0)
    return piece_tensor


def split_piece_tensor(piece_tensor: Tensor):
    p_type = torch.narrow(piece_tensor, 1, 0, 1).long()
    pos = torch.narrow(piece_tensor, 1, 1, 2).long()
    action_mask = torch.narrow(piece_tensor, 1, 3, 13).bool()
    return p_type, pos, action_mask


class LuxPPOAgent(LuxAgent):
    MAP_FEATURES = 18
    ACTION_SPACE = 13

    def __init__(self, learning: dict, model: dict, **kwargs):
        super(LuxPPOAgent, self).__init__()

        self.learning_rate = learning.get("learning_rate", 0.001)
        self.gamma = learning.get("gamma", 0.98)
        self.tau = learning.get("tau", 0.8)
        self.batch_size = learning.get("batch_size")
        self.clip_param = learning.get("clip_param", 0.2)
        self.epochs = learning.get("epochs", 2)
        self.entropy_weight = learning.get("entropy_weight", 0.0005)

        self.learning_config = learning
        self.model_config = model

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        self.piece_dim = 1
        self.map_size = None

        self.actor_critic = ActorCritic(**model, device=self.device).to(self.device)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

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
        _map = torch.FloatTensor(observation["_map"]).unsqueeze(0)
        for piece_id, piece in observation.items():
            if piece_id.startswith("_"):
                continue
            piece_tensor = piece_to_tensor(piece)
            action = self.select_action(_map, piece_tensor)
            actions[piece_id] = int(action)
        self.last_returned_actions_length = len(actions.keys())
        return actions

    def receive_reward(self, reward: float, done: int):
        length = self.last_returned_actions_length
        rewards = torch.FloatTensor([reward / length] * length).to(self.device)
        masks = torch.FloatTensor([1 - done] * length).to(self.device)
        self.masks.extend(masks)
        self.rewards.extend(rewards)

    def select_action(self, map_tensor: Tensor, piece_tensor: Tensor):
        """Select a action for from the map and piece
        """
        map_tensor, piece_tensor = map_tensor.to(self.device), piece_tensor.to(self.device)
        with torch.no_grad():
            action, dist, value = self.actor_critic(map_tensor, piece_tensor)
            selected_action = torch.argmax(dist.logits) if self.is_test else action  # get most probable action

        if not self.is_test:
            # in training mode
            self.map_states.append(map_tensor.detach())
            self.piece_states.append(piece_tensor.detach())
            self.actions.append(torch.unsqueeze(selected_action, 0).detach())
            self.values.append(value.detach())
            self.log_probs.append(torch.unsqueeze(torch.Tensor([dist.log_prob(selected_action)]), 0).to(self.device))

        return selected_action.cpu().detach().numpy()

    def update_model(self, last_obs):
        device = self.device

        with torch.no_grad():

            _map = torch.FloatTensor(last_obs["_map"]).unsqueeze(0).to(device)
            _map_emb = self.actor_critic.embed_map(_map)
            next_value = self.actor_critic.value(_map_emb)

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

        losses = []

        for map_tensor, piece_tensor, _, old_value, old_log_prob, return_, advantage in ppo_iter(
                epochs=self.epochs,
                batch_size=self.batch_size,
                map_states=map_states,
                piece_states=piece_states,
                actions=actions,
                values=values,
                log_probs=log_probs,
                returns=returns,
                advantages=advantages,
        ):
            action, dist, value = self.actor_critic(map_tensor, piece_tensor)
            entropy = dist.entropy().mean()
            new_log_prob = dist.log_prob(action)

            ratio = (new_log_prob - old_log_prob).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - self.entropy_weight * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach().item())

        self.piece_states, self.map_states, self.actions, self.rewards = [], [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        mean_loss = sum(losses) / len(losses)

        wandb.log({
            "mean_loss": mean_loss
        })

        return mean_loss

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
        self.actor_critic = ActorCritic(**self.model_config, device=self.device)

        checkpoint = torch.load(path)
        self.learning_rate = checkpoint["learning_rate"]
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]
        self.batch_size = checkpoint["batch_size"]
        self.clip_param = checkpoint["clip_param"]
        self.epochs = checkpoint["epochs"]
        self.entropy_weight = checkpoint["entropy_weight"]

        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.actor_critic.to(self.device)

    def save(self, games_played: int, target="models"):
        Path(target).mkdir(exist_ok=True)

        target = Path(target).joinpath(f"PPO_model_game_{games_played}")
        torch.save({
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "clip_param": self.clip_param,
            "epochs": self.epochs,
            "entropy_weight": self.entropy_weight,
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        },
            target)
