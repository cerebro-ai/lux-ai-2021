from collections import deque
from typing import List, Deque

import numpy as np
import torch
from torch import nn, optim, Tensor
from torch_geometric.graphgym import GNN, GCNConv

from luxai21.agent.lux_agent import LuxAgent


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
        states: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
):
    """Yield mini-batches."""
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[
                rand_ids
            ], log_probs[rand_ids], returns[rand_ids], advantages[rand_ids]


class Coach(nn.Module):
    """
    Perform Graph convolution on map and output "strategy" tensor
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Coach, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, lux_obs: dict, edge_index: torch.Tensor):
        map = lux_obs["_map"]
        x = self.conv1(map, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x


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


class LuxPPOAgent(LuxAgent):

    def __init__(self, learning_rate, gamma, tau, batch_size, epsilon, epoch, rollout_len, entropy_weight):
        super(LuxPPOAgent, self).__init__()

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epoch = epoch
        self.rollout_len = rollout_len
        self.entropy_weight = entropy_weight

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def load(self, model):
        #  TODO implement load from saved state
        #   Should also set all hyperparameters
        pass
