import numpy as np
from torch import nn
from torch.distributions import Categorical
import wandb


class LuxAgent:
    def __init__(self):
        self.statistics = {
            "episode_length": 0,  # int
            "city_tiles_per_turn": [],  # line
            "city_tiles_at_end": 0,
            "worker_per_turn": []
        }
        self.actions = [3, 4, 5, 6]

    def generate_actions(self, observation: dict):
        raise NotImplementedError('generate_action not implemented')

    def receive_reward(self, reward: float, done: int):
        raise NotImplementedError('receive_reward not implemented')

    def match_over_callback(self):
        pass
