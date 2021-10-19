import numpy as np
from torch import nn
from torch.distributions import Categorical
import wandb


class LuxAgent:
    def __init__(self):
        # TODO init model
        model = None

        self.statistics = {
            "episode_length": 0,  # int
            "city_tiles_per_turn": [],  # line
            "city_tiles_at_end": 0,
            "worker_per_turn": []
        }
        self.actions = [3, 4, 5, 6]

    def generate_actions(self, observation: dict):
        # TODO implement get_actions
        raise NotImplementedError('generate_action not implemented')

    def receive_reward(self, reward: float, done: int):
        # TODO implement get_reward
        raise NotImplementedError('receive_reward not implemented')

    def match_over_callback(self):
        pass


def model_forward(obs: dict):
    gnn = nn.Module()
    W = 12
    obs = {
        "_map": (18, W, W),
        "_game_state": (22),
        "u1": {
            "type": 0,
            "pos": (7, 4),
            "action_mask": [1, 1, 1, 1, 0, ..., 1]
        },
        "ct_8_4": {
            "type": 2,
            "pos": (8, 4),
            "action_mask": [1, 1, 1, 1, 0, ..., 1]
        }
    }
    actions = {}

    cell_strategies = gnn(obs["_map"])
    for piece_id in obs.keys():
        if piece_id.startswith("_"):
            continue

        piece_type = obs[piece_id]["type"]
        pos = obs[piece_id]["pos"]
        action_mask = obs[piece_id]["action_mask"]
        action_logits = cell_strategies[:, pos[0], pos[1]]
        #  TODO mask action_logits
        action = Categorical(logits=action_logits).sample(1)

        actions[piece_id] = action
