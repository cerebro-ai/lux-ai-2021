"""
This file is to get a sense how many rewards does an random agent, as well as an agent that does nothing accumulate
"""
import os
import time
from datetime import datetime
import random
from typing import Dict, Callable

import numpy as np
from tqdm import tqdm

from luxai21.env.lux_ma_env import LuxMAEnv

config = {
    "save_replay": {
        "wandb_every_x": 0,
        "local_every_x": 0,
    },
    "game": {
        "height": 12,
        "width": 12
    },
    "env": {
        "allow_carts": False
    },
    "team_spirit": 0
}

env = LuxMAEnv(config=config)


def get_eager_city_action(action_mask):
    if action_mask[1] == 1:
        return 1
    elif action_mask[2] == 1:
        return 2
    elif action_mask[3] == 1:
        return 3
    else:
        return 0


def get_random_worker_action(action_mask):
    legal_action_ids = []
    for i in range(len(action_mask)):
        if action_mask[i] == 1:
            legal_action_ids.append(i)
    return random.choice(legal_action_ids)


def get_noop_worker_action(action_mask):
    return 0


def get_actions(obs: Dict[str, Dict], team0_policy: Callable, team1_policy: Callable):
    actions = {}
    for piece_id, piece_obs in obs.items():
        if "ct_" in piece_id:
            # city_tile
            actions[piece_id] = get_eager_city_action(piece_obs["action_mask"])
            continue
        else:
            team = int(piece_id[1])
            if team == 0:
                actions[piece_id] = team0_policy(piece_obs["action_mask"])
            else:
                actions[piece_id] = team1_policy(piece_obs["action_mask"])
    return actions


def get_total_team_rewards(team: int, rewards: Dict):
    """
    filter out city_tiles
    """
    total_reward = 0
    for piece_id, reward in rewards.items():
        if "ct_" not in piece_id:
            if int(piece_id[1]) == team:
                total_reward += reward
    return total_reward


if __name__ == '__main__':

    print("pid", os.getpid())

    start_time = time.time()
    n_games = 200

    # array with (games, turns)
    all_rewards = [np.zeros((n_games, 361)), np.zeros((n_games, 361))]

    # iterate over games
    for i_game in tqdm(range(n_games)):
        obs = env.reset()
        done = False
        while not done:
            actions = get_actions(obs, get_random_worker_action, get_noop_worker_action)
            obs, rewards, dones, infos, rewards_list = env.env_step(actions)
            turn = env.turn

            all_rewards[0][i_game, turn] = get_total_team_rewards(0, rewards)
            all_rewards[1][i_game, turn] = get_total_team_rewards(1, rewards)

            done = dones["__all__"]

    np.save("rewards.npx", all_rewards)

    end_time = time.time()

    print("Total time: ", end_time - start_time)
    print("Games / sec: ", n_games / (end_time - start_time))
    print("* ---- *")

    print("Mean cumulative game reward:")
    print("Team 0", np.sum(all_rewards[0], 1).mean())
    print("Team 1", np.sum(all_rewards[1], 1).mean())
