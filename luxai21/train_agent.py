import random

import numpy as np
import torch

from luxai21.agent.lux_agent import LuxAgent
from luxai21.agent.ppo_agent import LuxPPOAgent
from luxai21.env import example_config
from luxai21.env.lux_env import LuxEnv
from luxai21.env.utils import get_city_tile_count

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    n_games = 10

    config = example_config.config

    agent_config = {
        "learning_rate": 0.001,
        "gamma": 0.95,
        "tau": 0.8,
        "batch_size": 80,
        "epsilon": 0.2,
        "epoch": 16,
        "entropy_weight": 0.005
    }

    agent1 = LuxPPOAgent(**agent_config)
    agent2 = LuxPPOAgent(**agent_config)

    agents = {
        "player_0": agent1,
        "player_1": agent2
    }

    for agent in agents.values():
        agent.is_test = False

    env = LuxEnv(config)

    losses = {
        player: {
            "actor_losses": [],
            "critic_losses": []
        } for player in agents.keys()
    }

    scores = []
    score = 0

    for game_i in range(n_games):
        print('game', game_i)

        obs = env.reset()
        done = env.game_state.match_over()

        while not done:
            print("turn", env.turn)
            actions = {
                player: agent.generate_actions(obs[player])
                for player, agent in agents.items()
            }
            obs, rewards, dones, infos = env.step(actions)

            for agent_id, agent in agents.items():
                agent.receive_reward(rewards[agent_id], dones[agent_id])

            done = env.game_state.match_over()

        print(f"Player0 score: {get_city_tile_count(env.game_state, 0)}")
        print(f"Player1 score: {get_city_tile_count(env.game_state, 1)}")

        for player, agent in agents.items():
            actor_loss, critic_loss = agent.update_model(obs[player])
            losses[player]["actor_losses"].append(actor_loss)
            losses[player]["critic_losses"].append(critic_loss)

        for agent in agents.values():
            agent.match_over_callback()


if __name__ == '__main__':
    with torch.autograd.detect_anomaly():
        main()
