import random

import numpy as np
import torch
import wandb

from luxai21.agent.lux_agent import LuxAgent
from luxai21.agent.ppo_agent import LuxPPOAgent
from luxai21.env import example_config
from luxai21.env.lux_env import LuxEnv
from luxai21.env.utils import get_city_tile_count, log_citytiles_game_end

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    config = example_config.config
    wandb.init(
        project=config["wandb"]["project"],
        notes=config["wandb"]["notes"],
        tags=config["wandb"]["tags"],
        config=config)

    n_games = 10

    agent_config = config["agent"]
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

        log_citytiles_game_end(env.game_state)

        for player, agent in agents.items():
            actor_loss, critic_loss = agent.update_model(obs[player])
            losses[player]["actor_losses"].append(actor_loss)
            losses[player]["critic_losses"].append(critic_loss)

        for agent in agents.values():
            agent.match_over_callback()

        if game_i % config["wandb"]["replay_at_epochs"] == 0:
            wandb.log({
                f"Replay_epoch{game_i}": wandb.Html(env.render())
            })


if __name__ == '__main__':
    main()
