import random

import numpy as np
import torch
import wandb

from luxai21.agent.ppo_agent import LuxPPOAgent
from luxai21.env import example_config
from luxai21.env.lux_env import LuxEnv
from luxai21.env.utils import log_citytiles_game_end


def set_seed(seed: int):
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    config = example_config.config

    set_seed(config["seed"])

    wandb.init(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
        notes=config["wandb"]["notes"],
        tags=config["wandb"]["tags"],
        config=config)

    total_turns = 10000

    agent1 = LuxPPOAgent(**config["agent"])
    agent2 = LuxPPOAgent(**config["agent"])

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

    turn_step = 0
    games_played = 0
    model_updates = 0

    while turn_step < total_turns:
        print('games played so far:', games_played)

        obs = env.reset()

        # gather data by playing "rollout_length" turn_steps
        for s in range(config["agent"]["rollout_length"]):
            if turn_step % 100 == 0:
                print("turn step", turn_step)

            # generate actions
            actions = {
                player: agent.generate_actions(obs[player])
                for player, agent in agents.items()
            }
            # pass actions to env
            obs, rewards, dones, infos = env.step(actions)

            # pass reward to agents
            for agent_id, agent in agents.items():
                agent.receive_reward(rewards[agent_id], dones[agent_id])

            # check if game is over
            done = env.game_state.match_over()
            if done:
                log_citytiles_game_end(env.game_state)
                games_played += 1
                obs = env.reset()

                if games_played % config["wandb"]["save_replay_every_n_games"] == 0:
                    wandb.log({
                        f"Replay_game_{games_played}": wandb.Html(env.render())
                    })

            turn_step += 1

        for player, agent in agents.items():
            model_updates += 1
            print("Update step", model_updates)
            actor_loss, critic_loss = agent.update_model(obs[player])
            losses[player]["actor_losses"].append(actor_loss)
            losses[player]["critic_losses"].append(critic_loss)


if __name__ == '__main__':
    main()
