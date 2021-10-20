import random

import numpy as np
import torch
import wandb
import time
import copy

from luxai21.agent.ppo_agent import LuxPPOAgent
from luxai21.env import example_config
from luxai21.env.lux_env import LuxEnv
from luxai21.env.utils import get_city_tile_count, log_and_get_citytiles_game_end


def set_seed(seed: int):
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(config=None):
    start_time = time.time() # since kaggle notebooks only run for 9 hours

    if config is None:
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

    total_games = 0
    best_citytiles_end = 10
    count_updates = 0

    while total_games < config["training"]["max_games"]:
        games = 0
        citytiles_end = []

        # gather data by playing complete games until replay_buffer of one agent is larger than given threshold
        while len(agent1.rewards) < config["training"]["max_replay_buffer_size"]:
            obs = env.reset()
            done = env.game_state.match_over()

            while not done:
                # generate actions
                actions = {
                    player: agent.generate_actions(obs[player])
                    for player, agent in agents.items()
                }

                # pass actions to env
                obs, rewards, dones, infos = env.step(actions)

                # pass reward to agent
                for agent_id, agent in agents.items():
                    agent.receive_reward(rewards[agent_id], dones[agent_id])

                # check if game is over
                done = env.game_state.match_over()
            citytiles_end.append(log_and_get_citytiles_game_end(env.game_state))
            games += 1

        total_games += games
        mean_citytiles_end = sum(citytiles_end)/len(citytiles_end)
        wandb.log({
            "citytiles_end_mean_episode": mean_citytiles_end
        })
        if mean_citytiles_end > best_citytiles_end:
            agent1.save(name='most_citytiles_end')

        if total_games % config["training"]["save_checkpoint_every_x_games"] == 0 and total_games != 0:
            agent1.save()

        # Update models and append losses
        count_updates += 1
        for player, agent in agents.items():
            actor_loss, critic_loss = agent.update_model(obs[player])
            losses[player]["actor_losses"].append(actor_loss)
            losses[player]["critic_losses"].append(critic_loss)

        for agent in agents.values():
            agent.match_over_callback()

        if count_updates % config["wandb"]["replay_every_x_updates"] == 0 and count_updates != 0:
            wandb.log({
                f"Replay_step{total_games}": wandb.Html(env.render())
            })

        if time.time() - start_time > config["training"]["max_training_time"]:
            agent1.save()

        agent2.critic = copy.deepcopy(agent1.critic)
        agent2.actor = copy.deepcopy(agent1.actor)

if __name__ == '__main__':
    train()
