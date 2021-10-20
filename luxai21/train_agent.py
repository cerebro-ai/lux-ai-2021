import random

import numpy as np
import torch
import wandb
import time
import copy

from loguru import logger as log

from luxai21.agent.ppo_agent import LuxPPOAgent
from luxai21.env import example_config
from luxai21.env.lux_env import LuxEnv
from luxai21.env.utils import log_and_get_citytiles_game_end


def set_seed(seed: int):
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    log.info(f"Start training")

    start_time = time.time()  # since kaggle notebooks only run for 9 hours
    config = example_config.config

    set_seed(config["seed"])

    wandb.init(
        entity=config["wandb"]["entity"],
        project=config["wandb"]["project"],
        notes=config["wandb"]["notes"],
        tags=config["wandb"]["tags"],
        config=config)

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
    best_model = None
    obs = None

    while total_games < config["training"]["max_games"]:
        games = 0
        citytiles_end = []

        # gather data by playing complete games until replay_buffer of one agent is larger than given threshold
        # times two since we use also the replay data of agent2
        while len(agent1.rewards) * 2 < config["training"]["max_replay_buffer_size"]:
            obs = env.reset()
            done = env.game_state.match_over()

            # GAME TURNS
            while not done:
                # 1. generate actions
                actions = {
                    player: agent.generate_actions(obs[player])
                    for player, agent in agents.items()
                }

                # 2. pass actions to env
                obs, rewards, dones, infos = env.step(actions)

                # 3. pass reward to agents
                agent1.receive_reward(rewards["player_0"], dones["player_0"])
                agent2.receive_reward(rewards["player_1"], dones["player_1"])

                # 4. check if game is over
                done = env.game_state.match_over()

            # GAME ENDS
            citytiles_end.append(log_and_get_citytiles_game_end(env.game_state))
            games += 1

        log.debug(f"Replay buffer full. Games played: {games}")

        total_games += games
        log.debug(f"Total games so far: {total_games}")

        mean_citytiles_end = sum(citytiles_end) / len(citytiles_end)

        wandb.log({
            "citytiles_end_mean_episode": mean_citytiles_end
        })
        if mean_citytiles_end > best_citytiles_end:
            agent1.save(name='most_citytiles_end')

        if total_games % config["training"]["save_checkpoint_every_x_games"] == 0 and total_games != 0:
            agent1.save()

        # Update models and append losses
        # transfer replay data from agent1 to agent2
        agent1.extend_replay_data(agent2)
        actor_loss, critic_loss = agent1.update_model(obs["player_0"])
        losses["player_0"]["actor_losses"].append(actor_loss)
        losses["player_0"]["critic_losses"].append(critic_loss)

        for agent in agents.values():
            agent.match_over_callback()

        if total_games % config["wandb"]["replay_every_x_games"] == 0 and total_games != 0:
            wandb.log({
                f"Replay_step{total_games}": wandb.Html(env.render())
            })

        if time.time() - start_time > config["training"]["max_training_time"]:
            agent1.save()

        # transfer agent1 model to agent2
        agent2.critic = copy.deepcopy(agent1.critic)
        agent2.actor = copy.deepcopy(agent1.actor)


if __name__ == '__main__':
    main()
