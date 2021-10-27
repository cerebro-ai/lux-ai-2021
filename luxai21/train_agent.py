import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
import time
import copy

import yaml
from tqdm import tqdm
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


def train(config=None):
    log.info(f"Start training")
    start_time = time.time()  # since kaggle notebooks only run for 9 hours

    loglevel = os.environ.get("LOGURU_LEVEL")
    log.debug(f"Seed: {config.get('seed')}")

    if config is None:
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
    agent2.is_test = True

    agents = {
        "player_0": agent1,
        "player_1": agent2
    }

    for agent in agents.values():
        agent.is_test = False

    env = LuxEnv(config)

    losses = {
        player: {
            "mean_losses": []
        } for player in agents.keys()
    }

    update_step = 0
    total_games = 0
    best_citytiles_end = 10
    obs = None

    while total_games < config["training"]["max_games"]:
        games = 0
        citytiles_end = []

        # gather data by playing complete games until replay_buffer of one agent is larger than given threshold
        # times two since we use also the replay data of agent2
        while len(agent1.rewards) < config["training"]["max_replay_buffer_size"]:
            obs = env.reset()
            done = env.game_state.match_over()
            log.debug(f"Start game {games}")

            if loglevel not in ["WARNING", "ERROR"]:
                turn_bar = tqdm(total=360, desc="Game progress", ncols=90)
            # GAME TURNS
            while not done:
                # 1. generate actions
                actions = {
                    player: agent.generate_actions(obs[player])
                    for player, agent in agents.items()
                }

                # 2. pass actions to env
                try:
                    obs, rewards, dones, infos = env.step(actions)
                except AttributeError as e:
                    # Game env errored
                    log.error(e)
                    agent1.receive_reward(0, 1)
                    # agent2.receive_reward(0, 1)
                    break

                # 3. pass reward to agents
                agent1.receive_reward(rewards["player_0"], dones["player_0"])
                # agent2.receive_reward(rewards["player_1"], dones["player_1"])

                # 4. check if game is over
                done = env.game_state.match_over()

                if loglevel not in ["WARNING", "ERROR"]:
                    turn_bar.update(1)

            if loglevel not in ["WARNING", "ERROR"]:
                turn_bar.close()
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

        if (update_step % config["training"]["save_checkpoint_every_x_updates"]) == 0 and update_step != 0:
            agent1.save(total_games)

        mean_loss = agent1.update_model(obs["player_0"])
        losses["player_0"]["mean_losses"].append(mean_loss)

        if (update_step % config["wandb"]["replay_every_x_updates"]) == 0 and update_step != 0:
            log.debug("Save replay")
            wandb.log({
                f"Replay_step{total_games}": wandb.Html(env.render())
            })

        if time.time() - start_time > config["training"]["max_training_time"]:
            log.debug("Time exceeded 'max_training_time': save model")
            agent1.save(total_games)

        # transfer agent1 model to agent2
        agent2.actor_critic = copy.deepcopy(agent1.actor_critic)

        update_step += 1


if __name__ == '__main__':
    params_file = Path(__file__).parent.parent.joinpath("hyperparams.yaml")
    with params_file.open("r") as f:
        hyperparams = yaml.safe_load(f)

    train(hyperparams)
