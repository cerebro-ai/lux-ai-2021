import argparse
import glob
import os
import sys
import random
from typing import Callable

from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent_policy import AgentPolicy
from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default

from config import ParamConfigurator

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=SubprocVecEnv#multiprocessing-unleashing-the-power-of-vectorized-environments
from src.models.feature_extr import CustomFeatureExtractor
from src.models.policy import CustomActorCriticPolicy


def make_env(local_env, rank, seed=0):
    """
    Utility function for multi-processed env.

    :param local_env: (LuxEnvironment) the environment
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        local_env.seed(seed + rank)
        return local_env

    set_random_seed(seed)
    return _init


def train(config: ParamConfigurator):
    """
    The main training loop
    :param config: (ParamConfigurator) The parameters from the config
    """
    print(config)

    # Run a training job
    configs = LuxMatchConfigs_Default

    # Create a default opponent agent
    opponent = Agent()

    # Create a RL agent in training mode
    player = AgentPolicy(mode="train")

    # Train the model
    env_eval = None
    if config.n_envs == 1:
        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)
    else:
        env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                     learning_agent=AgentPolicy(mode="train"),
                                                     opponent_agent=opponent), i) for i in range(config.n_envs)])

    run_id = config.id
    print("Run id %s" % run_id)

    # --- START INIT MODEL ---

    if config.path:
        # by default previous model params are used (lr, batch size, gamma...)
        model = PPO.load(config.path)
        model.set_env(env=env)

        # Update the learning rate
        model.lr_schedule = get_schedule_fn(config.learning_rate)

        # TODO: Update other training parameters
    else:

        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(map_emb_dim=config.map_emb_dim),
            net_arch=[*config.net_arch_shared_layers,
                      dict(
                          vf=config.net_arch_vf,
                          pi=config.net_arch_pi
                      )]
        )

        model = PPO(CustomActorCriticPolicy,
                    env,  # sets observation & action space
                    verbose=1,
                    tensorboard_log="./lux_tensorboard/",
                    learning_rate=config.learning_rate,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    batch_size=config.batch_size,
                    n_steps=config.n_steps,
                    policy_kwargs=policy_kwargs
                    )

    # --- END INIT MODEL ---

    callbacks = []
    # Save a checkpoint every 100K steps
    callbacks.append(
        CheckpointCallback(save_freq=100000,
                           save_path='./models/',
                           name_prefix=f'rl_model_{run_id}')
    )

    # Since reward metrics don't work for multi-environment setups, we add an evaluation logger
    # for metrics.
    if config.n_envs > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        env_eval = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                          learning_agent=AgentPolicy(mode="train"),
                                                          opponent_agent=opponent), i) for i in range(4)])

        callbacks.append(
            EvalCallback(env_eval, best_model_save_path=f'./logs_{run_id}/',
                         log_path=f'./logs_{run_id}/',
                         eval_freq=config.n_steps * 2,  # Run it every 2 training iterations
                         n_eval_episodes=30,  # Run 30 games
                         deterministic=False, render=False)
        )

    print("Training model...")
    model.learn(total_timesteps=config.step_count,
                callback=callbacks)
    if not os.path.exists(f'models/rl_model_{run_id}_{config.step_count}_steps.zip'):
        model.save(path=f'models/rl_model_{run_id}_{config.step_count}_steps.zip')
    print("Done training model.")

    # Inference the model
    print("Inference model policy with rendering...")
    saves = glob.glob(f'models/rl_model_{run_id}_*_steps.zip')
    latest_save = sorted(saves, key=lambda x: int(x.split('_')[-2]), reverse=True)[0]
    model.load(path=latest_save)
    obs = env.reset()
    for i in range(600):
        action_code, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action_code)
        if i % 5 == 0:
            print("Turn %i" % i)
            env.render()

        if done:
            print("Episode done, resetting.")
            obs = env.reset()
    print("Done")

    '''
    # Learn with self-play against the learned model as an opponent now
    print("Training model with self-play against last version of model...")
    player = AgentPolicy(mode="train")
    opponent = AgentPolicy(mode="inference", model=model)
    env = LuxEnvironment(configs, player, opponent)
    model = PPO("MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./lux_tensorboard/",
        learning_rate = 0.0003,
        gamma=0.999,
        gae_lambda = 0.95
    )

    model.learn(total_timesteps=2000)
    env.close()
    print("Done")
    '''


def main():
    config = ParamConfigurator()
    train(config)


if __name__ == "__main__":
    main()
