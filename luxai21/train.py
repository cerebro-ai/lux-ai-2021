import glob
import os

from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed, get_schedule_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from luxai21.agent_policy import AgentPolicy
from luxpythonenv.env.agent import Agent
from luxpythonenv.env.lux_env import LuxEnvironment
from luxpythonenv.game.constants import LuxMatchConfigs_Default

from luxai21.config import Hyperparams, default_config

# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=SubprocVecEnv#multiprocessing-unleashing-the-power-of-vectorized-environments
from luxai21.models.feature_extr import CustomFeatureExtractor
from luxai21.models.policy import CustomActorCriticPolicy


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


def train(config: Hyperparams):
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
    if config.training.n_envs == 1:
        env = LuxEnvironment(configs=configs,
                             learning_agent=player,
                             opponent_agent=opponent)
    else:
        env = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                     learning_agent=AgentPolicy(mode="train"),
                                                     opponent_agent=opponent), i) for i in range(config.training.n_envs)])

    run_id = config.training.id
    print("Run id %s" % run_id)

    # --- START INIT MODEL ---

    if config.training.path:
        # by default previous model params are used (lr, batch size, gamma...)
        model = PPO.load(config.training.path)
        model.set_env(env=env)

        # Update the learning rate
        model.lr_schedule = get_schedule_fn(config.training.learning_rate)

        # TODO: Update other training parameters
    else:

        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(map_emb_dim=config.model.map_emb_dim),
            net_arch=[*config.model.net_arch_shared_layers,
                      dict(
                          vf=config.model.net_arch_vf,
                          pi=config.model.net_arch_pi
                      )]
        )

        model = PPO(CustomActorCriticPolicy,
                    env,  # sets observation & action space
                    verbose=1,
                    tensorboard_log="./lux_tensorboard/",
                    learning_rate=config.training.learning_rate,
                    gamma=config.training.gamma,
                    gae_lambda=config.training.gae_lambda,
                    batch_size=config.training.batch_size,
                    n_steps=config.training.n_steps,
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
    if config.training.n_envs > 1:
        # An evaluation environment is needed to measure multi-env setups. Use a fixed 4 envs.
        env_eval = SubprocVecEnv([make_env(LuxEnvironment(configs=configs,
                                                          learning_agent=AgentPolicy(mode="train"),
                                                          opponent_agent=opponent), i) for i in range(4)])

        callbacks.append(
            EvalCallback(env_eval, best_model_save_path=f'./logs_{run_id}/',
                         log_path=f'./logs_{run_id}/',
                         eval_freq=config.training.n_steps * 2,  # Run it every 2 training iterations
                         n_eval_episodes=30,  # Run 30 games
                         deterministic=False, render=False)
        )

    print("Training model...")
    model.learn(total_timesteps=config.training.step_count,
                callback=callbacks)
    if not os.path.exists(f'models/rl_model_{run_id}_{config.training.step_count}_steps.zip'):
        model.save(path=f'models/rl_model_{run_id}_{config.training.step_count}_steps.zip')
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
    config = Hyperparams.load(str(default_config))
    train(config)


if __name__ == "__main__":
    main()
