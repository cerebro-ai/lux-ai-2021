import contextlib
import os

import numpy as np
from hydra import initialize, compose
from omegaconf import DictConfig
from ray.rllib import RolloutWorker
from ray.rllib.agents import ppo, MultiCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models import ModelCatalog
from ray.tune import register_env, tune
from ray.util.client import ray

from luxai21.callbacks.metrics import MetricsCallback
from luxai21.callbacks.weights import UpdateWeightsCallback
from luxai21.callbacks.wandb import WandbLoggerCallback
from luxai21.env.lux_ma_env import LuxMAEnv
from luxai21.env.stacked_env import StackedLuxMAEnv
from luxai21.models.rllib.city_tile import BasicCityTileModel
from luxai21.models.rllib.worker import WorkerModel
from luxai21.models.rllib.worker_tile_lstm import WorkerLSTMModel
from luxai21.models.rllib.worker_v2 import WorkerLSTMModelV2
from luxai21.models.rllib.worker_v3 import WorkerModelV3
from luxai21.policy.city_tile import EagerCityTilePolicy
from luxai21.policy.worker import get_worker_policy, get_do_nothing_worker_policy
from luxai21.policy.random import RandomWorkerPolicy


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def run(cfg: DictConfig):
    if cfg.debug:
        ray.init(local_mode=True)
    else:
        ray.init()

    print("pid", os.getpid())

    # ENVIRONMENT
    env_creator = lambda env_config: StackedLuxMAEnv(config=env_config)
    register_env("lux_ma_env", lambda env_config: env_creator(env_config=env_config))

    # MODEL
    ModelCatalog.register_custom_model("worker_model", WorkerModel)
    ModelCatalog.register_custom_model("worker_model_v2", WorkerLSTMModelV2)
    ModelCatalog.register_custom_model("worker_model_v3", WorkerModelV3)
    ModelCatalog.register_custom_model("basic_city_tile_model", BasicCityTileModel)

    # Update callback settings
    class MetricsCallbacks(MetricsCallback):
        log_replays = cfg["metrics"].get("log_replays", False)

    class WeightsCallback(UpdateWeightsCallback):
        win_rate_to_rotate = cfg["weights"].get("win_rate_to_update", 0.75)
        min_steps_between_updates = cfg["weights"].get("min_steps_between_updates", 10)

    def policy_mapping_fn(agent_id: str, episode: MultiAgentEpisode, worker: RolloutWorker, **kwargs):
        # p0_u_1_23245
        # p0_ct_c_2_2_3_2345
        if "ct_" in agent_id:
            return "city_tile_policy"
        else:
            team = int(agent_id[1])
            player_team = (episode.episode_id % 2)
            if team == player_team:
                return "player_worker"
            else:
                if cfg.weights.self_play:
                    return "player_worker"

                if (episode.episode_id % 999) == 0:
                    return "do_nothing_worker"
                else:
                    episode_id = episode.episode_id
                    # use episode_id as seed such that all agents
                    # in one episode are mapped to the same policy
                    with temp_seed(episode_id):
                        opponent_id = np.random.choice([1, 2, 3], p=[2 / 3, 2 / 9, 1 / 9])
                    return "opponent_worker_" + str(opponent_id)

    config = {
        "multiagent": {
            "policies": {
                "player_worker": get_worker_policy(cfg),
                "opponent_worker_1": get_worker_policy(cfg),
                "opponent_worker_2": get_worker_policy(cfg),
                "opponent_worker_3": get_worker_policy(cfg),
                "do_nothing_worker": get_do_nothing_worker_policy(),
                "city_tile_policy": EagerCityTilePolicy
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["player_worker"],
        },
        "env": cfg.env.env,
        "env_config": {
            **cfg.env.env_config,
            "wandb": cfg.wandb
        },
        "callbacks": MultiCallbacks([
            MetricsCallbacks,
            WeightsCallback
        ]),
        **cfg.algorithm.config,
        "framework": "torch",
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    }

    if cfg.debug:

        trainer = ppo.PPOTrainer(config=config, env="lux_ma_env")
        for i in range(30):
            result = trainer.train()
    else:
        results = tune.run(cfg.algorithm.name,
                           config=config,
                           stop=dict(cfg.stop),
                           verbose=cfg.verbose,
                           local_dir=None,
                           checkpoint_at_end=cfg.checkpoint_at_end,
                           checkpoint_freq=cfg["checkpoint_freq"],
                           restore=cfg.get("restore", None),
                           callbacks=[
                               WandbLoggerCallback(
                                   **cfg.wandb,
                                   init_config=dict(cfg),
                                   log_config=False)
                           ])

    ray.shutdown()


if __name__ == '__main__':
    initialize(config_path="conf")
    cfg = compose(config_name="config")
    run(cfg)
