import os

from hydra import initialize, compose
from omegaconf import DictConfig
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune import register_env, tune
from ray.util.client import ray

from luxai21.callbacks.wandb import WandbLoggerCallback
from luxai21.env.lux_ma_env import LuxMAEnv
from luxai21.models.rllib.city_tile import BasicCityTileModel
from luxai21.models.rllib.worker import WorkerModel
from luxai21.policy.city_tile import get_city_tile_policy
from luxai21.policy.worker import get_worker_policy


def run(cfg: DictConfig):
    if cfg.debug:
        ray.init(local_mode=True)
    else:
        ray.init()

    # ENVIRONMENT
    env_creator = lambda env_config: LuxMAEnv(config=env_config)
    register_env("lux_ma_env", lambda env_config: env_creator(env_config=env_config))

    # MODEL
    ModelCatalog.register_custom_model("worker_model", WorkerModel)
    ModelCatalog.register_custom_model("basic_city_tile_model", BasicCityTileModel)

    def policy_mapping_fn(agent_id, **kwargs):
        if "ct_" in agent_id:
            return "city_tile_policy"
        else:
            return "worker_policy"

    config = {
        "multiagent": {
            "policies": {
                "worker_policy": get_worker_policy(cfg.model.worker),
                "city_tile_policy": get_city_tile_policy()
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["worker_policy"],
        },
        "env": cfg.env.env,
        "env_config": {
            **cfg.env.env_config,
            "wandb": cfg.wandb
        },
        **cfg.algorithm.config,
        "framework": "torch",
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    }

    if cfg.debug:

        trainer = ppo.PPOTrainer(config=config, env="lux_ma_env")
        for i in range(3):
            result = trainer.train()
    else:
        results = tune.run(cfg.algorithm.name,
                           config=config,
                           stop=dict(cfg.stop),
                           verbose=cfg.verbose,
                           checkpoint_at_end=cfg.checkpoint_at_end,
                           checkpoint_freq=cfg.checkpoint_freq,
                           callbacks=[
                               WandbLoggerCallback(
                                   **cfg.wandb,
                                   log_config=False)
                           ])

    ray.shutdown()


if __name__ == '__main__':
    initialize(config_path="conf")
    cfg = compose(config_name="config")
    run(cfg)
