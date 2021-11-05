from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune import register_env, tune
from ray.util.client import ray
from ray.tune.integration.wandb import WandbLoggerCallback

from luxai21.env.lux_ma_env import LuxMAEnv
from luxai21.models.rllib.city_tile import BasicCityTilePolicy, BasicCityTileModel
from luxai21.models.rllib.worker import WorkerModel


def run(config=None, ppo_config=None, stop=None, debug=True):
    if config is None:
        from luxai21.config_rllib import config, ppo_config, stop

    if debug:
        ray.init(local_mode=True)
    else:
        ray.init()

    # ENVIRONMENT
    env_creator = lambda env_config: LuxMAEnv(config=env_config)
    register_env("lux_ma_env", lambda env_config: env_creator(env_config=env_config))

    # MODEL
    ModelCatalog.register_custom_model("worker_model", WorkerModel)
    ModelCatalog.register_custom_model("basic_city_tile_model", BasicCityTileModel)

    if debug:

        config = {**config, **ppo_config}
        trainer = ppo.PPOTrainer(config=config, env="lux_ma_env")

        for i in range(3):
            result = trainer.train()
    else:
        config = {**config, **ppo_config}
        results = tune.run("PPO", config=config, stop=stop, verbose=1,
                           checkpoint_at_end=True, checkpoint_freq=5,
                           callbacks=[
                               WandbLoggerCallback(
                                   project="luxai21",
                                   group="cerebro-ai",
                                   notes="Shared embedding network",
                                   tags=["GNNs", "Dev", "rrlib"],
                                   log_config=True)])

    ray.shutdown()


if __name__ == '__main__':
    run(debug=False)
