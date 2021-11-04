import os

import numpy as np
from gym.spaces import *
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env, tune
from ray.util.client import ray
from ray.tune.integration.wandb import WandbLoggerCallback

from luxai21.env.lux_ma_env import LuxMAEnv
from luxai21.models.rllib.city_tile import BasicCityTilePolicy, BasicCityTileModel
from luxai21.models.rllib.worker import WorkerModel


def run(debug=True):
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

    def policy_mapping_fn(agent_id, **kwargs):
        if "ct_" in agent_id:
            return "city_tile_policy"
        else:
            return "worker_policy"

    config = {
        "env": "lux_ma_env",
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "env_config": {
            "game": {
                "height": 12,
                "width": 12,
            },
            "env": {
                "allow_carts": False
            }
        },
        "multiagent": {
            "policies": {
                "worker_policy": PolicySpec(
                    action_space=Discrete(9),
                    observation_space=Dict(
                        **{'map': Box(shape=(12, 12, 21),
                                      dtype=np.float64,
                                      low=-float('inf'),
                                      high=float('inf')
                                      ),
                           'game_state': Box(shape=(26,),
                                             dtype=np.float64,
                                             low=float('-inf'),
                                             high=float('inf')
                                             ),
                           'type': Discrete(3),
                           'pos': Box(shape=(2,),
                                      dtype=np.float64,
                                      low=0,
                                      high=99999
                                      ),
                           'action_mask': Box(shape=(9,),
                                              dtype=np.float64,
                                              low=0,
                                              high=1
                                              )}),
                    config={
                        "model": {
                            "custom_model": "worker_model",
                            "custom_model_config": {
                                "use_meta_node": True,
                                "map_embedding": {
                                    "input_dim": 21,  # TODO use game_state
                                    "hidden_dim": 64,
                                    "output_dim": 64
                                },
                                "policy_hidden_dim": 32,
                                "policy_output_dim": 9,
                                "value_hidden_dim": 32,
                            }
                        }
                    }),
                # Hardcoded policy
                "city_tile_policy": PolicySpec(
                    policy_class=BasicCityTilePolicy,
                    action_space=Discrete(4),
                    observation_space=Dict(
                        **{'map': Box(shape=(12, 12, 21),
                                      dtype=np.float64,
                                      low=-float('inf'),
                                      high=float('inf')
                                      ),
                           'game_state': Box(shape=(26,),
                                             dtype=np.float64,
                                             low=float('-inf'),
                                             high=float('inf')
                                             ),
                           'type': Discrete(3),
                           'pos': Box(shape=(2,),
                                      dtype=np.float64,
                                      low=0,
                                      high=99999
                                      ),
                           'action_mask': Box(shape=(4,),
                                              dtype=np.float64,
                                              low=0,
                                              high=1
                                              )}),
                    # config={
                    #     "model": {
                    #         "custom_model": "basic_city_tile_model"
                    #     }
                    # }
                ),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["worker_policy"],
        },
        "framework": "torch",
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    }

    stop = {
        "timesteps_total": 100000
    }

    ppo_config = {
        "rollout_fragment_length": 512,
        "train_batch_size": 512,
        "num_sgd_iter": 5,
        "lr": 1e-4,
        "sgd_minibatch_size": 256,
        "batch_mode": "truncate_episodes",
    }

    if debug:

        config = {**config, **ppo_config}
        trainer = ppo.PPOTrainer(config=config, env="lux_ma_env")

        for i in range(3):
            result = trainer.train()
    else:
        config = {**config, **ppo_config}
        results = tune.run("PPO", config=config, stop=stop, verbose=1, callbacks=[
            WandbLoggerCallback(
                project="luxai21",
                group="cerebro-ai",
                notes="Shared embedding network",
                tags=["GNNs", "Dev", "rrlib"],
                log_config=True)])

    ray.shutdown()


if __name__ == '__main__':
    run(debug=False)
