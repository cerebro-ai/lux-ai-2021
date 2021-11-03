import os

import gym.spaces
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env, tune
from ray.util.client import ray

from luxai21.env.lux_ma_env import LuxMAEnv
from luxai21.models.rllib.city_tile import BasicCityTilePolicy
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

    def policy_mapping_fn(agent_id, **kwargs):
        if "ct_" in agent_id:
            return "city_tile_policy"
        else:
            "worker_policy"

    config = {
        "env": "lux_ma_env",
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
                    action_space=gym.spaces.Discrete(9),
                    config={
                        "model": {
                            "custom_model": "worker_model",
                            "custom_model_config": {
                                "use_meta_node": True,
                                "map_embedding": {
                                    "input_dim": 18,  # TODO use game_state
                                    "hidden_dim": 64,
                                    "output_dim": 32
                                },
                                "policy_hidden_dim": 64,
                                "policy_output_dim": 9,
                                "value_hidden_dim": 32,
                            }
                        }
                    }),
                # Hardcoded policy
                "city_tile_policy": PolicySpec(
                    policy_class=BasicCityTilePolicy,
                    action_space=gym.spaces.Discrete(4)
                ),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_learn": ["worker_policy"],
        },
        "framework": "torch",
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    }

    stop = {
        "timesteps_total": 100
    }

    if debug:
        trainer = ppo.PPOTrainer(config=config, env="lux_ma_env")

        for i in range(3):
            result = trainer.train()
    else:
        results = tune.run("PPO", config=config, stop=stop, verbose=1)

    ray.shutdown()


if __name__ == '__main__':
    run()
