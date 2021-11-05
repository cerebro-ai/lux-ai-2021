# lux-ai-2021

Lux AI 2021 Competition

Code for the Lux AI Reinforcement Challenge

- Kaggle Overview: https://www.kaggle.com/c/lux-ai-2021/overview
- Specs: https://www.lux-ai.org/specs-2021
- Visualize: https://2021vis.lux-ai.org/

## Run locally

Check out the repo

Install requirements
```shell
pip install -r requirements.txt
```

Create config_rllib.py file
```python
# ./config_rllib.py

import os
import numpy as np
from gym.spaces import *
from ray.rllib.policy.policy import PolicySpec


from luxai21.models.rllib.city_tile import BasicCityTilePolicy


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
        },
        "team_spirit": 0,
        "reward": {
            "move": 0,
            "transfer": 0,
            "build_city": 1,
            "pillage": 0,

            "build_worker": 1,
            "build_cart": 0.1,
            "research": 1,

            "turn_worker": 0.1,
            "turn_citytile": 0.1,

            "death_city": -1,
            "win": 10
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
                                "hidden_dim": 32,
                                "output_dim": 32
                            },
                            "policy_hidden_dim": 16,
                            "policy_output_dim": 9,
                            "value_hidden_dim": 16,
                        },
                        "use_lstm": False,
                        # Max seq len for training the LSTM, defaults to 20.
                        "max_seq_len": 20,
                        # Size of the LSTM cell.
                        "lstm_cell_size": 256,
                        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
                        "lstm_use_prev_action": True

                        # Attention is untested for Torch and buggy

                        # "use_attention": True,
                        # "max_seq_len": 10,
                        # "attention_num_transformer_units": 1,
                        # "attention_dim": 9,
                        # "attention_memory_inference": 10,
                        # "attention_memory_training": 10,
                        # "attention_num_heads": 1,
                        # "attention_head_dim": 32,
                        # "attention_position_wise_mlp_dim": 9,
                        # "attention_use_n_prev_actions": 1
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
    "timesteps_total": 50000
}

ppo_config = {
    "rollout_fragment_length": 16,
    "train_batch_size": 128,
    "num_sgd_iter": 3,
    "lr": 2e-4,
    "sgd_minibatch_size": 128,
    "batch_mode": "truncate_episodes"
}
```

Run locally
```shell
export PYTHONPATH=.
python luxai21/train_agent.py
```

## Set up kaggle notebook

Install the required packages

```shell
!pip install git+https://<username>:<token or password>@github.com/cerebro-ai/lux-ai-2021.git
!pip install git+https://<username>:<token or password>@github.com/cerebro-ai/lux-python-env.git
!pip install wandb --upgrade
``` 

Set API Key for Weights & Biases (Metrics logging)
and loglevel of logger

```python
import os

os.environ["WANDB_API_KEY"] = "your api key"
os.environ["LOGURU_LEVEL"] = "WARNING"
```

Copy the config_rllib code from above and start training

```python
from luxai21.train_rllib import run

run(config, ppo_config, stop, debug=False)
```