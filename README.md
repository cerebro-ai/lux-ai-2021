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

!pip install omegaconf
!pip install lz4
``` 

Set API Key for Weights & Biases (Metrics logging)
and loglevel of logger

```python
import os

os.environ["WANDB_API_KEY"] = "your api key"
os.environ["LOGURU_LEVEL"] = "WARNING"
```

Insert following code in cell

```python
config = {
    'algorithm': {
        'name': 'PPO', 
        'config': {
            'num_workers': 1, 
            'num_envs_per_worker': 1, 
            'rollout_fragment_length': 16, 
            'train_batch_size': 128, 
            'num_sgd_iter': 3, 
            'lr': 0.0002, 
            'sgd_minibatch_size': 128, 
            'batch_mode': 'truncate_episodes'
        }
    }, 
    'env': {
        'env': 'lux_ma_env', 
        'env_config': {
            'game': {
                'height': 12, 
                'width': 12
            }, 'env': {
                'allow_carts': False
            }, 
            'team_spirit': 0, 
            'reward': {
                'move': 0, 
                'transfer': 0, 
                'build_city': 1, 
                'pillage': 0, 
                'build_worker': 1, 
                'build_cart': 0.1, 
                'research': 1, 
                'turn_worker': 0.1, 
                'turn_citytile': 0.1, 
                'death_city': -1, 
                'win': 10,
                'wood_collected': 0.1,
                'coal_collected': 0.2,
                'uranium_collected': 0.3,
                'fuel_generated': 0.2,
                'research_points': 0.1,
                'coal_researched': 2,
                'uranium_researched': 5,
                'citytiles_end': 2,
                'citytiles_end_opponent': -1
            }, 
            'save_replay': {
                'local_every_x': 1, 
                'wandb_every_x': 1
            }
        }
    }, 
    'model': {
        'worker': {
            'custom_model': 
                'worker_model', 
            'custom_model_config': {
                'use_meta_node': True, 
                'policy_hidden_dim': 16, 
                'policy_output_dim': 9, 
                'value_hidden_dim': 16, 
                'map_embedding': {
                    'input_dim': 21, 
                    'hidden_dim': 32, 
                    'output_dim': 32
                }
                ''
            }
        }
    }, 
    'debug': False, 
    'verbose': 1, 
    'stop': {
        'timesteps_total': 50000
    }, 
    'checkpoint_at_end': True, 
    'checkpoint_freq': 5, 
    'wandb': {
        'entity': 'cerebro-ai', 
        'project': 'luxai21', 
        'group': 'dev', 
        'notes': 'just testing lstms', 
        'tags': ['GNN', 'Dev', 'Rllib']
    }
}

from omegaconf import OmegaConf
config = OmegaConf.create(config)


```

and start training

```python
from luxai21.train_rllib import run

run(config, ppo_config, stop, debug=False)
```