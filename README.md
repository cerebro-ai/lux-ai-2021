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

Create hyperparameters file
```yaml
# ./hyperparams.yaml

seed: 505

wandb:
  entity: rkstgr
  project: luxai21
  notes: First run with GNNs (example config)
  tags:
    - GNNs
    - Reward_func1
  replay_every_x_updates: 5

game:
  height: 12
  width: 12

training:
  max_games: 100000
  max_replay_buffer_size: 4096
  max_training_time: 30600
  save_checkpoint_every_x_updates: 10

agent:
  batch_size: 512
  entropy_weight: 0.005
  epochs: 2
  epsilon: 0.1
  gamma: 0.985
  learning_rate: 0.001
  tau: 0.8
  use_meta_node: true  # include a meta node that is connected to every cell

env:
  allow_carts: false

reward:
  BUILD_CART: 0.05
  BUILD_CITY_TILE: 0.1
  BUILD_WORKER: 0.05
  CITY_AT_END: 1
  GAIN_RESEARCH_POINT: 0.01
  RESEARCH_COAL: 0.1
  RESEARCH_URANIUM: 0.5
  START_NEW_CITY: 0
  TURN: 0.01
  UNIT_AT_END: 0.1
  WIN: 1
  ZERO_SUM: false

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

Now you can create a dict with the hyperparameters

```python
config = {
    "seed": 505,
    "wandb": {
        "entity": "cerebro-ai",
        "project": "luxai21",
        "notes": "",
        "tags": ["GNNs"],
        "replay_every_x_updates": 10
    },
    "game": {
        "width": 12,
        "height": 12,
    },
    "env": {
        # if action_mask should allow the building of carts, also affects transfer to carts action
        "allow_carts": False,
        # TODO implement "allow_transfer"
    },
    "training": {
        # total games that should be played
        "max_games": 100000,
        "max_training_time": 30600,

        # will update model after replay_buffer size exceeds this 
        "max_replay_buffer_size": 4096,
        "save_checkpoint_every_x_updates": 10
    },
    "agent": {
        "learning_rate": 0.001,
        "gamma": 0.985,  # discount of future rewards
        "tau": 0.8,
        "batch_size": 512,
        "epsilon": 0.1,  # PPO clipping range
        "epochs": 2,
        "entropy_weight": 0.005
        "use_meta_node": True  # include a meta node that is connected to every cell
    },
    "reward": {
        "TURN": 0.01,  # base reward every turn

        # BUILDING
        # reward for every new piece, 
        # will also be applied negatively if units disappear
        "BUILD_CITY_TILE": 0.1,
        "BUILD_WORKER": 0.05,
        "BUILD_CART": 0.05,

        "START_NEW_CITY": 0,  # if the building of a city_tile leads to a new city

        # RESEARCH
        "GAIN_RESEARCH_POINT": 0.01,
        "RESEARCH_COAL": 0.1,  # reach enough research points for coal
        "RESEARCH_URANIUM": 0.5,

        # END
        "CITY_AT_END": 1,
        "UNIT_AT_END": 0.1,  # worker and cart
        "WIN": 1,

        # if true it will center the agent rewards around zero, and one agent will get a negative reward
        "ZERO_SUM": False
    },
}

```

and start training

```python
from luxai21.train_agent import train

train(config)
```