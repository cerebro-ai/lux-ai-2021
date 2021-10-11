# lux-ai-2021
Lux AI 2021 Competition

Code for the Lux AI Reinforcement Challenge

- Kaggle Overview: https://www.kaggle.com/c/lux-ai-2021/overview
- Specs: https://www.lux-ai.org/specs-2021
- Visualize: https://2021vis.lux-ai.org/


## Set up kaggle notebook

Install the required packages
```shell
!pip install git+https://<username>:<token or password>@github.com/cerebro-ai/lux-ai-2021.git
!pip install git+https://<username>:<token or password>@github.com/cerebro-ai/lux-python-env.git
!pip install wandb --upgrade
``` 

Set API Key for Weights & Biases (Metrics logging)
```python
import os

os.environ["WANDB_API_KEY"] = "your api key"
# optional
os.environ["WANDB_NOTES"] = "Smaller learning rate, larger batch size"

```


Now you can create a dict with the hyperparameters
```python
import yaml

config_dict = yaml.safe_load("""---
wandb:
  entity: cerebro-ai
  project: luxai21
  # supporting all attributes: https://docs.wandb.ai/ref/python/init 

training:
  learning_rate: 0.001
  gamma: 0.995
  gae_lambda: 0.95
  batch_size: 128 # testing gradients
  step_count: 8388608
  n_steps: 8192
  n_envs: 1  # Number of parallel environments to use in training

model:
  map_emb_dim: 128
  # TODO insert feature_extractor parameters
  net_arch_shared_layers: [128]
  net_arch_pi: [64, 32] # policy-function
  net_arch_vf: [128, 64, 32] # value-function
  lstm_config:
    # input_size is implicitly given through the map_emb_dim
    # batch_first is also always true
    hidden_size: 128
    num_layers: 4
    # dropout: 0.2
    # supports all attributes of pytorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
""")
```

and start training

```python
from luxai21.train import train
from luxai21.config import Hyperparams

config = Hyperparams.load(config_dict)

train(config)
```