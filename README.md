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
``` 



Now you can create a dict with the hyperparameters
```python
import yaml

config = yaml.safe_load("""
---
training:
  learning_rate: 0.002
  gamma: 0.995
  gae_lambda: 0.95
  batch_size: 256
  step_count: 524288
  n_steps: 8192
  n_envs: 1

model:
  map_emb_dim: 128
  net_arch_shared_layers: [128]
  net_arch_pi: [64, 32] # policy-function
  net_arch_vf: [128, 64, 32] # value-function
""")
```

and start training

```python
from luxai21.train import train
from luxai21.config import Hyperparams

configurator = Hyperparams(config['training'], config['model']).load(config)

train(config)
```