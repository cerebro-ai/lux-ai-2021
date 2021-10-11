import configparser
from dataclasses import dataclass
import random
from typing import Union, List

import yaml
import pathlib

default_config = pathlib.Path(__file__).parent.joinpath('config.yaml')


@dataclass
class HyperparamsTraining:
    learning_rate: float
    gamma: float
    gae_lambda: float
    batch_size: int
    step_count: int
    n_steps: int
    n_envs: int
    path: Union[str, pathlib.Path] = None
    id: str = str(random.randint(0, 10000))


@dataclass
class HyperparamsModel:
    map_emb_dim: int
    net_arch_shared_layers: List[int]
    net_arch_pi: List[int]
    net_arch_vf: List[int]


@dataclass()
class Hyperparams:
    training: HyperparamsTraining
    model: HyperparamsModel

    @staticmethod
    def load(source: Union[dict, str]):
        if isinstance(source, dict):
            params = {
                "training": HyperparamsTraining(**source["training"]),
                "model": HyperparamsModel(**source["model"])
            }
            return Hyperparams(**params)
        else:
            if pathlib.Path(source).exists():
                with open(source, "r") as f:
                    params = yaml.safe_load(f)
                    params = {
                        "training": HyperparamsTraining(**params["training"]),
                        "model": HyperparamsModel(**params["model"])
                    }
                    return Hyperparams(**params)
            else:
                params = yaml.safe_load(source)
                params = {
                    "training": HyperparamsTraining(**params["training"]),
                    "model": HyperparamsModel(**params["model"])
                }
                return Hyperparams(**params)
