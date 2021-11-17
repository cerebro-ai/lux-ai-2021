import numpy as np
from gym.spaces import Discrete, Dict, Box
from omegaconf import OmegaConf
from ray.rllib.policy.policy import PolicySpec


def get_worker_policy(config):
    return PolicySpec(
        action_space=Discrete(9),
        observation_space=Dict(
            **{'map': Box(shape=(12, 12, 10),
                          dtype=np.float64,
                          low=-float('inf'),
                          high=float('inf')
                          ),
               'game_state': Box(shape=(3,),
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
                **OmegaConf.to_container(config.model.worker)
            }
        })
