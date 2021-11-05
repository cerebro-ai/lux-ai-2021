import numpy as np
from gym.spaces import Discrete, Dict, Box
from ray.rllib.policy.policy import PolicySpec

from luxai21.models.rllib.city_tile import BasicCityTilePolicy


def get_city_tile_policy():
    return PolicySpec(
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
                                  )})
    )