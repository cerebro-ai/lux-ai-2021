from typing import Union, List, Optional, Tuple, Dict

import numpy as np
import torch
import gym.spaces as spaces
from ray.rllib import Policy
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import ModelWeights
from torch import TensorType, Tensor


class BasicCityTilePolicy(Policy):
    """Simple hardcoded citytile policy

    Perform the first valid action given the action mask in this particular order:
    - build worker
    - build cart
    - research
    - do nothing

    """

    def get_weights(self) -> ModelWeights:
        pass

    def set_weights(self, weights: ModelWeights) -> None:
        pass

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(
            self,
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorType], TensorType] = None,
            prev_reward_batch: Union[List[TensorType], TensorType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> \
            Tuple[Tensor, List[TensorType], Dict[str, TensorType]]:
        obs = restore_original_dimensions(torch.tensor(obs_batch), self.observation_space, "torch")

        actions = []
        for action_mask in obs["action_mask"]:
            if action_mask[1]:
                actions.append(1)
            elif action_mask[2]:
                actions.append(2)
            elif action_mask[3]:
                actions.append(3)
            else:
                actions.append(0)
        actions = torch.tensor(actions)
        return actions, state_batches, {}


EagerCityTilePolicy = PolicySpec(
    policy_class=BasicCityTilePolicy,
    action_space=spaces.Discrete(4),
    observation_space=spaces.Dict(
        **{'map': spaces.Box(shape=(12, 12, 10),
                             dtype=np.float64,
                             low=-float('inf'),
                             high=float('inf')
                             ),
           'game_state': spaces.Box(shape=(3,),
                                    dtype=np.float64,
                                    low=float('-inf'),
                                    high=float('inf')
                                    ),
           'type': spaces.Discrete(3),
           'pos': spaces.Box(shape=(2,),
                             dtype=np.float64,
                             low=0,
                             high=99999
                             ),
           'action_mask': spaces.Box(shape=(4,),
                                     dtype=np.float64,
                                     low=0,
                                     high=1
                                     )}),
    config={}
)
