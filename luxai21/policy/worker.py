from typing import Union, List, Optional, Tuple, Dict

import numpy as np
import torch
import gym.spaces as spaces
from omegaconf import OmegaConf
from ray.rllib import Policy
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import ModelWeights
from torch import TensorType, Tensor


def get_worker_policy(config):
    return PolicySpec(
        action_space=spaces.Discrete(9),
        observation_space=spaces.Dict(
            **{'map': spaces.Box(shape=(32, 32, 30),
                                 dtype=np.float64,
                                 low=-float('inf'),
                                 high=float('inf')
                                 ),
               'game_state': spaces.Box(shape=(6,),
                                        dtype=np.float64,
                                        low=float('-inf'),
                                        high=float('inf')
                                        ),
               'map_size': spaces.Box(shape=(3,),
                                      dtype=np.float64,
                                      low=0,
                                      high=100),
               'pos': spaces.Box(shape=(6,),
                                 dtype=np.float64,
                                 low=0,
                                 high=99999
                                 ),
               'action_mask': spaces.Box(shape=(27,),
                                         dtype=np.float64,
                                         low=0,
                                         high=1
                                         )}),
        config={
            "model": {
                **OmegaConf.to_container(config.model.worker)
            }
        })


class DoNothingWorkerPolicy(Policy):
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
        if isinstance(obs_batch, np.ndarray):
            batches = obs_batch.shape[0]
        elif isinstance(obs_batch, torch.Tensor):
            batches = obs_batch.size()[0]
        actions = torch.zeros((batches,)).int()
        return actions, state_batches, {}


def get_do_nothing_worker_policy():
    return PolicySpec(
        policy_class=DoNothingWorkerPolicy,
        action_space=spaces.Discrete(9),
        observation_space=spaces.Dict(
            **{'map': spaces.Box(shape=(32, 32, 30),
                                 dtype=np.float64,
                                 low=-float('inf'),
                                 high=float('inf')
                                 ),
               'game_state': spaces.Box(shape=(6,),
                                        dtype=np.float64,
                                        low=float('-inf'),
                                        high=float('inf')
                                        ),
               'map_size': spaces.Box(shape=(3,),
                                      dtype=np.float64,
                                      low=0,
                                      high=100),
               'pos': spaces.Box(shape=(6,),
                                 dtype=np.float64,
                                 low=0,
                                 high=99999
                                 ),
               'action_mask': spaces.Box(shape=(27,),
                                         dtype=np.float64,
                                         low=0,
                                         high=1
                                         )}))
