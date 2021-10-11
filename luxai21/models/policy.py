# from collections import Callable
# from typing import Optional, List, Union, Dict, Type

import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from luxai21.models.action_net import CustomMlpExtractor


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule,  # removed Callable[[float], float],
            net_arch=None,  # removed: : Optional[List[Union[int, Dict[str, List[int]]]]]
            activation_fn=nn.Tanh,
            *args,
            **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

        # We overwrite the action net of sb3 so that we can use our own action net in models/action_net.py
        # This is a workaround for masked_actions
        self.action_net = nn.Identity()


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
