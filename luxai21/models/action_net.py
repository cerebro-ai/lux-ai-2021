# from typing import Tuple
# from typing import Dict, List, Tuple, Type, Union
import torch
from stable_baselines3.common.torch_layers import MlpExtractor
from torch import nn
import torch as th

from luxai21.models.feature_extr import ACTION_SIZE


class CustomMlpExtractor(MlpExtractor):
    """
      Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
      a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
      of them are shared between the policy network and the value network. It is assumed to be a list with the following
      structure:

      1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
         If the number of ints is zero, there will be no shared layers.
      2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
         It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
         If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

      For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
      network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
      would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
      would be specified as [128, 128].

      Adapted from Stable Baselines.

      :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
      :param net_arch: The specification of the policy and value networks.
          See above for details on its formatting.
      :param activation_fn: The activation function to use for the networks.
      :param device:
      """

    def __init__(
            self,
            feature_dim: int,
            net_arch,  # removed: : List[Union[int, Dict[str, List[int]]]]
            activation_fn,  # removed:  Type[nn.Module]
            device="auto",  # removed: : Union[th.device, str]
    ):
        super(CustomMlpExtractor, self).__init__(
            feature_dim=feature_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            device=device
        )
        # net_arch[-1]["pi"][-1] # output dimension of pi network
        self.linear = nn.Linear(net_arch[-1]["pi"][-1], 9)  # TODO insert actual action_size

    def forward(self, features: th.Tensor):  # removed -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
        """

        action_mask = th.narrow(features, 1, features.shape[1] - ACTION_SIZE, ACTION_SIZE)

        embedding = th.narrow(features, 1, 0, features.shape[1] - ACTION_SIZE)

        shared_latent = self.shared_net(embedding)

        value = self.value_net(shared_latent)
        action_logits: th.Tensor = self.linear(self.policy_net(shared_latent))

        if ACTION_SIZE == 0:
            action_mask = torch.ones_like(action_logits, dtype=torch.bool)

        # set logits of illegal actions to -inf
        masked_action_logits = action_logits.masked_fill(~action_mask, float("-inf"))

        return action_logits, value


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
            self,
            feature_dim: int,
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor):  # removed -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        embedding = features[:-total_action_size]  # TODO should be replaced by th.narrow(observations, 1, MAP_SIZE, GAME_SIZE)
        action_mask = features[-total_action_size:]

        value = self.value_net(embedding)
        action_logits = self.policy_net(embedding)

        # set logits of illegal actions to -inf
        ids = th.where(action_mask == 0)
        action_logits[ids] = -float('inf')

        return action_logits, value
