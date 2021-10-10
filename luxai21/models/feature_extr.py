import math

import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
from luxai21.models.base_nets.InceptionNet import InceptionNet_v1

# TODO Update
MAP_PLANES = 18
MAP_SIZE = 32 * 32 * MAP_PLANES
GAME_SIZE = 22
UNIT_SIZE = 3
ACTION_SIZE = 12

TOTAL_SIZE = MAP_SIZE + GAME_SIZE + UNIT_SIZE + ACTION_SIZE


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, map_emb_dim: int = 256):
        super(CustomFeatureExtractor, self).__init__(observation_space, map_emb_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        assert observation_space.shape[0] == TOTAL_SIZE
        self.map_height = math.sqrt(MAP_SIZE // MAP_PLANES)
        self.map_emb_dim = map_emb_dim

        self.cnn = InceptionNet_v1(MAP_PLANES)

        self.linear = nn.Sequential(
            nn.Linear(self.cnn.output_size, map_emb_dim),
            nn.ReLU()
        )

    @property
    def features_dim(self) -> int:
        return self.map_emb_dim + GAME_SIZE + UNIT_SIZE  # + ACTION_SIZE @rkstgr schau mal aber ich glaube das muss hier raus

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # get the first part which is the map flattened

        map_flattened: th.Tensor = th.narrow(observations, 1, 0, MAP_SIZE)
        game_state = th.narrow(observations, 1, MAP_SIZE, GAME_SIZE)
        unit_state = th.narrow(observations, 1, MAP_SIZE + GAME_SIZE, UNIT_SIZE)
        action_mask = th.narrow(observations, 1, MAP_SIZE + GAME_SIZE + UNIT_SIZE, ACTION_SIZE)

        map_state = map_flattened.view((observations.shape[0], int(MAP_PLANES), int(self.map_height), int(self.map_height)))

        map_embedding = self.linear(self.cnn(map_state))

        features = torch.cat((map_embedding, game_state, unit_state, action_mask), dim=1)

        return features


if __name__ == '__main__':
    from torchsummary import summary
    extractor = CustomFeatureExtractor(gym.spaces.Box(0, 1, (TOTAL_SIZE,)), 128)
    print(summary(extractor.cnn, (MAP_PLANES, 32, 32)))