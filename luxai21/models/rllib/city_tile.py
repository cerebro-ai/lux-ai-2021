from typing import Dict, List

import gym
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from torch import TensorType, nn


class BasicCityTileModel(TorchModelV2, nn.Module):

    def import_from_h5(self, h5_file: str) -> None:
        pass

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super(BasicCityTileModel, self).__init__(obs_space,
                                                 action_space, num_outputs,
                                                 model_config, name)
        nn.Module.__init__(self)

        self.value = None

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        action_mask = input_dict["obs"]["action_mask"]  # array [BATCH, ACTION_MASK]

        #  spawn worker
        action_logits = []
        mask_value = torch.finfo(torch.float32).min

        for batch in action_mask:
            l = [mask_value, mask_value, mask_value, mask_value]
            if batch[1]:
                l[1] = 1
            elif batch[2]:
                l[2] = 1
            elif batch[3]:
                l[3] = 1
            else:
                l[0] = 1
            action_logits.append(torch.tensor(l))

        action_logits = torch.cat(action_logits, dim=0)

        self.value = torch.zeros((action_logits.size()[0], 1))
        return action_logits

    def value_function(self) -> TensorType:
        return self.value
