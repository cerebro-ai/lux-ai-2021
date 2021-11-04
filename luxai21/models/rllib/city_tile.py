from typing import Dict, List, T, Union, Optional, Tuple

import gym
import torch
from ray.rllib import Policy, SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import ModelWeights, ModelConfigDict
from torch import TensorType, nn

from luxai21.models.gnn.map_embedding import MapEmbeddingTower
from luxai21.models.gnn.utils import get_board_edge_index, batches_to_large_graph, large_graph_to_batches


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


class BasicCityTilePolicy(Policy):
    """Simple hardcoded citytile policy

    Perform the first valid action given the action mask in this particular order:
    - build worker
    - build cart
    - research
    - do nothing

    """

    @override(Policy)
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

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
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        pass

    def compute_actions_from_input_dict(
            self,
            input_dict: SampleBatch,
            explore: bool = None,
            timestep: Optional[int] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        #  spawn worker

        dict_or_tuple_obs = restore_original_dimensions(torch.tensor(input_dict["obs"]), self.observation_space,
                                                        "torch")

        actions = []
        for action_mask in dict_or_tuple_obs["action_mask"]:
            if action_mask[1]:
                actions.append(1)
            elif action_mask[2]:
                actions.append(2)
            elif action_mask[3]:
                actions.append(3)
            else:
                actions.append(0)
        actions = torch.tensor(actions).unsqueeze(-1)
        return actions, torch.tensor([]), {}
