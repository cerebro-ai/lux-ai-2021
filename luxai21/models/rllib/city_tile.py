from typing import Dict, List, T, Union, Optional, Tuple

import torch
from ray.rllib import Policy
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import TensorType, nn

from luxai21.models.gnn.map_embedding import MapEmbeddingTower
from luxai21.models.gnn.utils import get_board_edge_index, batches_to_large_graph, large_graph_to_batches


class BasicCityTileModel(TorchModelV2, nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        action_mask = input_dict["obs"]["action_mask"]

        #  spawn worker
        actions = []
        for batch in action_mask:
            if batch[1]:
                actions.append(1)
            elif batch[2]:
                actions.append(2)
            elif batch[3]:
                actions.append(3)
            else:
                actions.append(0)

        return torch.Tensor(actions)


class BasicCityTilePolicy(Policy):
    """Simple hardcoded citytile policy

    Perform the first valid action given the action mask in this particular order:
    - build worker
    - build cart
    - research
    - do nothing

    """

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
            **kwargs):

        #  spawn worker
        actions = []
        for batch in obs_batch:
            action_mask = batch["action_mask"]
            if action_mask[1]:
                actions.append(1)
            elif action_mask[2]:
                actions.append(2)
            elif action_mask[3]:
                actions.append(3)
            else:
                actions.append(0)

        return actions, [], {}
