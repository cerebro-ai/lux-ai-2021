from typing import Dict, List, Union

import gym
import numpy as np
import torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn, Tensor, TensorType

from luxai21.models.gnn.map_embedding import MapEmbeddingTower
from luxai21.models.gnn.utils import get_board_edge_index, batches_to_large_graph, large_graph_to_batches

# Small neural nets with PyTorch

import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        # TODO scale residual with small constant (0.1/0.2)
        return F.relu(x + (self.conv(x)))


class Representation(nn.Module):
    ''' Conversion from observation to inner abstract state '''

    def __init__(self, input_shape, num_filters, num_blocks):
        super().__init__()
        self.input_shape = input_shape
        self.board_size = self.input_shape[1] * self.input_shape[2]

        self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
        self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

    def forward(self, x):
        h = F.relu(self.layer0(x))
        for block in self.blocks:
            h = block(h)
        # h = (b, filters, 12, 12)
        h, _ = torch.max(torch.reshape(h, (h.size(0), h.size(1), -1)), dim=2)
        return h

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            rp = self(x)
        return rp


class WorkerModelV3(TorchModelV2, nn.Module):

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):

        nn.Module.__init__(self)
        super(WorkerModelV3, self).__init__(obs_space,
                                            action_space, num_outputs,
                                            model_config, name)

        self.device = "cpu"
        self.config = model_config["custom_model_config"]
        self.use_meta_node = self.config["use_meta_node"]

        self.map_model = Representation(input_shape=(30, 12, 12),
                                        num_filters=self.config["map"]["num_filters"],
                                        num_blocks=self.config["map"]["num_blocks"]
                                        )

        self.game_state_model = nn.Sequential(
            nn.Linear(in_features=self.config["game_state_model"]["input_dim"],
                      out_features=self.config["game_state_model"]["hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["game_state_model"]["hidden_dim"],
                      out_features=self.config["game_state_model"]["output_dim"]),
        )

        self.feature_input_size = self.config["map"]["num_filters"] \
                                  + self.config["game_state_model"]["output_dim"]

        self.feature_model = nn.Sequential(
            nn.Linear(in_features=self.feature_input_size,
                      out_features=self.config["embedding_size"])
        )

        self.edge_index_cache = {}

        self.policy_branch = nn.Sequential(
            nn.Linear(in_features=self.config["embedding_size"],
                      out_features=self.config["policy"]["hidden_size_1"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy"]["hidden_size_1"],
                      out_features=self.config["policy"]["hidden_size_2"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy"]["hidden_size_2"],
                      out_features=self.config["policy"]["hidden_size_3"])
        )

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=self.config["embedding_size"],
                      out_features=self.config["value"]["hidden_size_1"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value"]["hidden_size_1"],
                      out_features=self.config["value"]["hidden_size_2"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value"]["hidden_size_2"],
                      out_features=1)
        )

        # Postprocess LSTM output with another hidden layer and compute values.
        self.logits_head = SlimFC(
            in_size=self.config["policy"]["hidden_size_3"],
            out_size=self.config["policy"]["output_size"],
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_)

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, Dict[str, Tensor]],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        global_map = input_dict["obs"]["map"] # 12x12x10
        game_state = input_dict["obs"]["game_state"]

        global_strategy = self.map_model(torch.permute(global_map, dims=(0, 3, 1, 2)))
        game_strategy = self.game_state_model(game_state)

        strategy = torch.cat([global_strategy, game_strategy], dim=1)

        features = self.feature_model(strategy)
        self._features = features

        self.action_mask = input_dict["obs"]["action_mask"].int()


        policy_features = self.policy_branch(self._features)
        action_logits = self.logits_head(policy_features)
        masked_logits = self.mask_logits(action_logits, self.action_mask)

        return masked_logits, state

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        # Place hidden states on same device as model.
        linear = next(self.logits_head._model.children())
        h = [
            linear.weight.new(1, self.config["embedding_size"]).zero_().squeeze(0),
            linear.weight.new(1, self.config["embedding_size"]).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self) -> TensorType:
        features = torch.reshape(self._features, [-1, self._features.size()[-1]])
        value = self.value_branch(features)
        value = value.squeeze(1)
        return value

    def mask_logits(self, logits, action_mask):
        mask_value = torch.finfo(logits.dtype).min

        # shorten the last dimension of action_mask
        action_mask = torch.narrow(action_mask, action_mask.dim() - 1, 0, logits.size()[-1])
        inf_mask = torch.maximum(torch.log(action_mask), torch.tensor(mask_value))

        logits_masked = logits + inf_mask
        return logits_masked

    def to(self, device, *args):
        self.device = device
        return super(WorkerModelV3, self).to(device=device, *args)

    def import_from_h5(self, h5_file: str) -> None:
        pass
