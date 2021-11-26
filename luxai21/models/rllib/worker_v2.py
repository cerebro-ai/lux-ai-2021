from typing import Dict, List, Union

import gym
import numpy as np
import torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
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


class WorkerLSTMModelV2(RecurrentNetwork, nn.Module):
    """An LSTM wrapper serving as an interface for ModelV2s that set use_lstm.
    """

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):

        nn.Module.__init__(self)
        super(WorkerLSTMModelV2, self).__init__(obs_space,
                                                action_space, num_outputs,
                                                model_config, name)

        self.device = "cpu"
        self.config = model_config["custom_model_config"]
        self.use_meta_node = self.config["use_meta_node"]

        self.map_model = Representation(input_shape=(10, 12, 12),
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
            nn.Linear(in_features=self.config["lstm"]["hidden_size"],
                      out_features=self.config["policy"]["hidden_size_1"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy"]["hidden_size_1"],
                      out_features=self.config["policy"]["hidden_size_2"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy"]["hidden_size_2"],
                      out_features=self.config["policy"]["hidden_size_3"])
        )

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=self.config["lstm"]["hidden_size"],
                      out_features=self.config["value"]["hidden_size_1"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value"]["hidden_size_1"],
                      out_features=self.config["value"]["hidden_size_2"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value"]["hidden_size_2"],
                      out_features=1)
        )

        self.lstm = nn.LSTM(
            input_size=self.config["embedding_size"],
            hidden_size=self.config["lstm"]["hidden_size"],
            batch_first=True
        )

        # Postprocess LSTM output with another hidden layer and compute values.
        self.logits_head = SlimFC(
            in_size=self.config["policy"]["hidden_size_3"],
            out_size=self.config["policy"]["output_size"],
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_)

    @override(RecurrentNetwork)
    def forward(self, input_dict: Dict[str, Dict[str, Tensor]],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        global_map = input_dict["obs"]["map"] # 12x12x10
        game_state = input_dict["obs"]["game_state"]

        global_strategy = self.map_model(torch.permute(global_map, dims=(0, 3, 1, 2)))
        game_strategy = self.game_state_model(game_state)

        strategy = torch.cat([global_strategy, game_strategy], dim=1)

        features = self.feature_model(strategy)

        self.action_mask = input_dict["obs"]["action_mask"].int()

        flat_inputs = features.float()
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )

        action_mask = add_time_dimension(
            self.action_mask,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major
        )

        output, new_state = self.forward_rnn(inputs, state, seq_lens, action_mask)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs: TensorType, state: List[TensorType],
                    seq_lens: TensorType, action_mask: TensorType) -> (TensorType, List[TensorType]):
        '''
                self.lstm = nn.LSTM(
            self.config["policy_output_dim"], self.config["lstm_cell_size"], batch_first=False)

        >>> rnn = nn.LSTM(9, 256, 1)
        >>> input = torch.randn(32, 1, 9)
        >>> h0 = torch.randn(1, 1, 256)
        >>> c0 = torch.randn(1, 1, 256)

        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)

        '''

        self._features, [h, c] = self.lstm(
            inputs,
            [torch.unsqueeze(state[0], 0),
             torch.unsqueeze(state[1], 0)])

        policy_features = self.policy_branch(self._features)
        action_logits = self.logits_head(policy_features)
        masked_logits = self.mask_logits(action_logits, action_mask)

        return masked_logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        # Place hidden states on same device as model.
        linear = next(self.logits_head._model.children())
        h = [
            linear.weight.new(1, self.config["lstm"]["hidden_size"]).zero_().squeeze(0),
            linear.weight.new(1, self.config["lstm"]["hidden_size"]).zero_().squeeze(0)
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
        inf_mask = torch.maximum(torch.log(action_mask), torch.tensor(mask_value))
        logits_masked = logits + inf_mask
        return logits_masked

    def embed_map(self, map_tensor: Tensor, model: nn.Module):
        """
        Returns:
            flat_map_emb: Tensor of size [batches, nodes, features]
        """
        assert map_tensor.dim() == 4

        batches = map_tensor.size()[0]
        map_size_x = map_tensor.size()[1]
        map_size_y = map_tensor.size()[2]
        features = map_tensor.size()[3]

        assert map_size_x == map_size_y, f"Map is not quadratic: {map_tensor.size()}"

        map_flat = torch.reshape(map_tensor, (batches, -1, features))  # batch, nodes, features

        if self.use_meta_node:
            meta_node = torch.zeros((batches, 1, features)).to(self.device)
            map_flat = torch.cat([map_flat, meta_node], dim=1)

        # get edge_index from cache or compute new and cache
        if map_size_x in self.edge_index_cache:
            edge_index = self.edge_index_cache[map_size_x].to(self.device)
        else:
            edge_index = get_board_edge_index(map_size_x, map_size_y, self.use_meta_node).to(self.device)
            self.edge_index_cache[map_size_x] = edge_index

        x, large_edge_index, _ = batches_to_large_graph(map_flat, edge_index)
        x, large_edge_index = x.to(self.device), large_edge_index.to(self.device)
        large_map_emb_flat = model(x, large_edge_index)

        node_embeddings, _ = large_graph_to_batches(large_map_emb_flat, None, batches)

        # now aggregate over the whole graph
        map_embedding = torch.mean(node_embeddings, dim=1).squeeze(1)

        return map_embedding

    def to(self, device, *args):
        self.device = device
        return super(WorkerLSTMModelV2, self).to(device=device, *args)

    def import_from_h5(self, h5_file: str) -> None:
        pass
