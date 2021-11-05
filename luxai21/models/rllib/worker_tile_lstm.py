import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from typing import Dict, List, Union

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from torch import nn, Tensor, TensorType
import torch

from luxai21.models.gnn.map_embedding import MapEmbeddingTower
from luxai21.models.gnn.utils import get_board_edge_index, batches_to_large_graph, large_graph_to_batches


class WorkerLSTMModel(RecurrentNetwork, nn.Module):
    """An LSTM wrapper serving as an interface for ModelV2s that set use_lstm.
    """

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):

        nn.Module.__init__(self)
        super(WorkerLSTMModel, self).__init__(obs_space,
                                          action_space, num_outputs,
                                          model_config, name)

        self.device = "cpu"
        self.config = model_config["custom_model_config"]
        self.use_meta_node = self.config["use_meta_node"]

        self.map_emb_model = MapEmbeddingTower(**self.config["map_embedding"])
        self.map_emb_flat = None  # caching map embedding

        self.edge_index_cache = {}

        self.policy_head_network = nn.Sequential(
            nn.Linear(in_features=self.config["map_embedding"]["output_dim"],
                      out_features=self.config["policy_hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy_hidden_dim"],
                      out_features=self.config["policy_hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy_hidden_dim"],
                      out_features=self.config["policy_output_dim"])
        )

        self.value_head_network = nn.Sequential(
            nn.Linear(in_features=self.config["map_embedding"]["output_dim"],
                      out_features=self.config["value_hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value_hidden_dim"],
                      out_features=self.config["value_hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value_hidden_dim"],
                      out_features=1),
            nn.Tanh()
        )
        # Define actual LSTM layer (with num_outputs being the nodes coming
        # from the wrapped (underlying) layer).
        self.lstm = nn.LSTM(
            self.config["policy_output_dim"], self.config["lstm_cell_size"], batch_first=True)

        # Postprocess LSTM output with another hidden layer and compute values.
        self._logits_branch = SlimFC(
            in_size=self.config["lstm_cell_size"],
            out_size=self.config["policy_output_dim"],
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_)
        # self._value_branch = SlimFC(
        #     in_size=self.cell_size,
        #     out_size=1,
        #     activation_fn=None,
        #     initializer=torch.nn.init.xavier_uniform_)

    @override(RecurrentNetwork)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        map_tensor = input_dict["obs"]["map"]
        pos_tensor = input_dict["obs"]["pos"].float()
        self.pos_tensor = pos_tensor
        action_mask = input_dict["obs"]["action_mask"].int()

        map_emb_flat = self.embed_map(map_tensor)
        self.map_emb_flat = map_emb_flat

        action_logits = self.action_logits(map_emb_flat, pos_tensor, action_mask)

        lstm_input_dict = {'obs_flat': action_logits}

        return super().forward(lstm_input_dict, state, seq_lens)

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs: TensorType, state: List[TensorType],
                    seq_lens: TensorType) -> (TensorType, List[TensorType]):
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

        model_out = self._logits_branch(self._features)
        return model_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(ModelV2)
    def get_initial_state(self) -> Union[List[np.ndarray], List[TensorType]]:
        # Place hidden states on same device as model.
        linear = next(self._logits_branch._model.children())
        h = [
            linear.weight.new(1, self.config["lstm_cell_size"]).zero_().squeeze(0),
            linear.weight.new(1, self.config["lstm_cell_size"]).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self) -> TensorType:
        if self.use_meta_node:
            meta_node_state = self.map_emb_flat[:, -1, :]
        else:
            # aggregate over all nodes
            meta_node_state = torch.mean(self.map_emb_flat, dim=1)
        value = self.value_head_network(meta_node_state)
        return value.squeeze(1)

    def embed_map(self, map_tensor: Tensor):
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
        large_map_emb_flat = self.map_emb_model(x, large_edge_index)

        map_emb_flat, _ = large_graph_to_batches(large_map_emb_flat, None, batches)
        return map_emb_flat

    def action_logits(self, map_emb_flat, worker_pos, action_mask):
        batches = map_emb_flat.size()[0]

        j_h = torch.Tensor([12, 1]).unsqueeze(0).repeat(batches, 1).to(self.device)
        j = torch.sum(worker_pos * j_h, 1).long()
        indices = j[..., None, None].expand(-1, 1, map_emb_flat.size(2))
        cell_state = torch.gather(map_emb_flat, dim=1, index=indices).squeeze(1)

        logits = self.policy_head_network(cell_state)
        mask_value = torch.finfo(logits.dtype).min
        inf_mask = torch.maximum(torch.log(action_mask), torch.tensor(mask_value))
        logits_masked = logits + inf_mask
        return logits_masked

    def to(self, device, *args):
        self.device = device
        return super(WorkerLSTMModel, self).to(device=device, *args)

    def import_from_h5(self, h5_file: str) -> None:
        pass
