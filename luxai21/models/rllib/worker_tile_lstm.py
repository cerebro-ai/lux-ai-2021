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

        self.map_emb_model = MapEmbeddingTower(**self.config["gnn"])
        self.map_emb_flat = None  # caching map embedding

        self.edge_index_cache = {}

        self.policy_branch = nn.Sequential(
            nn.Linear(in_features=self.config["lstm"]["hidden_size"],
                      out_features=self.config["policy"]["hidden_size"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy"]["hidden_size"],
                      out_features=self.config["policy"]["hidden_size"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy"]["hidden_size"],
                      out_features=self.config["policy"]["hidden_size"])
        )

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=self.config["lstm"]["hidden_size"],
                      out_features=self.config["value"]["hidden_size"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value"]["hidden_size"],
                      out_features=self.config["value"]["hidden_size"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value"]["hidden_size"],
                      out_features=1)
        )

        self.lstm = nn.LSTM(
            input_size=self.config["gnn"]["output_dim"],
            hidden_size=self.config["lstm"]["hidden_size"],
            batch_first=True
        )

        # Postprocess LSTM output with another hidden layer and compute values.
        self.logits_head = SlimFC(
            in_size=self.config["policy"]["hidden_size"],
            out_size=self.config["policy"]["output_size"],
            activation_fn=None,
            initializer=torch.nn.init.xavier_uniform_)

    @override(RecurrentNetwork)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        map_tensor = input_dict["obs"]["map"]
        pos_tensor = input_dict["obs"]["pos"].float()
        self.action_mask = input_dict["obs"]["action_mask"].int()

        map_emb_flat = self.embed_map(map_tensor)
        self.map_emb_flat = map_emb_flat

        piece_map_emb = self.pick_by_position(map_emb_flat, pos_tensor)

        flat_inputs = piece_map_emb.float()
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
        features = torch.reshape(self._features, [-1, self.config["lstm"]["hidden_size"]])
        value = self.value_branch(features).squeeze(1)
        return value

    def pick_by_position(self, map_emb_flat, pos_tensor):
        """
        Pick the corresponding node from the map_emb_flat on the specific position

        Args:
            map_emb_flat: [B, N, F] Tensor
            pos_tensor: [B, 2] Tensor

        Returns:
            piece_map_emb: [B, F]
        """
        batches = map_emb_flat.size()[0]

        j_h = torch.Tensor([12, 1]).unsqueeze(0).repeat(batches, 1).to(self.device)
        j = torch.sum(pos_tensor * j_h, 1).long()
        indices = j[..., None, None].expand(-1, 1, map_emb_flat.size(2))
        piece_map_emb = torch.gather(map_emb_flat, dim=1, index=indices).squeeze(1)
        return piece_map_emb

    def mask_logits(self, logits, action_mask):
        mask_value = torch.finfo(logits.dtype).min
        inf_mask = torch.maximum(torch.log(action_mask), torch.tensor(mask_value))
        logits_masked = logits + inf_mask
        return logits_masked

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

        logits = self.policy_branch(cell_state)
        mask_value = torch.finfo(logits.dtype).min
        inf_mask = torch.maximum(torch.log(action_mask), torch.tensor(mask_value))
        logits_masked = logits + inf_mask
        return logits_masked

    def to(self, device, *args):
        self.device = device
        return super(WorkerLSTMModel, self).to(device=device, *args)

    def import_from_h5(self, h5_file: str) -> None:
        pass
