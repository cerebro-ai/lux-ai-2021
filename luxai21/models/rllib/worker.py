import json
from typing import Dict, List

import gym
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from torch import TensorType, nn, Tensor

from luxai21.models.gnn.map_embedding import MapEmbeddingTower
from luxai21.models.gnn.utils import get_board_edge_index, batches_to_large_graph, large_graph_to_batches


class WorkerModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super(WorkerModel, self).__init__(obs_space,
                                          action_space, num_outputs,
                                          model_config, name)
        nn.Module.__init__(self)

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
                      out_features=self.config["policy_hidden_dim"]//2),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy_hidden_dim"]//2,
                      out_features=self.config["policy_output_dim"])
        )

        self.value_head_network = nn.Sequential(
            nn.Linear(in_features=self.config["map_embedding"]["output_dim"],
                      out_features=self.config["value_hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value_hidden_dim"],
                      out_features=self.config["value_hidden_dim"]//2),
            nn.ELU(),
            nn.Linear(in_features=self.config["value_hidden_dim"]//2,
                      out_features=1)
        )

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        map_tensor = input_dict["obs"]["map"]
        pos_tensor = input_dict["obs"]["pos"].float()
        action_mask = input_dict["obs"]["action_mask"].int()

        map_emb_flat = self.embed_map(map_tensor)
        self.map_emb_flat = map_emb_flat

        action_logits = self.action_logits(map_emb_flat, pos_tensor, action_mask)

        return action_logits, state

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
        return super(WorkerModel, self).to(device=device, *args)

    def import_from_h5(self, h5_file: str) -> None:
        pass
