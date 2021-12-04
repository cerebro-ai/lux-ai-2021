import json
import math
from typing import Dict, List

import gym
import numpy as np
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from torch import TensorType, nn, Tensor
from torch.nn.functional import pad

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

        self.embed_directions = nn.Sequential(
            nn.Linear(in_features=self.config["map_embedding"]["output_dim"] * 5,
                      out_features=self.config["directions"]["hidden_1"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["directions"]["hidden_1"],
                      out_features=self.config["directions"]["hidden_1"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["directions"]["hidden_1"],
                      out_features=self.config["directions"]["output_dim"]),
            nn.ELU(),
        )

        self.policy_head_network = nn.Sequential(
            nn.Linear(in_features=self.config["directions"]["output_dim"],
                      out_features=self.config["policy_hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy_hidden_dim"],
                      out_features=self.config["policy_hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["policy_hidden_dim"],
                      out_features=self.config["policy_output_dim"])
        )

        self.value_head_network = nn.Sequential(
            nn.Linear(in_features=self.config["directions"]["output_dim"],
                      out_features=self.config["value_hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value_hidden_dim"],
                      out_features=self.config["value_hidden_dim"]),
            nn.ELU(),
            nn.Linear(in_features=self.config["value_hidden_dim"],
                      out_features=1)
        )
        self.features = None

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        pos_tensor = input_dict["obs"]["pos"].float()[:, :2]

        maps = self.unpad_maps(input_dict)  # [ (12,12,30), (16,16,30) ]

        directions_embedded = self.embed_different_sized_maps(maps, pos_tensor)  # (batches, [north|west|south|central])

        self.features = self.embed_directions(directions_embedded)

        action_mask = input_dict["obs"]["action_mask"].int()

        policy_logits = self.policy_head_network(self.features)

        action_logits = self.mask_logits(policy_logits, action_mask)

        return action_logits, state

    def embed_different_sized_maps(self, unpadded_maps: List[torch.Tensor], pos_tensor):
        break_indices = [0]
        for i in range(1, len(unpadded_maps)):
            if unpadded_maps[i].size()[0] != unpadded_maps[i - 1].size()[0]:
                break_indices.append(i)

        break_indices.append(len(unpadded_maps))

        unit_states = []
        for j in range(1, len(break_indices)):
            g_map = torch.stack(unpadded_maps[break_indices[j - 1]: break_indices[j]], dim=0)

            map_embedded = self.embed_map(g_map)  # (nodes, node_emb_features)
            pos_section = pos_tensor[break_indices[j - 1]:break_indices[j], :]

            # pick N, W, S, E
            state = self.pick_from_map(map_embedded, pos_section)

            unit_states.append(state)

        state = torch.cat(unit_states)
        return state

    def unpad_maps(self, input_dict) -> List[torch.Tensor]:
        map_sizes = input_dict["obs"]["map_size"][:, 0]

        unpadded_maps = []
        for i, map_size in enumerate(map_sizes.tolist()):
            if map_size == 0:  # only needed for test run
                map_size = 16
            pad = int((input_dict["obs"]["map"][i, :, :, :].size()[0] - map_size) // 2)
            if pad == 0:
                u_map = input_dict["obs"]["map"][i, :, :, :]
            else:
                u_map = input_dict["obs"]["map"][i, :, :, :][pad:-pad, pad:-pad, :]
            unpadded_maps.append(u_map)
        return unpadded_maps

    def value_function(self) -> TensorType:
        value = self.value_head_network(self.features)
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

        # throw away the meta node
        map_emb_flat = map_emb_flat[:, :-1, :]

        return map_emb_flat

    def pick_from_map(self, map_emb_flat, pos):
        """
        map_emb_flat: (30, 145, 128) // (batches, nodes, features)
        """
        assert map_emb_flat.shape[0] == pos.shape[0], "map and pos dont have equal batches"
        batches = map_emb_flat.size()[0]
        nodes = map_emb_flat.shape[1]
        map_size = int(math.sqrt(nodes))

        # pad map with zero
        map_emb = torch.reshape(map_emb_flat, (batches, map_size, map_size, map_emb_flat.shape[-1]))
        # map_emb (b, 12, 12, 32)
        map_emb_padded = pad(map_emb, (0, 0, 1, 1, 1, 1))
        map_emb_padded_flat = map_emb_padded.view(batches, -1, map_emb_flat.shape[-1])

        j_h = torch.Tensor([map_size, 1]).unsqueeze(0).repeat(batches, 1).to(self.device)
        j = torch.sum(pos * j_h, 1).long()

        cell_states = []
        #               NORTH,  EAST      SOUTH   WEST    CENTRAL
        for offset in [(-1, 0), (0, -1), (1, 0), (0, 1), (0, 0)]:
            offset = np.array(offset)
            o_pos = pos + offset + np.array(
                (1, 1))  # we have to correct the position due to the padding
            j = torch.sum(o_pos * j_h, 1).long()
            indices = j[..., None, None].expand(-1, 1, map_emb_padded_flat.size(2))
            cell_state = torch.gather(map_emb_padded_flat, dim=1, index=indices).squeeze(1)
            cell_states.append(cell_state)

        state = torch.cat(cell_states, dim=1)
        return state

    def mask_logits(self, logits, action_mask):
        mask_value = torch.finfo(logits.dtype).min

        # shorten the last dimension of action_mask
        action_mask = torch.narrow(action_mask, action_mask.dim() - 1, 0, logits.size()[-1])
        inf_mask = torch.maximum(torch.log(action_mask), torch.tensor(mask_value))

        logits_masked = logits + inf_mask
        return logits_masked

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
