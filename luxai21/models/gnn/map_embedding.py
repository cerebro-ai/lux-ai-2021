from functools import partial

import torch.nn.functional
from torch import nn, Tensor
from torch_geometric.nn import GATv2Conv, GCNConv, GatedGraphConv, TransformerConv


class MapEmbeddingBlock(nn.Module):

    def __init__(self, gnn_module, hidden_dim, activation, **kwargs):
        super(MapEmbeddingBlock, self).__init__()

        self.gnn = gnn_module(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              **kwargs
                              )
        self.activation = activation

    def forward(self, map_tensor, edge_index):
        x = self.gnn(map_tensor, edge_index)
        x = self.activation(x)
        return x


class MapEmbeddingTower(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(MapEmbeddingTower, self).__init__()

        self.output_dim = output_dim

        self.activation = torch.nn.ELU()

        self.layer1 = GATv2Conv(in_channels=input_dim,
                                out_channels=hidden_dim,
                                **kwargs
                                )

        self.layer2 = GATv2Conv(in_channels=hidden_dim,
                                out_channels=hidden_dim,
                                **kwargs
                                )

        self.layer3 = GATv2Conv(in_channels=hidden_dim,
                                out_channels=hidden_dim,
                                **kwargs
                                )

        # self.blocks = [
        #     MapEmbeddingBlock(gnn_module=GNN,
        #                       hidden_dim=hidden_dim,
        #                       activation=self.activation,
        #                       **kwargs
        #                       )
        #     for _ in range(num_layers - 2)
        # ]

        self.last_layer = GCNConv(in_channels=3 * hidden_dim + input_dim,
                                  out_channels=output_dim,
                                  **kwargs
                                  )

    def forward(self, map_tensor: Tensor, edge_index: Tensor):
        # TODO implement skip connections

        out1 = self.layer1(map_tensor, edge_index)
        out1 = self.activation(out1)

        out2 = self.layer2(out1, edge_index)
        out2 = self.activation(out2)

        out3 = self.layer2(out2, edge_index)
        out3 = self.activation(out3)

        out_cat = torch.cat([map_tensor, out1, out2, out3], dim=1)

        x = self.last_layer(out_cat, edge_index)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    from utils import get_board_edge_index, batches_to_large_graph, large_graph_to_batches

    edge_index = get_board_edge_index(12, 12, True)
    tower = MapEmbeddingTower(
        input_dim=19,
        hidden_dim=36,
        output_dim=128,
    )

    map_tensor = torch.rand((2, 145, 19))
    x, e, b = batches_to_large_graph(map_tensor, edge_index)

    y = tower(x, e)

    map_emb, ee = large_graph_to_batches(y, e, b)

    print(map_emb.size())

    assert torch.all(edge_index == ee)

    # towerforward = tower.forward
    #
    #
    # def forward(*x):
    #     return towerforward(x[0][0], edge_index)
    #
    #
    # tower.forward = forward
    #
    # summary(tower, (145, 19), batch_size=1)
