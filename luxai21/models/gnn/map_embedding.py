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

    def __init__(self, gnn_module, input_dim, hidden_dim, output_dim, num_layers, activation, **kwargs):
        super(MapEmbeddingTower, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation

        GNN = gnn_module

        self.layer1 = GNN(in_channels=self.input_dim,
                          out_channels=self.hidden_dim,
                          **kwargs
                          )

        self.blocks = [
            MapEmbeddingBlock(gnn_module=GNN,
                              hidden_dim=hidden_dim,
                              activation=self.activation,
                              **kwargs
                              )
            for _ in range(num_layers - 2)
        ]

        self.last_layer = GNN(in_channels=hidden_dim,
                              out_channels=output_dim,
                              **kwargs
                              )

    def forward(self, map_tensor: Tensor, edge_index: Tensor):
        x = self.layer1(map_tensor, edge_index)
        x = self.activation(x)

        for block in self.blocks:
            x = block(x, edge_index)

        x = self.last_layer(x, edge_index)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    from utils import get_board_edge_index

    edge_index = get_board_edge_index(12, 12, False)

    gnn = GatedGraphConv(out_channels=64,
                         num_layers=4)

    y = gnn(torch.rand((144, 19)), edge_index)
    print(y.size())

    tower = MapEmbeddingTower(
        gnn_module=gnn,
        input_dim=19,
        hidden_dim=64,
        output_dim=64,
        num_layers=20,
        activation=torch.nn.functional.elu
    )

    block = MapEmbeddingBlock(
        gnn_module=gnn,
        hidden_dim=64,
        activation=torch.nn.functional.elu
    )

    tower.forward = partial(tower.forward, edge_index=edge_index)
    block.forward = partial(block.forward, edge_index=edge_index)

    summary(tower, (144, 19))

    summary(block, (144, 64))
