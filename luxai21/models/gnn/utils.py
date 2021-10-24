import networkx as nx
import torch
from torch_geometric.utils import from_networkx


def get_board_edge_index(width: int, height: int, with_meta_node=True):
    # build the graph
    G: nx.Graph = nx.grid_2d_graph(width, height)

    if with_meta_node:
        new_node_id = "meta"
        G.add_node(new_node_id)

        new_edges = [(new_node_id, x) for x in G.nodes if x != new_node_id]
        new_edges += [(x, new_node_id) for x in G.nodes if x != new_node_id]
        G.add_edges_from(new_edges)

    geometric = from_networkx(G)
    return geometric.edge_index


def batches_to_large_graph(map_batch: torch.Tensor, edge_index: torch.Tensor):
    """Create a large disconnected "graph" from the given batch

    We assume that all given graphs should have the same structure, thus only one edge index has to be given

    For more information, see https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html

    Args:
        map_batch: three dimensional tensor with size (B, N, F). Where B are the number of batches
        N are the number of nodes in the graph, and F the features per node

        edge_index: two dimensional tensor with size [2, num_edges]

    """
    assert map_batch.dim() == 3
    assert edge_index.dim() == 2
    n_batches, n_nodes, features = map_batch.size()

    x = map_batch.view(-1, features)

    edge_indices = []
    for n in range(n_batches):
        i = edge_index + n * n_nodes
        edge_indices.append(i)

    large_edge_index = torch.cat(edge_indices, dim=1)

    return x, large_edge_index, n_batches


def large_graph_to_batches(map_tensor, edge_index, n_batches):
    """Inverse of function 'batches_to_large_graph'


    """
    assert map_tensor.dim() == 2
    assert edge_index.dim() == 2

    features = map_tensor.size()[1]
    x = map_tensor.view(n_batches, -1, features)

    e = None
    if edge_index is not None:
        l = edge_index.size()[1]
        assert l % n_batches == 0
        n = l // n_batches
        e = edge_index[:, 0:n]

    return x, e