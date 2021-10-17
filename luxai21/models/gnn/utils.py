import networkx as nx
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
