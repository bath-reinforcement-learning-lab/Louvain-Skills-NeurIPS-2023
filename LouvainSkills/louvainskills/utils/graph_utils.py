import networkx as nx
import igraph as ig

from typing import List, Hashable


def convert_nx_to_ig(stg):
    # Convert from NetworkX to iGraph graph type.
    stg_ig = ig.Graph.from_networkx(stg)
    return stg_ig


def convert_ig_to_nx(stg_ig):
    # Convert from iGraph to NetworkX graph type.
    stg = stg_ig.to_networkx()

    # Relabel nodes based on `_nx_name` attibute, if present.
    if _nx_name_present(stg):
        # Relabel nodes by nx identifier instead of ig int.
        stg = nx.relabel_nodes(stg, lambda x: stg.nodes[x]["_nx_name"])

        # Delete "_nx_name" node attribute.
        for node in stg:
            del stg.nodes[node]["_nx_name"]

    return stg


def get_all_neighbours(graph: nx.Graph, node: Hashable) -> List[Hashable]:
    """
    Returns both successors and predecessors of a given node on a given graph.

    Args:
        graph (nx.DiGraph): A NetworkX graph (either directed or undirected).
        node (Hashable): The ID of the node whose neighbours you want to find.

    Returns:
        List[Hashable]: A list of IDs of nodes either preceding or succeding the given node on the graph.
    """
    return list(set(list(graph.successors(node)) + list(graph.predecessors(node))))


def _nx_name_present(stg):
    for node in stg:
        if "_nx_name" not in stg.nodes[node]:
            return False
    return True
