import networkx as nx
import igraph as ig

from typing import List, Hashable
from collections import defaultdict


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


def aggregate_graph_from_node_attribute(graph: nx.DiGraph, attribute_name: str, directed: bool = True) -> nx.Graph:
    """
    Given a graph and a node attirbute name, this function generates an aggregate graph where the nodes are
    the unique values of the attribute and the edges are the connections between the nodes reflect the connections
    between original nodes with different attribute values in the original graph.

    Args:
        graph (nx.Graph): A NetworkX graph.
        attribute_name (str): The attribute to use for aggregation.
        directed (bool, optional): Whether the returned aggregate graph should be directed. Defaults to True.

    Returns:
        nx.Graph: The aggregate graph.

    Note:
        The edges of the aggregate graph are unweighted.
    """
    # Initialise the aggregate graph.
    aggregate_graph = nx.DiGraph()

    # Create super-nodes based on attribute values and count cluster sizes.
    node_to_supernode = {}
    cluster_sizes = defaultdict(int)

    for node in graph.nodes:
        attribute_value = graph.nodes[node][attribute_name]
        if attribute_value not in aggregate_graph.nodes:
            aggregate_graph.add_node(attribute_value, cluster_size=0)
        node_to_supernode[node] = attribute_value
        cluster_sizes[attribute_value] += 1

    # Update cluster sizes in the aggregate graph.
    for supernode, size in cluster_sizes.items():
        aggregate_graph.nodes[supernode]["cluster_size"] = size

    # Add directed edges between super-nodes based on the original graph.
    for u, v in graph.edges():
        super_u = node_to_supernode[u]
        super_v = node_to_supernode[v]
        if super_u != super_v:  # Avoid self-loops.
            aggregate_graph.add_edge(super_u, super_v)

    if not directed:
        aggregate_graph = aggregate_graph.to_undirected()

    return aggregate_graph


def _nx_name_present(stg):
    for node in stg:
        if "_nx_name" not in stg.nodes[node]:
            return False
    return True


def compress_cluster_labels(graph: nx.Graph, cluster_attr: str) -> nx.Graph:
    """
    Relabels nodes so that cluster labels form a contiguous range of integers.

    Args:
        graph (nx.Graph): The input graph with cluster labels.

    Returns:
        nx.Graph: The graph with compressed cluster labels.
    """
    clusters = nx.get_node_attributes(graph, cluster_attr)
    unique_ids = sorted(set(clusters.values()))
    id_map = {old: new for new, old in enumerate(unique_ids)}
    relabelled = {node: id_map[cid] for node, cid in clusters.items()}
    nx.set_node_attributes(graph, relabelled, cluster_attr)
    return graph
