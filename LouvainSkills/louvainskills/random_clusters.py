import numpy as np
import igraph as ig
import networkx as nx


def random_clusters_nx(
    stg: nx.Graph,
    num_clusters: int,
    return_aggregate_graphs: bool = False,
):
    """
    Takes a networkx graph and randomly assigns each node to one of n clusters.
    Specifically, n nodes are chosen (randomly) as cluster centres, and each node is assigned
    to the nearest cluster centre.

    Args:
        stg (nx.Graph): The networkx graph to partition.
        num_clusters (int): The desired number of clusters.

    Returns:
        nx.Graph: The input networkx graph with node attributes added to identify cluster memberships.
        List[nx.Graph], optional: A list of iGraph graphs representing the aggregate graph at each level of the clsuter hierarchy. Only returned if return_aggregate_graphs is True.
    """

    ## Randomly select n nodes as cluster centres.
    seed_nodes = np.random.choice(stg.nodes, num_clusters, replace=False)
    seed_nodes = list(seed_nodes)

    ## Create a dictionary mapping each node to the nearest cluster centre.
    cluster_mapping = {}
    for node in stg.nodes:
        distances = {seed_node: stg[node][seed_node] for seed_node in seed_nodes}
        cluster_mapping[node] = min(distances, key=distances.get)
        stg.nodes[node]["cluster-0"] = seed_nodes.index(cluster_mapping[node])

    ## Generate the aggregate graph.
    # Add the nodes (one for each cluster).
    aggregate_graph = nx.DiGraph()
    for i in range(num_clusters):
        aggregate_graph.add_node(i)

    # Add the edges (one for each edge between nodes in different clusters
    # in the original graph).
    for u, v in stg.edges:
        cluster_u = stg.nodes[u]["cluster-0"]
        cluster_v = stg.nodes[v]["cluster-0"]
        if cluster_u != cluster_v:
            aggregate_graph.add_edge(cluster_u, cluster_v, weight=stg[u][v])

    if not nx.is_directed(stg):
        aggregate_graph = aggregate_graph.to_undirected()

    if return_aggregate_graphs:
        return stg, [aggregate_graph]
    else:
        return stg


def random_clusters_ig(
    stg: ig.Graph,
    num_clusters: int,
    return_aggregate_graphs: bool = False,
):
    """
    Takes an iGraph graph and randomly assigns each node to one of n clusters.
    Specifically, n nodes are chosen (randomly) as cluster centres, and each node is assigned
    to the nearest cluster centre.
    Args:
        stg (ig.Graph): The iGraph graph to partition.
        num_clusters (int): The desired number of clusters.
    Returns:
        ig.Graph: The input iGraph graph with node attributes added to identify cluster memberships.
        List[ig.Graph], optional: A list of iGraph graphs representing the aggregate graph at each level of the clsuter hierarchy. Only returned if return_aggregate_graphs is True.
    """
    # Randomly select n nodes as cluster centres.
    seed_nodes = np.random.choice(range(stg.vcount()), num_clusters, replace=False)
    seed_nodes = list(seed_nodes)

    # Compute all-pairs shortest paths
    undir_stg = stg.as_undirected()
    all_shortest_paths = undir_stg.shortest_paths_dijkstra()

    # Create a dictionary mapping each node to the nearest cluster centre.
    cluster_mapping = {}
    for node in range(stg.vcount()):
        distances = {seed: all_shortest_paths[node][seed] for seed in seed_nodes}
        closest_seed = min(distances, key=distances.get)
        cluster_id = seed_nodes.index(closest_seed)
        cluster_mapping[node] = cluster_id
        stg.vs[node]["cluster-0"] = cluster_id

    # Generate the aggregate graph
    aggregate_edges = {}
    for edge in stg.es:
        source = edge.source
        target = edge.target
        cluster_u = cluster_mapping[source]
        cluster_v = cluster_mapping[target]
        if cluster_u != cluster_v:
            key = tuple(sorted((cluster_u, cluster_v))) if not stg.is_directed() else (cluster_u, cluster_v)
            aggregate_edges[key] = aggregate_edges.get(key, 0) + (
                edge["weight"] if "weight" in edge.attributes() else 1
            )

    # Create the aggregate igraph
    aggregate_graph = ig.Graph(directed=stg.is_directed())
    aggregate_graph.add_vertices(num_clusters)
    edge_list = list(aggregate_edges.keys())
    weights = list(aggregate_edges.values())
    aggregate_graph.add_edges(edge_list)
    aggregate_graph.es["weight"] = weights

    if return_aggregate_graphs:
        return stg, [aggregate_graph]
    else:
        return stg
