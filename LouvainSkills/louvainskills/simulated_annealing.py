import math
import random
import numpy as np
import igraph as ig
import networkx as nx

from collections import Counter
from copy import deepcopy

from louvainskills.utils.graph_utils import compress_cluster_labels


def apply_simulated_annealing_nx(
    stg: nx.Graph,
    resolution: float = 1.0,
    initial_temp: float = 1.0,
    final_temp: float = 1e-5,
    alpha: float = 0.999,
    max_iter: int = 100_000,
    weight: str | None = None,
    verbose: bool = False,
    return_aggregate_graphs: bool = False,
):
    stg = deepcopy(stg)
    nodes = list(stg.nodes())
    num_nodes = len(nodes)

    # Start with the singleton partition (i.e., each node in its own cluster).
    clusters = {node: i for i, node in enumerate(nodes)}
    nx.set_node_attributes(stg, clusters, "cluster-0")

    def current_modularity():
        communities = [[] for _ in range(num_nodes)]
        for node, cid in clusters.items():
            communities[cid].append(node)
        communities = [c for c in communities if c]
        return nx.algorithms.community.modularity(stg, communities, weight=weight, resolution=resolution)

    mod = current_modularity()
    temp = initial_temp
    iteration = 0

    while iteration < max_iter:
        node = random.choice(nodes)
        current_cluster = clusters[node]
        candidate_clusters = set(clusters.values()) - {current_cluster}

        if len(set(clusters.values())) == 1:
            break

        if not candidate_clusters:
            temp *= alpha
            iteration += 1
            continue

        new_cluster = random.choice(list(candidate_clusters))
        original = clusters[node]

        # Try moving the node
        clusters[node] = new_cluster
        nx.set_node_attributes(stg, clusters, "cluster-0")
        new_mod = current_modularity()

        delta = new_mod - mod
        if delta > 0 or random.random() < math.exp(delta / temp):
            mod = new_mod  # Accept the move.
        else:
            clusters[node] = original  # Revert.
            nx.set_node_attributes(stg, clusters, "cluster-0")

        if verbose:
            print(f"{iteration}: {new_mod}")

        temp *= alpha
        iteration += 1

    # Compress cluster labels.
    stg = compress_cluster_labels(stg, cluster_attr="cluster-0")

    ## Build an aggregate graph.
    # Add the nodes.
    aggregate_graph = nx.DiGraph()
    for node in stg.nodes():
        node_cluster_id = stg.nodes[node]["cluster-0"]
        aggregate_graph.add_node(node_cluster_id)

    # Add the edges.
    for u, v in stg.edges():
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


def _compress_cluster_labels_ig(g: ig.Graph, membership, attr: str = "cluster-0"):
    """
    Map arbitrary cluster IDs to a contiguous 0..k-1 range and write to vertex attribute `attr`.
    Returns the remapped membership list.
    """
    unique = sorted(set(membership))
    remap = {old: new for new, old in enumerate(unique)}
    remapped = [remap[c] for c in membership]
    g.vs[attr] = remapped
    return remapped


def _build_aggregate_graph_ig(
    g: ig.Graph,
    membership,
    weight: str = "weight",
) -> ig.Graph:
    """
    Build an aggregate (cluster-level) graph.
    - One vertex per cluster (id == cluster label).
    - Edge weights summed over edges crossing between clusters.
    - Directedness matches the input graph.
    """
    k = len(set(membership))
    directed = g.is_directed()

    # Prepare weight vector if present; otherwise treat edges as weight 1
    has_w = weight in g.es.attributes()
    w = g.es[weight] if has_w else None

    # Accumulate weights between cluster pairs
    agg_weights = {}  # (cu, cv) if directed; (min(cu,cv), max(cu,cv)) if undirected
    for eidx, (u, v) in enumerate(g.get_edgelist()):
        cu, cv = membership[u], membership[v]
        if cu == cv:
            continue
        wt = w[eidx] if w is not None else 1.0
        if directed:
            key = (cu, cv)
        else:
            key = (cu, cv) if cu <= cv else (cv, cu)
        agg_weights[key] = agg_weights.get(key, 0.0) + wt

    # Build aggregate graph
    agg = ig.Graph(n=k, directed=directed)
    if agg_weights:
        edges = list(agg_weights.keys())
        weights = [agg_weights[e] for e in edges]
        agg.add_edges(edges)
        agg.es["weight"] = weights

    return agg


def apply_simulated_annealing(
    stg: ig.Graph,
    resolution: float = 1.0,
    initial_temp: float = 1.0,
    final_temp: float = 1e-5,
    alpha: float = 0.9999,
    max_iter: int = 100_000,
    weight: str = "weight",
    verbose: bool = False,
    return_aggregate_graphs: bool = False,
):
    """
    Pure-igraph simulated annealing for modularity maximisation.

    Behaviour mirrors your NetworkX version:
      - Starts from the singleton partition (each vertex its own cluster).
      - Nodes move only between existing clusters (no creation of new clusters).
      - Uses igraph's built-in modularity with a resolution parameter (supported in python-igraph 0.11.6).
      - Final membership written to vertex attribute 'cluster-0' (labels compacted to 0..k-1).
      - Optionally returns a cluster-level aggregate graph with inter-cluster weights summed.

    Args:
        stg: ig.Graph (directed or undirected).
        resolution: Modularity resolution parameter (rho).
        initial_temp, final_temp, alpha, max_iter: SA schedule.
        weight: Edge weight attribute name; if absent, edges are treated as weight 1.
        verbose: Print modularity per accepted/rejected proposal.
        return_aggregate_graphs: If True, also return [aggregate_graph].

    Returns:
        stg (with vertex attribute 'cluster-0' set), and optionally [aggregate_graph].
    """
    # Work on a copy
    stg = stg.copy()
    n = stg.vcount()
    vertices = list(range(n))

    # Singleton partition
    clusters = list(range(n))  # membership indexed by vertex id

    # Prepare weights for igraph.modularity()
    weights_vec = weight if (weight in stg.es.attributes()) else None

    def current_modularity():
        # python-igraph 0.11.6 supports resolution directly
        return stg.modularity(clusters, weights=weights_vec, resolution=resolution)

    mod = current_modularity()
    temp = initial_temp
    iteration = 0

    while iteration < max_iter:
        # Early exit if only one cluster remains
        if len(set(clusters)) == 1:
            break

        v = random.choice(vertices)
        current_cluster = clusters[v]
        candidate_clusters = set(clusters) - {current_cluster}

        if not candidate_clusters:
            # Advance schedule to avoid a tight loop
            temp *= alpha
            iteration += 1
            continue

        new_cluster = random.choice(list(candidate_clusters))
        original = clusters[v]

        # Propose move
        clusters[v] = new_cluster
        new_mod = current_modularity()

        delta = new_mod - mod
        if delta > 0 or random.random() < math.exp(delta / temp):
            mod = new_mod  # accept
        else:
            clusters[v] = original  # revert

        if verbose and iteration % 1000 == 0:
            print(f"{iteration}: {new_mod}")

        temp *= alpha
        iteration += 1

    # Compress labels and store on vertices as 'cluster-0'
    clusters = _compress_cluster_labels_ig(stg, clusters, attr="cluster-0")

    # Build aggregate graph
    aggregate_graph = _build_aggregate_graph_ig(stg, clusters, weight=weight)

    if return_aggregate_graphs:
        return stg, [aggregate_graph]
    else:
        return stg


if __name__ == "__main__":
    resolutions = [0.1, 1.0, 10.0, 25.0]

    # from simpleenvs.envs.discrete_rooms.explorable_rooms import ExplorableRameshMaze
    from simpleenvs.envs.discrete_rooms import XuFourRooms, ExplorableXuFourRooms
    from louvainskills.utils.graph_utils import convert_nx_to_ig

    env = ExplorableXuFourRooms()
    env.reset()

    G_final_nx = env.generate_interaction_graph(directed=True, weighted=False)

    # Base graph for storing all cluster labels as vertex attributes
    # G_final = ig.Graph.Famous("Zachary")  # Karate club graph
    G_final = convert_nx_to_ig(G_final_nx)

    n = G_final.vcount()

    for resolution in resolutions:
        # G = ig.Graph.Famous("Zachary")
        G = convert_nx_to_ig(G_final_nx)

        G_result, aggs = apply_simulated_annealing(
            G,
            resolution=resolution,
            weight="weight",  # use if present; otherwise treated as unweighted
            initial_temp=1.0,
            final_temp=1e-4,
            alpha=0.9999,  # slower decay
            max_iter=2_000_000,  # more iterations
            return_aggregate_graphs=True,
            verbose=True,
        )

        # Read back final membership from vertex attribute "cluster-0"
        membership = G_result.vs["cluster-0"]
        cluster_sizes = Counter(membership)
        num_clusters = len(cluster_sizes)

        # Check aggregate graph node count matches number of clusters
        agg = aggs[0]
        assert agg.vcount() == num_clusters, f"Aggregate graph has {agg.vcount()} nodes, expected {num_clusters}."

        # Compute modularity at this resolution (weights auto-handled)
        # If your graph had edge weights in attribute "weight", igraph uses them here
        weights_vec = "weight" if ("weight" in G_result.es.attributes()) else None
        mod = G_result.modularity(membership, weights=weights_vec, resolution=resolution)
        print(f"Resolution {resolution}: Modularity {mod}")

        # Store cluster labels on G_final with a per-resolution attribute
        # Ensure attribute length matches vertex count
        if G_final.vcount() != n or G_result.vcount() != n:
            raise ValueError("Vertex count mismatch between graphs.")
        G_final.vs[str(resolution)] = membership

    # G_final.write("karate_sa_ig.graphml")
    G_final.write("maze_sa_ig.graphml")


if __name__ == "__main__nx":
    resolutions = [0.01, 0.1, 1.0, 10.0, 25.0]
    resolutions = [1.0]

    from simpleenvs.envs.discrete_rooms.explorable_rooms import ExplorableRameshMaze

    env = ExplorableRameshMaze()
    env.reset()

    G_final = env.generate_interaction_graph(directed=True, weighted=False)

    for resolution in resolutions:
        G = env.generate_interaction_graph(directed=True, weighted=False)

        G_result, aggs = apply_simulated_annealing_nx(
            G,
            resolution=resolution,
            weight=None,
            initial_temp=1.0,
            final_temp=1e-4,
            alpha=0.9999,
            max_iter=1_000_000,
            return_aggregate_graphs=True,
            verbose=True,
        )

        cluster_assignments = nx.get_node_attributes(G_result, "cluster-0")
        cluster_sizes = Counter(cluster_assignments.values())
        num_clusters = len(cluster_sizes)

        assert aggs[0].number_of_nodes() == num_clusters, (
            f"Number of clusters in aggregate graph ({aggs[0].number_of_nodes()}) does not match expected ({num_clusters})."
        )

        partition = [[] for _ in range(num_clusters)]
        for node, cid in cluster_assignments.items():
            partition[cid].append(node)

        mod = nx.algorithms.community.modularity(G_result, partition, resolution=resolution, weight="weight")
        print(f"Resolution {resolution}: Modularity {mod}")

        nx.set_node_attributes(G_final, cluster_assignments, str(resolution))

    nx.write_gexf(G_final, "maze_sa.gexf", prettyprint=True)
