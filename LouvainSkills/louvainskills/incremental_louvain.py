# louvainskills/incremental_louvain.py

import math
import operator
import itertools
from typing import List, Set

import networkx as nx
import networkx.algorithms.community as nx_comm

from louvainskills.utils.graph_utils import get_all_neighbours


def apply_incremental_louvain(original_stg: nx.DiGraph, new_nodes: List) -> nx.DiGraph:
    """
    Applies the incremental partition update used previously inside the agent:
    1) Assign new nodes to clusters at level 0 via local modularity moves.
    2) Push any newly-formed clusters upwards through the hierarchy.
    3) Reassign disconnected clusters (if any) so each connected component has its own label.

    Note that this function preserves the exact behaviour of the in-agent implementation.
    """
    stg = original_stg.copy()

    # First, assign new nodes to clusters (level 0, then push up).
    stg = _assign_node_clusters(stg, new_nodes)

    # Next, deal with disconnected clusters which occur as a result of the Louvain algorithm.
    stg = _reassign_disconnected_clusters(stg)

    # Return the updated STG.
    return stg


# ──────────────────────────────────────────────────────────────────────────────
# Extracted helpers (unchanged in behaviour).
# ──────────────────────────────────────────────────────────────────────────────


def _assign_node_clusters(original_stg: nx.DiGraph, new_nodes: List):
    stg = original_stg.copy()
    existing_clusters = _get_existing_clusters(stg, 0)
    current_levels = _get_current_number_of_levels(stg)

    # Add each new node to its own cluster.
    new_cluster_id = max(existing_clusters, default=-1) + 1
    for node in new_nodes:
        stg.nodes[node]["cluster-0"] = new_cluster_id
        new_cluster_id += 1

    # Iterate through each new node and place it in the neighbouring cluster that maximises modularity.
    # Repeat until no increase in modularity can be found.
    while True:
        last_modularity = nx_comm.modularity(stg, _get_clusters_from_level(stg, 0), weight=None)
        for node in new_nodes:
            # Get neighbouring nodes and their respective clusters.
            neighbours = get_all_neighbours(stg, node)
            neighbour_clusters = set(
                {node: nx.get_node_attributes(stg, "cluster-0")[node] for node in neighbours}.values()
            )
            own_cluster = stg.nodes[node]["cluster-0"]

            # Assign node to the cluster that maximises modularity.
            best_modularity = nx_comm.modularity(stg, _get_clusters_from_level(stg, 0), weight=None)
            best_cluster = own_cluster
            for cluster in neighbour_clusters:
                stg.nodes[node]["cluster-0"] = cluster

                modularity = nx_comm.modularity(stg, _get_clusters_from_level(stg, 0), weight=None)

                if modularity >= best_modularity:
                    best_modularity = modularity
                    best_cluster = cluster

            stg.nodes[node]["cluster-0"] = best_cluster

        if math.isclose(best_modularity, last_modularity, abs_tol=1e-6):
            break

    # Assign new nodes to appropriate higher-level clusters.
    nodes_to_merge = []
    for node in new_nodes:
        # If this node has been assigned to an existing cluster, we can derive its higher-level
        # cluster membership from other nodes in its cluster.
        if stg.nodes[node]["cluster-0"] in existing_clusters:
            # Get another node in this cluster, so that we can copy its cluster membership.
            node_from_same_cluster = next(
                neighbour
                for neighbour in stg.nodes
                if stg.nodes[neighbour]["cluster-0"] == stg.nodes[node]["cluster-0"] and neighbour not in new_nodes
            )

            for att in stg.nodes[node_from_same_cluster]:
                if att.startswith("cluster-"):
                    stg.nodes[node][att] = stg.nodes[node_from_same_cluster][att]
        # Otherwise, we set it aside for adding to a new higher-level cluster.
        else:
            nodes_to_merge.append(node)

    if current_levels <= 1:
        return _assign_new_higher_level_clusters(stg, list(stg.nodes), 1, current_levels)
    else:
        if len(nodes_to_merge) == 0:
            return stg
        else:
            return _assign_new_higher_level_clusters(stg, nodes_to_merge, 1, current_levels)


def _assign_new_higher_level_clusters(original_stg: nx.DiGraph, new_nodes: List, level: int, num_existing_levels: int):
    stg = original_stg.copy()
    existing_clusters = _get_existing_clusters(stg, level)

    # Add each new lower-level cluster to its own higher-level cluster.
    clusters_processed = []
    lower_level_clusters = []
    new_cluster_id = max(existing_clusters, default=-1) + 1
    nodes_labelled = 0
    for new_node in new_nodes:
        if stg.nodes[new_node][f"cluster-{level-1}"] in clusters_processed:
            continue

        # Get all of the nodes in this node's level - 1 cluster.
        nodes_in_same_cluster = [
            node
            for node in stg.nodes
            if stg.nodes[new_node][f"cluster-{level-1}"] == stg.nodes[node][f"cluster-{level-1}"]
        ]
        lower_level_clusters.append(nodes_in_same_cluster)

        # Add the new nodes to their own cluster.
        for node in nodes_in_same_cluster:
            stg.nodes[node][f"cluster-{level}"] = new_cluster_id
            nodes_labelled += 1
        new_cluster_id += 1

        clusters_processed.append(stg.nodes[new_node][f"cluster-{level-1}"])

    # Iterate through each new lower-level cluster and place it in the neighbouring super-cluster
    # that maximises modularity. Repeat until no increase in modularity can be found.
    while True:
        try:
            last_modularity = nx_comm.modularity(stg, _get_clusters_from_level(stg, level), weight=None)
        except Exception:
            # Exited early due to invalid partition labelling.
            return original_stg

        for cluster_of_nodes in lower_level_clusters:
            # Get the neighbours of all of the nodes in this cluster and their respective clusters.
            neighbours = []
            for node in cluster_of_nodes:
                for neighbour in get_all_neighbours(stg, node):
                    if neighbour not in cluster_of_nodes:
                        neighbours.append(neighbour)
            neighbour_clusters = set(
                {node: nx.get_node_attributes(stg, f"cluster-{level}")[node] for node in neighbours}.values()
            )
            own_cluster = stg.nodes[cluster_of_nodes[0]][f"cluster-{level}"]

            # Assign node to the cluster that maximises modularity.
            best_modularity = nx_comm.modularity(stg, _get_clusters_from_level(stg, level), weight=None)
            best_cluster = own_cluster
            for cluster in neighbour_clusters:
                for node in cluster_of_nodes:
                    stg.nodes[node][f"cluster-{level}"] = cluster

                modularity = nx_comm.modularity(stg, _get_clusters_from_level(stg, level), weight=None)

                if modularity >= best_modularity:
                    best_modularity = modularity
                    best_cluster = cluster

            for node in cluster_of_nodes:
                stg.nodes[node][f"cluster-{level}"] = best_cluster

        if math.isclose(best_modularity, last_modularity, abs_tol=1e-6):
            break

    # Deal with clusters that have been merged with existing clusters.
    # Send other clusters to be merged in the next level of the hierarchy.
    nodes_to_merge = []
    for cluster_of_nodes in lower_level_clusters:
        # If this cluster has been merged with an existing cluster, we can derive its higher-level
        # cluster membership from other nodes in its cluster.
        if stg.nodes[cluster_of_nodes[0]][f"cluster-{level}"] in existing_clusters:
            # Get another node in this cluster, so that we can copy its cluster membership.
            node_from_same_cluster = next(
                neighbour
                for neighbour in stg.nodes
                if stg.nodes[neighbour][f"cluster-{level}"] == stg.nodes[cluster_of_nodes[0]][f"cluster-{level}"]
                and neighbour not in new_nodes
            )
            for node in cluster_of_nodes:
                for att in stg.nodes[node_from_same_cluster]:
                    if att.startswith("cluster-"):
                        stg.nodes[node][att] = stg.nodes[node_from_same_cluster][att]
        # Otherwise, we set it aside for merging with a new higher-level cluster.
        else:
            nodes_to_merge.extend(cluster_of_nodes)

    # If the next level already exists, we want to assign new nodes to clusters in it.
    if num_existing_levels > level + 1:
        # If there are no nodes to merge, return the current stg.
        if len(nodes_to_merge) == 0:
            return stg
        else:
            return _assign_new_higher_level_clusters(stg, nodes_to_merge, level + 1, num_existing_levels)
    else:
        last_level_modularity = nx_comm.modularity(stg, _get_clusters_from_level(stg, level - 1), weight=None)
        if best_modularity < last_level_modularity or math.isclose(
            last_level_modularity, best_modularity, abs_tol=1e-6
        ):
            if num_existing_levels == level + 1:
                return stg
            else:
                return original_stg
        else:
            if len(nodes_to_merge) == 0:
                return stg
            elif num_existing_levels <= level + 1:
                return _assign_new_higher_level_clusters(stg, list(stg.nodes), level + 1, num_existing_levels)
            else:
                return _assign_new_higher_level_clusters(stg, nodes_to_merge, level + 1, num_existing_levels)


def _reassign_disconnected_clusters(stg: nx.DiGraph):
    number_of_levels = _get_current_number_of_levels(stg)

    # At each level of the hierarchy, we look at each cluster.
    for level in range(number_of_levels):
        existing_clusters = _get_existing_clusters(stg, level)
        for cluster in existing_clusters:
            # Create the subgraph containing only nodes in this cluster.
            nodes_in_cluster = [node for node in stg.nodes if stg.nodes[node][f"cluster-{level}"] == cluster]
            sub_stg = stg.subgraph(nodes_in_cluster)

            # Get the number of weakly connected components in this subgraph. If it is
            # greater than one, give each connected component a unique cluster label.
            if nx.number_weakly_connected_components(sub_stg) > 1:
                # Get list of all connected components, sorted in descending size order.
                connected_components = [
                    component for component in sorted(nx.weakly_connected_components(sub_stg), key=len, reverse=True)
                ]

                for i, component in enumerate(connected_components):
                    # Nodes in the first (i.e., largest) connected component gets to keep its current label.
                    if i == 0:
                        for node in component:
                            stg.nodes[node][f"cluster-{level}"] = cluster
                    # Nodes in other connected components get assigned a new label.
                    else:
                        new_id = max(_get_existing_clusters(stg, level), default=-1) + 1
                        for node in component:
                            stg.nodes[node][f"cluster-{level}"] = new_id
    return stg


def _get_clusters_from_level(stg: nx.DiGraph, level: int):
    return [
        list(map(operator.itemgetter(0), v))
        for _, v in itertools.groupby(
            nx.get_node_attributes(stg, f"cluster-{level}").items(),
            operator.itemgetter(1),
        )
    ]


def _get_current_number_of_levels(stg: nx.DiGraph):
    level_attributes = set(itertools.chain(*[(stg.nodes[n].keys()) for n in stg.nodes()]))
    level_attributes = {level_att for level_att in level_attributes if level_att.startswith("cluster-")}
    return len(level_attributes)


def _get_existing_clusters(stg: nx.DiGraph, level: int) -> Set[int]:
    return set(nx.get_node_attributes(stg, f"cluster-{level}").values())
