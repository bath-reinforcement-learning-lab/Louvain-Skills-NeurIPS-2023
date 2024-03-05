"""
Based on sample code from: https://leidenalg.readthedocs.io/en/stable/advanced.html
"""

import numpy as np
import igraph as ig
import leidenalg as la


def apply_leiden(
    stg: ig.Graph,
    resolution: float = 0.1,
    partition_type: la.VertexPartition.LinearResolutionParameterVertexPartition = None,
    return_aggregate_graphs: bool = False,
    first_levels_to_skip=0,
    weights=None,
):
    """
    Takes an iGraph graph and applies the Leiden algorithm for hierarchical graph
    clustering to it. The clusters assigned to each node at each level of the cluster
    hierarchy are recorded as node attributes on the returned graph.
    Optionally, returns the aggregate graph(s) representing the clsutering at each level
    of the cluster hierarchy.

    Args:
        stg (`ig.Graph`): The iGraph graph you wish to hierarchically cluster.
        resolution (`float`, optional): The resolution parameter to use when computing modularity (or a similar metric). Defaults to 0.1.
        partition_type (`la.VertexPartition.LinearResolutionParameterVertexPartition`, optional): The type of partition to optimise. Defaults to None, in which case modularity will be used.
        return_aggregate_graphs (`bool`, optional): Whether or not to return iGraph graphs representing the aggregate graphs found at each level of the cluster hierarchy. Defaults to False.

    Returns:
        `igraph.Graph`: The input iGraph graph with node attributes added to identify cluster membership at leach level of the hierarchy.
        `List[igraph.Graph]`, optional: A list of iGraph graphs representing the aggregate graph at each level of the clsuter hierarchy. Only returned if return_aggregate_graphs is True.
    """
    # Set optimisation metric, define initial partition, initialise optimiser.
    if partition_type is None:
        partition = la.RBConfigurationVertexPartition(stg, resolution_parameter=resolution, weights=weights)
        refined_partition = la.RBConfigurationVertexPartition(stg, resolution_parameter=resolution, weights=weights)
    else:
        partition = partition_type(stg, resolution_parameter=resolution, weights=weights)
        refined_partition = partition_type(stg, resolution_parameter=resolution, weights=weights)
    optimiser = la.Optimiser()

    # Initialise hierarchy level and initial aggregate graph.
    hierarchy_level = 0
    levels = 0
    partition_agg = refined_partition.aggregate_partition()

    if return_aggregate_graphs:
        aggregate_graphs = [partition_agg.cluster_graph()]

    while optimiser.move_nodes(partition_agg):  # Move nodes between neighbouring clusters to improve modularity.

        # Get individual membership for partition
        partition.from_coarse_partition(partition_agg, refined_partition.membership)

        # Refine partition.
        refined_partition.from_coarse_partition(partition_agg)
        optimiser.merge_nodes_constrained(refined_partition, partition)

        # Derive new aggregate graph from new cluster memberships.
        partition_agg = refined_partition.aggregate_partition()

        # But use membership of actual partition.
        aggregate_membership = [None] * len(refined_partition)
        for i in range(stg.vcount()):
            aggregate_membership[refined_partition.membership[i]] = partition.membership[i]
        partition_agg.set_membership(aggregate_membership)

        # Store current aggregate graph.
        if levels >= first_levels_to_skip:
            stg.vs[f"cluster-{hierarchy_level}"] = partition.membership
            hierarchy_level += 1

            if return_aggregate_graphs:
                aggregate_graphs.append(partition_agg.cluster_graph())

        levels += 1

    if return_aggregate_graphs:
        return stg, aggregate_graphs
    else:
        return stg
