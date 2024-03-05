import random
import igraph as ig


def apply_label_propagation(stg: ig.Graph):
    """
    Takes an iGraph graph and applies the Label Propagation Algorithm (LPA) for graph
    clustering to it. Neighbouring clusters in the partition found using LPA are then
    merged if doing so would lead to modularity gain.

    Args:
        stg (`ig.Graph`): The iGraph graph you wish to cluster.

    Returns:
        `igraph.Graph`: The input iGraph graph with node attributes added to identify cluster memberships.
        `List[igraph.Graph]` : A list containing a single graph, representing the aggregate graph formed by the found patition.
    """

    # Find a partition using the Label Propagation Algorithm (LPA).
    partition = stg.community_label_propagation(weights=None, initial=None, fixed=None)

    # stg.vs["cluster-0-original"] = copy.copy(partition.membership)

    improved = True
    while improved:
        improved = False
        merges = []
        merge_modularities = []
        current_modularity = partition.recalculate_modularity()

        # Look at the change im modularity from merging all pairs of connected clusters.
        for edge in partition.cluster_graph().es:
            u, v = edge.source, edge.target
            if (u != v) and ((u, v) not in merges) and ((v, u) not in merges):

                temp_membership = partition.membership
                for node, _ in enumerate(stg.vs):
                    if temp_membership[node] == u:
                        temp_membership[node] = v

                temp_partition = ig.VertexClustering(stg, temp_membership)

                temp_modularity = temp_partition.recalculate_modularity()
                if temp_modularity > current_modularity:
                    merges.append((u, v))
                    merge_modularities.append(temp_modularity)

        # Merge the two clusters that result in the highest modularity gain.
        # Ties between merges with equal maximum modularities broken randomly.
        if merges != []:
            improved = True
            u, v = merges[
                random.choice(
                    [idx for idx, modularity in enumerate(merge_modularities) if modularity == max(merge_modularities)]
                )
            ]

            membership = partition.membership
            for node, _ in enumerate(stg.vs):
                if membership[node] == u:
                    membership[node] = v

            partition = ig.VertexClustering(stg, membership)

            # print(f"Merged {u} and {v}.")

    stg.vs["cluster-0"] = partition.membership

    return stg, [partition.cluster_graph()]