import igraph as ig


def apply_edge_betweenness(stg: ig.Graph):
    """
    Takes an iGraph graph and applies the edge betweenness (Girvan-Newman) method for graph
    clustering to it. Each iteration, the edge with the highest edge betweenness is removed.
    At each iteration, clusters are defined by the disconnected clusters in the graph.
    The partition from the iteration that maximises modularity is returned.


    Args:
        stg (`ig.Graph`): The iGraph graph you wish to cluster.

    Returns:
        `igraph.Graph`: The input iGraph graph with node attributes added to identify cluster memberships.
        `List[igraph.Graph]` : A list containing a single graph, representing the aggregate graph formed by the found patition.
    """

    dendrogram = stg.community_edge_betweenness(clusters=None, directed=True)
    optimal_count = dendrogram.optimal_count
    partition = dendrogram.as_clustering(n=optimal_count)

    stg.vs["cluster-0"] = partition.membership

    return stg, [partition.cluster_graph()]
