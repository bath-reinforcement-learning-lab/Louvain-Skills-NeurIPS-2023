import os
import json
import networkx as nx
import matplotlib.pyplot as plt

from louvainskills.utils.graph_utils import aggregate_graph_from_node_attribute


if __name__ == "__main__":
    LEVELS_TO_SKIP = 1
    INPUT_DIR = "./Training Results/Chapter 2/Scaling/Scaling STGs/"
    OUTPUT_DIR = "./Experiments/Chapter 2 - Louvain Skills/Scaling/STG Scaling/"

    print("Processing STGs...")

    points = []
    for file in os.listdir(INPUT_DIR):
        # Skip non-graph files.
        if not file.endswith(".gexf"):
            continue

        # Read in graph file.
        print(file)
        stg: nx.DiGraph = nx.read_gexf(f"{INPUT_DIR}/{file}")

        # Record the number of states.
        num_states = stg.number_of_nodes()

        # Record number of hierarchy levels.
        level_attributes = [
            attr for attr in stg.nodes()[list(stg.nodes.keys())[0]].keys() if str(attr).startswith("cluster-")
        ]
        num_levels = len(level_attributes)

        # Generate aggregate graph for each level of the hierarchy.
        aggregate_graphs = [aggregate_graph_from_node_attribute(stg, attr) for attr in level_attributes]
        aggregate_graphs = aggregate_graphs[LEVELS_TO_SKIP:]

        # Find the number of outgoing edges from each node in each aggregate graph,
        # and multiply this by each node's "clsuter_size" attribtue.
        avg_num_skills = 0
        for graph in aggregate_graphs:
            for node in graph.nodes:
                avg_num_skills += graph.out_degree(node) * graph.nodes[node]["cluster_size"]
        avg_num_skills /= num_states

        points.append((num_states, avg_num_skills))

    # Sort points by number of nodes (first element of each tuple).
    points = sorted(points, key=lambda x: x[0])

    # Unzip points into two lists.
    stg_sizes, level_counts = zip(*points)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3, 3), constrained_layout=True)
    ax.plot(stg_sizes, level_counts, linestyle="-", marker="o", markersize=2)
    ax.set_xscale("log")
    ax.set_xlabel("Number of States")
    ax.set_ylabel("Skills Available Per State")
    ax.grid(visible=True, which="major", axis="both", linestyle="-")

    # Save points.
    print("Saving results...")
    results = {"STG Sizes": stg_sizes, "Available Skills": level_counts}
    with open(f"{OUTPUT_DIR}/skill availability_scaling.json", "w") as f:
        json.dump(results, f, indent=2)
