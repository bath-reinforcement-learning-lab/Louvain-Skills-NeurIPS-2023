import os
import json
import networkx as nx
import matplotlib.pyplot as plt


if __name__ == "__main__":
    RESULTS_DIR = "./Training Results/Scaling STGs/"

    print("Processing STGs...")

    points = []
    for file in os.listdir(RESULTS_DIR):
        # Skip non-graph files.
        if not file.endswith(".gexf"):
            continue

        # Read in graph file.
        print(file)
        stg: nx.DiGraph = nx.read_gexf(f"{RESULTS_DIR}/{file}")

        # Record the number of states.
        num_nodes = stg.number_of_nodes()

        # Record number of hierarchy levels.
        first_node = list(stg.nodes.keys())[0]
        level_attributes = [attr for attr in stg.nodes()[first_node].keys() if str(attr).startswith("cluster-")]
        num_levels = len(level_attributes) - 1

        points.append((num_nodes, num_levels))
        print(f"Processed Graph File {file}... Found {num_nodes} States and {num_levels} Levels.")

    # Sort points by number of nodes (first element of each tuple).
    points = sorted(points, key=lambda x: x[0])

    # Unzip points into two lists.
    stg_sizes, level_counts = zip(*points)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(2, 2), constrained_layout=True)
    ax.plot(stg_sizes, level_counts, linestyle="-", marker="o", markersize=2)
    ax.set_xscale("log")
    ax.set_yticks([4, 5, 6, 7])
    ax.set_xlabel("Number of States")
    ax.set_ylabel("Hierarchy Levels")
    ax.grid(visible=True, which="major", axis="both", linestyle="-")

    # Save plot.
    fig.savefig(f"{RESULTS_DIR}/depth scaling.pdf", dpi=300, bbox_inches="tight", transparent=True)

    # Save points.
    results = {"STG Sizes": stg_sizes, "Hierarchy Depths": level_counts}
    with open(f"{RESULTS_DIR}/depth scaling.json", "w") as f:
        json.dump(results, f, indent=2)
