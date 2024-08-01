import json
from pathlib import Path

from summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/Maze/Primitive Agent/",
    "./Training Results/Maze/Multi-Level Agent/",
    "./Training Results/Maze/Flat Agent/",
    "./Training Results/Maze/Single-Level Agents/Level 0/",
    "./Training Results/Maze/Single-Level Agents/Level 1/",
    "./Training Results/Maze/Single-Level Agents/Level 2/",
    "./Training Results/Maze/Single-Level Agents/Level 3/",
    "./Training Results/Maze/Betweenness/",
    "./Training Results/Maze/Edge Betweenness/",
    "./Training Results/Maze/Label Propagation/",
    "./Training Results/Maze/Eigenoptions/",
]

labels = [
    "primitive",
    "louvain",
    "louvain_flat",
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "node_betweenness",
    "edge_betweenness",
    "label_prop",
    "eigenoptions",
]

# Compute summary statistics.
results_dict = compute_statistics_for_directories(directories, labels)


# Ensure the output directory exists.
OUTPUT_DIRECTORY = "./Experiments/Graph Plotting/R Plots/Maze/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "maze.json", "w") as f:
    json.dump(results_dict, f, indent=4)
