import json
from pathlib import Path

from summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/Rooms/Primitive Agent/",
    "./Training Results/Rooms/Multi-Level Agent/",
    "./Training Results/Rooms/Flat Agent/",
    "./Training Results/Rooms/Single-Level Agents/Level 0/",
    "./Training Results/Rooms/Single-Level Agents/Level 1/",
    "./Training Results/Rooms/Single-Level Agents/Level 2/",
    "./Training Results/Rooms/Betweenness/",
    "./Training Results/Rooms/Edge Betweenness/",
    "./Training Results/Rooms/Label Propagation/",
    "./Training Results/Rooms/Eigenoptions/",
]

labels = [
    "primitive",
    "louvain",
    "louvain_flat",
    "level_1",
    "level_2",
    "level_3",
    "node_betweenness",
    "edge_betweenness",
    "label_prop",
    "eigenoptions",
]

# Compute summary statistics.
results_dict = compute_statistics_for_directories(directories, labels)


# Ensure the output directory exists.
OUTPUT_DIRECTORY = "./Experiments/Graph Plotting/R Plots/Rooms/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "rooms.json", "w") as f:
    json.dump(results_dict, f, indent=4)
