import json
from pathlib import Path

from summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/Grid/Primitive Agent/",
    "./Training Results/Grid/Multi-Level Agent/",
    "./Training Results/Grid/Betweenness/",
    "./Training Results/Grid/Edge Betweenness/",
    "./Training Results/Grid/Label Propagation/",
    "./Training Results/Grid/Eigenoptions/",
]

labels = [
    "primitive",
    "louvain",
    "node_betweenness",
    "edge_betweenness",
    "label_prop",
    "eigenoptions",
]

# Compute summary statistics.
results_dict = compute_statistics_for_directories(directories, labels)


# Ensure the output directory exists.
OUTPUT_DIRECTORY = "./Experiments/Graph Plotting/R Plots/Grid/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "grid.json", "w") as f:
    json.dump(results_dict, f, indent=4)
