import json
from pathlib import Path

from summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/Hanoi3P4D/Primitive Agent/",
    "./Training Results/Hanoi3P4D/Multi-Level Agent/",
    "./Training Results/Hanoi3P4D/Betweenness/",
    "./Training Results/Hanoi3P4D/Edge Betweenness/",
    "./Training Results/Hanoi3P4D/Label Propagation/",
    "./Training Results/Hanoi3P4D/Eigenoptions/",
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
OUTPUT_DIRECTORY = "./Experiments/Graph Plotting/R Plots/Hanoi/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "hanoi.json", "w") as f:
    json.dump(results_dict, f, indent=4)
