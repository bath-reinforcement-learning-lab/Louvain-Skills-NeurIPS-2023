import json
from pathlib import Path

from summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/office_2k/Primitive Agent/",
    "./Training Results/office_2k/Multi-Level Agent/",
    "./Training Results/office_2k/Betweenness/",
    "./Training Results/office_2k/Label Propagation/",
    "./Training Results/office_2k/Eigenoptions/",
]

labels = [
    "primitive",
    "louvain",
    "node_betweenness",
    "label_prop",
    "eigenoptions",
]

# Compute summary statistics.
results_dict = compute_statistics_for_directories(directories, labels)


# Ensure the output directory exists.
OUTPUT_DIRECTORY = "./Experiments/Graph Plotting/R Plots/Office 2k/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "office2k.json", "w") as f:
    json.dump(results_dict, f, indent=4)
