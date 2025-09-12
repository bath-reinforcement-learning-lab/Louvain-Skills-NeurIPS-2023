import json
from pathlib import Path

from louvainskills.utils.summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Primitive Agent",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Multi-Level Agent",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Flat Louvain",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Single-Level Agents/Level 0",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Single-Level Agents/Level 1",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Single-Level Agents/Level 2",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Xu",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Label Propagation",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Edge Betweenness",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Eigenoptions",
    "./Training Results/Chapter 2/Learning Curves/Hanoi/Episode/Betweenness",
]

labels = [
    "primitive",
    "louvain",
    "flat",
    "level 0",
    "level 1",
    "level 2",
    "xu",
    "label propagation",
    "edge betweenness",
    "eigenoptions",
    "betweenness",
]

# Compute summary statistics.
results_dict = compute_statistics_for_directories(directories, labels)

# Ensure the output directory exists.
OUTPUT_DIRECTORY = "./Experiments/Chapter 2 - Louvain Skills/Graph Plotting/Plotting/Hanoi/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "hanoi.json", "w") as f:
    json.dump(results_dict, f, indent=4)
