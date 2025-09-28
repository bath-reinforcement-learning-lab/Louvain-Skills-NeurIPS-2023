import json
from pathlib import Path

from louvainskills.utils.summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Primitive Agent",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Multi-Level Agent",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Flat Louvain",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Single-Level Agents/Level 0",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Single-Level Agents/Level 1",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Single-Level Agents/Level 2",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Single-Level Agents/Level 3",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Single-Level Agents/Level 4",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Xu",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Label Propagation",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Edge Betweenness",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Eigenoptions",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Betweenness",
    "./Training Results/Chapter 2/Learning Curves/Office/Episode/Best Leiden",
]

labels = [
    "primitive",
    "louvain",
    "flat",
    "level 0",
    "level 1",
    "level 2",
    "level 3",
    "level 4",
    "xu",
    "label propagation",
    "edge betweenness",
    "eigenoptions",
    "betweenness",
    "leiden",
]

# Compute summary statistics.
results_dict = compute_statistics_for_directories(directories, labels)

# Ensure the output directory exists.
OUTPUT_DIRECTORY = "./Experiments/Chapter 2 - Louvain Skills/Graph Plotting/Plotting/Office/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "office.json", "w") as f:
    json.dump(results_dict, f, indent=4)
