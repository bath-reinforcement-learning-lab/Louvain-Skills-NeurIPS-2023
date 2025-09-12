import json
from pathlib import Path

from louvainskills.utils.summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Primitive Agent",
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Modularity (0.01)",
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Modularity (0.1)",
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Modularity (1.0)",
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Modularity (10.0)",
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Xu",
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Label Propagation",
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Edge Betweenness",
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Eigenoptions",
    "./Training Results/Chapter 1/Learning Curves/Playroom/Episode/Betweenness",
]

labels = [
    "primitive",
    "modularity (0.01)",
    "modularity (0.1)",
    "modularity (1.0)",
    "modularity (10.0)",
    "xu",
    "label propagation",
    "edge betweenness",
    "eigenoptions",
    "betweenness",
]

# Compute summary statistics.
results_dict = compute_statistics_for_directories(directories, labels)

# Ensure the output directory exists.
OUTPUT_DIRECTORY = "./Experiments/Chapter 1 - Modularity/Graph Plotting/Plotting/Playroom/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "playroom.json", "w") as f:
    json.dump(results_dict, f, indent=4)
