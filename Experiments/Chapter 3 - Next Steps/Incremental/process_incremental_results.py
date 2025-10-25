import json
from pathlib import Path

from louvainskills.utils.summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/Chapter 3/Incremental/Rooms/Train/Primitive Agent",
    "./Training Results/Chapter 3/Incremental/Rooms/Train/Multi-Level Agent",
    "./Training Results/Chapter 3/Incremental/Rooms/Train/Replace",
    "./Training Results/Chapter 3/Incremental/Rooms/Train/Update",
    "./Training Results/Chapter 3/Incremental/Rooms/Train/Hybrid",
]

labels = [
    "primitive",
    "louvain",
    "replace",
    "update",
    "hybrid",
]

# Compute summary statistics.
results_dict = compute_statistics_for_directories(directories, labels)

# Ensure the output directory exists.
OUTPUT_DIRECTORY = "./Experiments/Chapter 3 - Next Steps/Incremental/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "incremental.json", "w") as f:
    json.dump(results_dict, f, indent=4)
