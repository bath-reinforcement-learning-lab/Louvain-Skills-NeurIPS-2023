import json
from pathlib import Path

from louvainskills.utils.summary_statistics import compute_statistics_for_directories

directories = [
    "./Training Results/Chapter 3/Incremental/Rooms/Episode/Primitive Agent",
    "./Training Results/Chapter 3/Incremental/Rooms/Episode/Multi-Level Agent",
    "./Training Results/Chapter 3/Incremental/Rooms/Episode/Replace",
    "./Training Results/Chapter 3/Incremental/Rooms/Episode/Update",
]

labels = [
    "primitive",
    "louvain",
    "replace",
    "update",
]

# Compute summary statistics.
results_dict = compute_statistics_for_directories(directories, labels)

# Ensure the output directory exists.
OUTPUT_DIRECTORY = "./Experiments/Chapter 3 - Next Steps/Incremental/"
Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Save the results to a .json file. Pretty print.
with open(OUTPUT_DIRECTORY + "incremental.json", "w") as f:
    json.dump(results_dict, f, indent=4)
