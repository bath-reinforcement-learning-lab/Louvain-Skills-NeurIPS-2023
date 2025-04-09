import json
from pathlib import Path

from summary_statistics import compute_statistics_for_directories

for reward_type in ["binary"]:
    for eval_type in ["Episode"]:
        directories = [
            f"./Training Results/Learning Curves/office_100k_{reward_type}/{eval_type}/Primitive Agent/",
            f"./Training Results/Learning Curves/office_100k_{reward_type}/{eval_type}/Multi-Level Agent/",
            f"./Training Results/Learning Curves/office_100k_{reward_type}/{eval_type}/Betweenness/",
        ]

        labels = [
            "primitive",
            "louvain",
            "node_betweenness",
        ]

        # Compute summary statistics.
        results_dict = compute_statistics_for_directories(directories, labels)

        # Ensure the output directory exists.
        OUTPUT_DIRECTORY = "./Experiments/Graph Plotting/R Plots/Office 100k/"
        Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

        # Save the results to a .json file. Pretty print.
        with open(OUTPUT_DIRECTORY + f"office100k_{reward_type}_{eval_type}.json", "w") as f:
            json.dump(results_dict, f, indent=4)
