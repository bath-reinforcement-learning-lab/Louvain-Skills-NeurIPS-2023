import json
from pathlib import Path

from louvainskills.utils.summary_statistics import compute_statistics_for_directories

envs = ["Rooms", "Hanoi"]
resolutions = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]


for env in envs:
    directories = [f"./Training Results/Chapter 2/Resolution Ablation/{env}/Episode/Primitive Agent"] + [
        f"./Training Results/Chapter 2/Resolution Ablation/{env}/{resolution}/Episode/Multi-Level Agent"
        for resolution in resolutions
    ]

    labels = [
        "primitive",
        "0.001",
        "0.005",
        "0.01",
        "0.05",
        "0.1",
        "0.5",
        "1.0",
        "5.0",
        "10.0",
        "50.0",
        "100.0",
    ]

    # Compute summary statistics.
    results_dict = compute_statistics_for_directories(directories, labels)

    # Ensure the output directory exists.
    OUTPUT_DIRECTORY = "./Experiments/Chapter 2 - Louvain Skills/Graph Plotting/Plotting/Resolution Ablation/"
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Save the results to a .json file. Pretty print.
    with open(OUTPUT_DIRECTORY + f"{env.lower()}.json", "w") as f:
        json.dump(results_dict, f, indent=4)
