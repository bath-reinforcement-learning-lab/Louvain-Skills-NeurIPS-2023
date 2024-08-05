import json
from pathlib import Path

from summary_statistics import compute_statistics_for_directories

for eval_type in ["Episode", "Epoch"]:

    directories = [
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Primitive Agent/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Multi-Level Agent/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Flat Agent/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Single-Level Agents/Level 0/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Single-Level Agents/Level 1/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Single-Level Agents/Level 2/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Single-Level Agents/Level 3/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Betweenness/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Edge Betweenness/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Label Propagation/",
        f"./Training Results/Learning Curves/Taxi/{eval_type}/Eigenoptions/",
    ]

    labels = [
        "primitive",
        "louvain",
        "louvain_flat",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "node_betweenness",
        "edge_betweenness",
        "label_prop",
        "eigenoptions",
    ]

    # Compute summary statistics.
    results_dict = compute_statistics_for_directories(directories, labels)

    # Ensure the output directory exists.
    OUTPUT_DIRECTORY = "./Experiments/Graph Plotting/R Plots/Taxi/"
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Save the results to a .json file. Pretty print.
    with open(OUTPUT_DIRECTORY + f"taxi_{eval_type}.json", "w") as f:
        json.dump(results_dict, f, indent=4)
