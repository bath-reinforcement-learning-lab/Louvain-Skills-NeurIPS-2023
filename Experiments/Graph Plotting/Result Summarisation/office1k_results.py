import json
from pathlib import Path

from summary_statistics import compute_statistics_for_directories

for eval_type in ["Episode", "Epoch"]:

    directories = [
        f"./Training Results/Learning Curves/office_1k/{eval_type}/Primitive Agent/",
        f"./Training Results/Learning Curves/office_1k/{eval_type}/Multi-Level Agent/",
        f"./Training Results/Learning Curves/office_1k/{eval_type}/Betweenness/",
        f"./Training Results/Learning Curves/office_1k/{eval_type}/Label Propagation/",
        f"./Training Results/Learning Curves/office_1k/{eval_type}/Eigenoptions/",
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
    OUTPUT_DIRECTORY = "./Experiments/Graph Plotting/R Plots/Office 1k/"
    Path(OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)

    # Save the results to a .json file. Pretty print.
    with open(OUTPUT_DIRECTORY + f"office1k_{eval_type}.json", "w") as f:
        json.dump(results_dict, f, indent=4)
