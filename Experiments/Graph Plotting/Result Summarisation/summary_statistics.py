import os
import json
import numpy as np


def compute_statistics(folder_path):
    arrays = []

    # Read all .json files in the folder.
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path) as f:
                data = json.load(f)
                arrays.append(data)

    # Convert list of arrays to a numpy array for easier computation.
    arrays = np.array(arrays)

    # Compute mean, standard deviation, and standard error along axis 0 (index-wise).
    means = np.mean(arrays, axis=0)
    std_devs = np.std(arrays, axis=0)

    return means, std_devs


def compute_statistics_for_directories(directories, labels):
    results = {}

    for directory, label in zip(directories, labels):
        means, std_devs = compute_statistics(directory)
        results[label] = {}
        results[label]["mean"] = list(means)
        results[label]["std_dev"] = list(std_devs)

    return results
