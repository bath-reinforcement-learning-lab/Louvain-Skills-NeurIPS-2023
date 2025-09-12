import json
import matplotlib.pyplot as plt

# Define office domain names.
env_names = [
    "office2k",
    "office5k",
    "office10k",
    "office20k",
    "office50k",
]

# Define the set of algorithms.
algorithms = {
    "primitive": "Primitives",
    "louvain": "Louvain",
    "node_betweenness": "Node Bet.",
    "label_prop": "Label Prop.",
    "eigenoptions": "Eigenoptions",
}

# Define the 80% performance threshold for each domain.
performance_thresholds = {
    "office2k": 0.132,
    "office5k": 0.092,
    "office10k": -0.042,
    "office20k": 0.0,
    "office50k": -0.384,
}

# Define the number of states in each domain.
num_states = {"office2k": 2537, "office5k": 5120, "office10k": 10282, "office20k": 20368, "office50k": 50533}

# Define the test interval for each domain.
time_intervals = {"office2k": 5, "office5k": 5, "office10k": 10, "office20k": 20, "office50k": 40}

DECISION_STAGES_PER_EPOCH = 1000

# For each domain...
overall_results = {}
for env_name in env_names:
    # Load algorithm's results.
    with open(f"./Experiments/Scaling/Office Results/{env_name}.json", "r") as f:
        office_results = json.load(f)

    # For each algorithm...
    for algorithm in algorithms.keys():
        # Skip if algorithm if it has not been tested in this office.
        if algorithm not in office_results.keys():
            continue

        if algorithm not in overall_results.keys():
            overall_results[algorithm] = []

        algorithm_results = office_results[algorithm]["mean"]

        # Walk through the results until we reach an index that is higher than the performance threshold.
        for i, result in enumerate(algorithm_results):
            if result > performance_thresholds[env_name]:
                break

        if i == len(algorithm_results) - 1:
            continue

        # Add this result to the overall results.
        overall_results[algorithm].append(
            (num_states[env_name], i * time_intervals[env_name] * DECISION_STAGES_PER_EPOCH)
        )

# Save the overall results to a .json file.
print(overall_results)
with open("./Experiments/Graph Plotting/R Plots/Scaling Performance Gap/scaling_results.json", "w") as f:
    json.dump(overall_results, f)

# For each algorithm, plot its results. For each tuple in its list, the first element is the number of states.
# and the second element is the time taken to reach the performance threshold.
for algorithm, results in overall_results.items():
    x = [result[0] for result in results]
    y = [result[1] for result in results]  # One epoch is 1000 decision stages.

    # Add a legend.
    plt.legend()

    plt.scatter(x, y, label=algorithms[algorithm])
    plt.plot(x, y)

    # Label axes.
    plt.xlabel("Number of States")
    plt.ylabel("Decision Stages to Performance Threshold")

    # Set axis limits.
    # plt.xlim(2000, 55000)
    # plt.ylim(0, 20000)

    # Set log scale on both axes.
    plt.xscale("log")
    plt.yscale("log")

    # Add gridlines.
    plt.grid(True)


# Show the plot.
plt.show()
