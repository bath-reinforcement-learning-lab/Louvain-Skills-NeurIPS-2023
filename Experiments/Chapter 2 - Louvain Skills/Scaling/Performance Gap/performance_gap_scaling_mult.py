import json
import matplotlib.pyplot as plt

# Define office domain names.
env_names = [
    "office2k",
    "office5k",
    "office10k",
    "office20k",
    "office50k",
    "office100k",
]

# Define the set of algorithms.
algorithms = {
    "primitive": "Primitives",
    "louvain": "Louvain",
    "node_betweenness": "Node Bet.",
    "label_prop": "Label Prop.",
    "eigenoptions": "Eigenoptions",
}

# Compute the 20%, 40%, 60%, and 80% performance thresholds for each domain based on the
# maximum performance of the primitive agent.
performance_levels = [0.2, 0.4, 0.6, 0.8]
performance_thresholds = {}
for env_name in env_names:
    performance_thresholds[env_name] = {}

    with open(f"./Experiments/Chapter 2 - Louvain Skills/Scaling/Office Results/{env_name}.json", "r") as f:
        office_results = json.load(f)

    # Get the maximum and minimum performance of the primitive agent.
    primitive_results = office_results["primitive"]["mean"]
    max_perf, min_perf = max(primitive_results), min(primitive_results)

    # Compute the performance thresholds.
    for level in performance_levels:
        performance_thresholds[env_name][level] = min_perf + level * (max_perf - min_perf)


# Define the number of states in each domain.
num_states = {
    "office2k": 2537,
    "office5k": 5120,
    "office10k": 10282,
    "office20k": 20368,
    "office50k": 50533,
    "office100k": 100530,
}

# Define the test interval for each domain.
time_intervals = {"office2k": 5, "office5k": 5, "office10k": 10, "office20k": 20, "office50k": 40, "office100k": 40}

DECISION_STAGES_PER_EPOCH = 1000

# For each domain...
overall_results = {}
for env_name in env_names:
    # Load algorithm's results.
    with open(f"./Experiments/Chapter 2 - Louvain Skills/Scaling/Office Results/{env_name}.json", "r") as f:
        office_results = json.load(f)

    # For each algorithm...
    for algorithm in algorithms.keys():
        # Skip if algorithm if it has not been tested in this office.
        if algorithm not in office_results.keys():
            continue

        # overall_results[algorithm][level]

        if algorithm not in overall_results.keys():
            overall_results[algorithm] = {level: [] for level in performance_levels}

        algorithm_results = office_results[algorithm]["mean"]

        # For each performance level...
        for level in performance_levels:
            # Walk through the results until we reach an index that is higher than the performance threshold.
            for i, result in enumerate(algorithm_results):
                if result > performance_thresholds[env_name][level]:
                    break

            if i == len(algorithm_results) - 1:
                continue

            # Add this result to the overall results.
            overall_results[algorithm][level].append(
                (num_states[env_name], i * time_intervals[env_name] * DECISION_STAGES_PER_EPOCH)
            )

# Save the overall results to a .json file.
print(overall_results)
with open("./Experiments/Chapter 2 - Louvain Skills/Scaling/Performance Gap/scaling_results_mult.json", "w") as f:
    json.dump(overall_results, f)

# For each algorithm, plot its results. For each tuple in its list, the first element is the number of states.
# and the second element is the time taken to reach the performance threshold.
for algorithm, level_results in overall_results.items():
    for level, results in level_results.items():
        x = [result[0] for result in results]
        y = [result[1] for result in results]  # One epoch is 1000 decision stages.

        plt.scatter(x, y, label=f"{algorithms[algorithm]} ({level})")
        plt.plot(x, y)

        # Label axes.
        plt.xlabel("Number of States")
        plt.ylabel("Decision Stages to Performance Threshold")

        # Set axis limits.
        plt.xlim(2000, 106000)
        plt.ylim(10, 100_000_000)

        # Set log scale on both axes.
        plt.xscale("log")
        plt.yscale("log")

        # Add gridlines.
        plt.grid(True)

        # Add a legend.
        plt.legend()

    # Show the plot.
    plt.show(block=True)
    plt.clf()
