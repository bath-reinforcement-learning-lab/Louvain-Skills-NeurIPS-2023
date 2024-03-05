import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from officeworld.generator import OfficeGenerator

from louvainskills.louvain import apply_louvain
from louvainskills.utils.graph_utils import convert_nx_to_ig, convert_ig_to_nx

max_power = 2  # Raise to 3 for a 1,000 floor office with 1,000,000 states, but be prepared to wait a *while*.
num_offices = 10  # Controls how many different-sized offices are generated.
num_floors_list = [int(round(i, 0)) for i in list(np.logspace(0, max_power, num=num_offices, base=10))]

print(num_floors_list)

for num_floors in tqdm(num_floors_list):
    office_gen = OfficeGenerator(num_floors=num_floors, elevator_location=(7, 7))
    office_building = office_gen.generate_office_building()
    stg = office_gen.generate_office_graph(office_building)

    # Convert networkx to igraph.
    stg_ig = convert_nx_to_ig(stg)

    # Perform hierarchical graph clustering.
    resolution = 0.05
    stg_ig, agg = apply_louvain(
        stg_ig, resolution=resolution, first_levels_to_skip=0, return_aggregate_graphs=True  # , weights="weight"
    )

    stg = convert_ig_to_nx(stg_ig)

    nx.write_gexf(stg, f"./Training Results/Scaling STGs/Office STG - {num_floors} Floors.gexf")

points = []
results_path = "./Training Results/Scaling STGs/"
for file in os.listdir("./Training Results/Scaling STGs/"):
    # Read in graph file.
    stg: nx.DiGraph = nx.read_gexf(f"{results_path}/{file}")

    # Record the number of states.
    num_nodes = stg.number_of_nodes()

    # Record number of hierarchy levels.
    first_node = list(stg.nodes.keys())[0]
    level_attributes = [attr for attr in stg.nodes()[first_node].keys() if str(attr).startswith("cluster-")]
    num_levels = len(level_attributes) - 1

    points.append((num_nodes, num_levels))
    print(f"Processed Graph File {file}... Found {num_nodes} States and {num_levels} Levels.")

# Sort points by number of nodes (first element of each tuple).
points = sorted(points, key=lambda x: x[0])

# Unzip points into two lists.
stg_sizes, level_counts = zip(*points)

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(2, 2), constrained_layout=True)
ax.plot(stg_sizes, level_counts, linestyle="-", marker="o", markersize=2)
ax.set_xscale("log")
ax.set_yticks([4, 5, 6, 7])
ax.set_xlabel("Number of States")
ax.set_ylabel("Hierarchy Levels")
ax.grid(visible=True, which="major", axis="both", linestyle="-")

# Save plot.
fig.savefig(f"./OfficeWorld Scaling.pdf", dpi=300, bbox_inches="tight", transparent=True)

# Save points.
results = {"STG Sizes": stg_sizes, "Hierarchy Depths": level_counts}
with open("stg scaling results.json", "w") as f:
    json.dump(results, f, indent=2)
