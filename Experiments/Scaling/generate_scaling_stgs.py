import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
from officeworld.generator import OfficeGenerator

from louvainskills.louvain import apply_louvain
from louvainskills.utils.graph_utils import convert_nx_to_ig, convert_ig_to_nx


def generate_office_louvain_hierarchy(num_floors: int) -> nx.DiGraph:

    # Define office building with `num_floors` floors.
    office_gen = OfficeGenerator(num_floors=num_floors, elevator_location=(7, 7))
    office_building = office_gen.generate_office_building()

    # Generate networkx stg.
    stg = office_gen.generate_office_graph(office_building)
    # print(f"\n{num_floors} floors, {stg.number_of_nodes()} states\n")

    # Convert from networkx to igraph.
    stg_ig = convert_nx_to_ig(stg)

    # Perform hierarchical graph clustering.
    resolution = 0.05
    stg_ig = apply_louvain(
        stg_ig, resolution=resolution, first_levels_to_skip=0, return_aggregate_graphs=False  # , weights="weight"
    )

    stg = convert_ig_to_nx(stg_ig)

    return stg


if __name__ == "__main__":

    NUM_THREADS = 10
    RESULTS_DIR = "./Training Results/Scaling STGs/"

    min_floors = 1
    max_floors = 800
    num_offices = 15

    # Generate a list of num_offices numbers between min_floors and max_floors,
    # and which are equally spaced on a logarithmic (base 10) scale.
    log_start = np.log10(min_floors)
    log_end = np.log10(max_floors)
    log_points = np.linspace(log_start, log_end, num=num_offices)
    num_floors_list = [int(round(10**i, 0)) for i in log_points]

    print(num_floors_list)

    with Pool(processes=NUM_THREADS) as pool:
        stgs = list(tqdm(pool.imap(generate_office_louvain_hierarchy, num_floors_list), total=len(num_floors_list)))

    # Save the STG files.
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)  # Testing performance.
    for i, stg in enumerate(stgs):
        nx.write_gexf(stg, f"{RESULTS_DIR}/Office STG - {num_floors_list[i]} Floors.gexf")

        print(f"Saved Office STG with {len(stg.nodes)} states and {num_floors_list[i]} floors.")
