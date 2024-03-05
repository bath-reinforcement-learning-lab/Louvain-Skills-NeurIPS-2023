import copy

import numpy as np
import igraph as ig
import networkx as nx

from pathlib import Path

from simpleenvs.envs.discrete_rooms.rooms import (
    DiscreteXuFourRooms,
    DiscreteDefaultNineRooms,
    RameshMaze,
)
from simpleenvs.envs.hanoi import HanoiEnvironment
from simpleenvs.envs.taxi import TaxiEnvironment

from louvainskills.louvain import apply_louvain
from louvainskills.utils.graph_layouts import gridlayout
from louvainskills.utils.graph_utils import convert_nx_to_ig, convert_ig_to_nx


if __name__ == "__main__":
    # Tuples of the form: (Env Class, Env Arguments, String Descriptor, Graph Layout Helper)
    environments = [
        (DiscreteDefaultNineRooms, {}, "Grid", gridlayout),
        (DiscreteXuFourRooms, {}, "Rooms", gridlayout),
        (RameshMaze, {}, "Maze", gridlayout),
        (TaxiEnvironment, {}, "Taxi", None),
        (HanoiEnvironment, {"num_disks": 4, "num_poles": 3}, "Hanoi", None),
    ]

    # Generate graph for each environment.
    for EnvironmentType, kwargs, env_name, layoutManager in environments:
        # Initialise environment.
        env = EnvironmentType(**kwargs)
        env.reset()

        # Generate networkx state-transition graph.
        stg = env.generate_interaction_graph(directed=True)

        # Convert networkx to igraph.
        stg_ig = convert_nx_to_ig(stg)

        # Perform hierarchical graph clustering.
        agg = []

        stg_ig, agg = apply_louvain(
            stg_ig,
            resolution=0.05,
            first_levels_to_skip=0,
            return_aggregate_graphs=True,
        )

        # Convert igraph to networkx.
        stg = convert_ig_to_nx(stg_ig)

        # Apply graph layout if applicable.
        if layoutManager is not None:
            layoutManager(stg)

        # Save graph to file.
        output_path = "./Training Results/STGs/"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        nx.write_gexf(stg, f"{output_path}/{env_name}.gexf", prettyprint=True)

        # I recommend opening the resulting .gexf files in Gephi for visualisation.
        # For graphs that haven't had a layout applied, you can use the ForceAtlas2
        # in Gephi - this is what was used for Taxi and Towers of Hanoi in the paper.
