import copy
import random

import numpy as np
import igraph as ig
import networkx as nx

from pathlib import Path

from simpleenvs.envs.discrete_rooms.explorable_rooms import (
    ExplorableFourRooms,
    ExplorableRameshMaze,
)
from simpleenvs.envs.hanoi import HanoiEnvironment
from simpleenvs.envs.taxi import TaxiEnvironment

from louvainskills.leiden import apply_leiden
from louvainskills.utils.graph_layouts import gridlayout
from louvainskills.utils.graph_utils import convert_nx_to_ig, convert_ig_to_nx

from officeworld import OfficeWorldEnvironment
from officeworld.utils.graph_utils import office_layout
from officeworld.utils.serialisation import OfficeBuildingJSONHandler


if __name__ == "__main__":
    output_dir = "./Experiments/Chapter 1 - Modularity/Partition Examples/"

    office = OfficeBuildingJSONHandler.load_from_json(
        "./Experiments/Chapter 1 - Modularity/Learning Curves/office_1k.json"
    )
    office_rooms = office.rooms

    # Tuples of the form: (Env Class, Env Arguments, String Descriptor, Graph Layout Helper)
    environments = [
        # (ExplorableFourRooms, {}, "Rooms", gridlayout, [0.01, 0.1, 1.0, 10.0], {"directed": True, "weighted": False}),
        (
            OfficeWorldEnvironment,
            {
                "office": office,
                "start_floor": 0,
                "start_room": random.choice(office_rooms[0]),
                "explorable": True,
            },
            "Office",
            office_layout,
            [0.01, 0.1, 1.0, 10.0],
            {"directed": True},
        ),
        # (TaxiEnvironment, {}, "Taxi", None, [0.01, 0.1, 1.0, 2.0, 10.0], {"directed": True, "weighted": False}),
        # (HanoiEnvironment, {"num_disks": 4, "num_poles": 3}, "Hanoi", None, [0.01, 0.1, 1.0, 10.0], {"directed": True, "weighted": False}),
        # (ExplorableRameshMaze, {}, "Maze", gridlayout, [0.01, 0.1, 1.0, 10.0], {"directed": True, "weighted": False}),
        # (Playroom, {}, "Playroom", None, [0.01, 0.1, 1.0, 10.0], {"directed": True, "weighted": True}),
    ]

    # Generate graph for each environment.
    for EnvironmentType, kwargs, env_name, layoutManager, resolutions, graph_properties in environments:
        # Initialise environment.
        env = EnvironmentType(**kwargs)
        env.reset()

        # Generate a clean version of the state-transition graph to save the final results on.
        if EnvironmentType is OfficeWorldEnvironment:
            clean_stg = env.generate_interaction_graph(directed=True)
        else:
            clean_stg = env.generate_interaction_graph(directed=False, weighted=False)

        # Apply graph layout if applicable.
        if layoutManager is not None:
            if EnvironmentType is OfficeWorldEnvironment:
                layoutManager(clean_stg, floor_height=len(office.layout[0]), floor_width=len(office.layout[0][0]))
            else:
                layoutManager(clean_stg)

        for resolution in resolutions:
            # Generate a version of the state-transition graph for the current resolution.
            if EnvironmentType is not OfficeWorldEnvironment:
                stg = env.generate_interaction_graph(
                    directed=graph_properties["directed"], weighted=graph_properties["weighted"]
                )
            else:
                stg = env.generate_interaction_graph(directed=graph_properties["directed"])

            # Convert networkx to igraph.
            stg_ig = convert_nx_to_ig(stg)

            # Perform hierarchical graph clustering.
            stg_ig, agg = apply_leiden(
                stg_ig,
                resolution=resolution,
                first_levels_to_skip=0,
                return_aggregate_graphs=True,
            )

            stg = convert_ig_to_nx(stg_ig)

            # Add the final partition for this resolution to the clean graph.
            for node in clean_stg.nodes:
                clean_stg.nodes[node][f"cluster_res_{resolution}"] = stg.nodes[node][f"cluster-{len(agg) - 2}"]

        # Save graph to file.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        nx.write_gexf(clean_stg, f"{output_dir}/{env_name}.gexf", prettyprint=True)
