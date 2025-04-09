import numpy as np
import scipy as sci
import networkx as nx


def derive_pvfs(original_stg: nx.Graph, num_pvfs):
    # We make the assumption that the STG is undirected, as per Machado 2017a and Jinnai 2019.
    if isinstance(original_stg, nx.DiGraph):
        stg = original_stg.to_undirected()

    node_list = list(stg.nodes())

    # Compute the normalised graph laplacian.
    laplacian = nx.normalized_laplacian_matrix(stg, nodelist=node_list)

    # Compute the eigenvalues and eigenvectors of the graph laplacian.
    vals, vecs = sci.linalg.eigh(laplacian.todense())

    # Create dictionaries mapping states to pvfs.
    pvfs = []
    negative_pvfs = []
    for i in range(min(num_pvfs, len(vecs))):
        pvfs.append({})
        negative_pvfs.append({})
        pvf = vecs[:, i]
        for j, node in enumerate(node_list):
            pvfs[i][node] = pvf[j]
            negative_pvfs[i][node] = -pvf[j]

    # Add pvfs to graph.
    for i in range(len(pvfs)):
        nx.set_node_attributes(stg, pvfs[i], f"PVF {i}")
        nx.set_node_attributes(stg, negative_pvfs[i], f"PVF -{i}")

    pvfs.extend(negative_pvfs)

    return pvfs, stg


if __name__ == "__main__":
    from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms, RameshMaze
    from simpleenvs.envs.hanoi import HanoiEnvironment
    from simpleenvs.envs.taxi import TaxiEnvironment
    from louvainskills.utils.graph_layouts import gridlayout, taxilayout
    from louvainskills.options import EigenOption
    from officeworld import OfficeWorldEnvironment

    import json
    from officeworld.utils.serialisation import as_enum
    from officeworld.utils.graph_utils import office_layout

    environments = [
        # (DiscreteXuFourRooms, {}, "XuFourRooms", gridlayout),
        # (HanoiEnvironment, {"num_disks": 4, "num_poles": 3}, "Hanoi3P4D", None),
        # (TaxiEnvironment, {}, "Taxi", taxilayout),
        # (RameshMaze, {}, "RameshMaze", gridlayout),
    ]

    # EnvironmentType, kwargs, env_name, layoutManager = environments[0]
    with open(f"./Experiments/Scaling/office_500/office_500.json", "r") as f:
        office = json.load(f, object_hook=as_enum)
    EnvironmentType = OfficeWorldEnvironment
    kwargs = {"office": office}
    env_name = "office_500"
    layoutManager = office_layout

    env = EnvironmentType(**kwargs)
    env.reset()

    # Generate STG and derive PVFs from it.
    num_pvfs = 32
    stg = env.generate_interaction_graph(directed=True)
    pvfs, stg = derive_pvfs(stg, num_pvfs)

    if layoutManager is not None:
        layoutManager(stg, 35, 35)

    for i, pfv in enumerate(pvfs):
        option = EigenOption(env, stg, pvfs[i], i)
        option.train()

    # nx.write_gexf(stg, f"eigenoption pvf test {env_name}.gexf", prettyprint=True)
