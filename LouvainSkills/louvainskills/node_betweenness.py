import networkx as nx


def apply_node_betweenness(stg: nx.DiGraph, goal_states: list = []):
    centralities = {}

    if goal_states == []:
        # Count all paths.
        centralities = nx.betweenness_centrality(stg, normalized=True, endpoints=True)
    else:
        # Count paths leading to goals.
        centralities = nx.betweenness_centrality_subset(stg, sources=list(stg), targets=goal_states)

    # Find local maxima of betweenness.
    local_maxima = []
    for node in stg:
        is_local_maxima = True

        if stg.out_degree[node] == 0:
            continue

        for neighbour in stg.neighbors(node):
            if centralities[neighbour] > centralities[node]:
                is_local_maxima = False
                break

        if is_local_maxima:
            local_maxima.append(node)

    # Return ordered list.
    return centralities, local_maxima


if __name__ == "__main__":
    from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms
    from simpleenvs.envs.hanoi import HanoiEnvironment
    from simpleenvs.envs.taxi import TaxiEnvironment
    from louvainskills.utils.graph_layouts import gridlayout, taxilayout

    environments = [
        (DiscreteXuFourRooms, {}, "XuFourRooms", gridlayout),
        # (HanoiEnvironment, {"num_disks": 4, "num_poles": 3}, "Hanoi3P4D", None),
        # (TaxiEnvironment, {}, "Taxi", taxilayout),
    ]

    for EnvironmentType, kwargs, env_name, layoutManager in environments:
        env = EnvironmentType(**kwargs)
        env.reset()

        # Generate networkx state-transition diagram.
        stg = env.generate_interaction_graph(directed=True)

        centralities, local_maxima = apply_node_betweenness(stg)

        print(local_maxima)
