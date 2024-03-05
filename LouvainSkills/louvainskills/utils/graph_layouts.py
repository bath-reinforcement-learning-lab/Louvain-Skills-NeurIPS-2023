import networkx as nx


def gridlayout(graph):
    default_pos = {node: {"viz": {"position": {"x": 1.0, "y": 1.0, "z": 1.0}}} for node in graph.nodes}
    nx.set_node_attributes(graph, default_pos)

    for node, _ in graph.nodes(data=True):
        (y, x) = node

        graph.nodes[node]["viz"]["position"]["x"] = x * 24.0
        graph.nodes[node]["viz"]["position"]["y"] = -y * 24.0


def taxilayout(graph):
    default_pos = {node: {"viz": {"position": {"x": 1.0, "y": 1.0, "z": 1.0}}} for node in graph.nodes}
    nx.set_node_attributes(graph, default_pos)

    for node, _ in graph.nodes(data=True):
        (pos, source, dest) = node
        x, y = _number_to_coords(pos)

        graph.nodes[node]["viz"]["position"]["x"] = x * 24.0 + 500 * source
        graph.nodes[node]["viz"]["position"]["y"] = -y * 24.0 + 500 * dest


def pinballlayout(graph):
    default_pos = {node: {"viz": {"position": {"x": 1.0, "y": 1.0, "z": 1.0}}} for node in graph.nodes}
    nx.set_node_attributes(graph, default_pos)

    for (
        node,
        _,
    ) in graph.nodes(data=True):
        (x, y, v_x, v_y) = node

        graph.nodes[node]["viz"]["position"]["x"] = x * 1000.0
        graph.nodes[node]["viz"]["position"]["y"] = -y * 1000.0


def _number_to_coords(square_number):
    taxi_y, taxi_x = divmod(square_number, 5)
    return taxi_x, taxi_y
