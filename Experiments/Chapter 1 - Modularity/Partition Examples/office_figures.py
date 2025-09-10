import numpy as np
import networkx as nx
import distinctipy as dp

width = 50
height = 40

stg = nx.read_gexf("./Experiments/Chapter 1 - Modularity/Partition Examples/Office.gexf")

for resolution in [0.1, 1.0, 10.0]:
    # Create a blank height x width image for us to fill in later.
    image = np.zeros((height, width, 3), dtype=np.uint8) * 255

    # Count the number of unique clusters at this resolution.
    cluster_ids = set()
    for node in stg.nodes():
        cluster_id = stg.nodes[node][f"cluster_res_{resolution}"]
        cluster_ids.add(cluster_id)

    # Use distinctipy to generate visually distinct colours for each cluster.
    colours = dp.get_colors(len(cluster_ids), pastel_factor=0.4)
    shuffle_indices = np.arange(len(cluster_ids))
    cluster_id_to_colour = {cid: (np.array(col) * 255).astype(np.uint8) for cid, col in zip(cluster_ids, colours)}

    for node in stg.nodes():
        # Convert node from string "(floor, x, y)" to tuple (floor, x, y)
        _, y, x = eval(node)
        y = int(y)
        x = int(x)

        cluster_id = stg.nodes()[node][f"cluster_res_{resolution}"]
        colour = cluster_id_to_colour[cluster_id]

        image[y, x] = colour

    # Save the image to file.
    from PIL import Image

    img = Image.fromarray(image, "RGB")
    img = img.resize((width * 10, height * 10), Image.NEAREST)
    img.save(f"./Experiments/Chapter 1 - Modularity/Partition Examples/office_res_{resolution}.png")
    print(f"Saved image for resolution {resolution}")
