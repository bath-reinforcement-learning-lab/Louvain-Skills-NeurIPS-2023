import json
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from simpleenvs.envs.hanoi import HanoiEnvironment
from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms

from louvainskills.louvain import apply_louvain
from louvainskills.utils.graph_utils import convert_nx_to_ig


plt.rc("font", family="serif")
plt.rc("text", usetex=False)
plt.rc("xtick", labelsize="medium")
plt.rc("ytick", labelsize="medium")
plt.rc("axes", labelsize="medium")
plt.rc("legend", fontsize="medium")
plt.rcParams["figure.constrained_layout.use"] = True


def get_cluster_sizes(EnvironmentType, env_kwargs, num_repeats, resolutions):
    results = []

    for _ in range(num_repeats):
        # print(f"Environment: {env_name}")

        for i, resolution in enumerate(resolutions):
            # print(f"Resolution: {resolution}")

            # Initialise environment.
            env = EnvironmentType(**env_kwargs)
            env.reset()

            # Generate networkx state-transition diagram.
            stg = env.generate_interaction_graph(directed=True)
            stg_ig = convert_nx_to_ig(stg)

            # Apply Louvain algorithm.
            stg_ig, __ = apply_louvain(
                stg_ig, resolution=resolution, first_levels_to_skip=0, return_aggregate_graphs=True
            )

            # Record the sizes of clusters.
            for j, attribute in enumerate(
                [attribute for attribute in stg_ig.vs[0].attributes() if attribute.startswith("cluster-")]
            ):
                clustering = ig.clustering.VertexClustering.FromAttribute(stg_ig, attribute)
                average_size = np.mean(clustering.sizes())
                # print(f"{j}: {average_size:.3f} Nodes (Modularity: {clustering.modularity:5.3f})")

                while j >= len(results):
                    results.append([])

                while i >= len(results[j]):
                    results[j].append([])

                results[j][i].append(average_size)

            # print("----------------------------------")
    return results


def plot_legend(num_levels):
    labels = [f"Level {level}" for level in range(num_levels + 1)]

    num_elements = len(labels)
    colourmap = list(plt.cm.tab20(np.linspace(0, 0.9, max([10, num_elements]))))

    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f("s", colourmap[i]) for i in range(len(labels))]
    labels = labels
    legend = plt.legend(handles, labels, loc=3, ncol=5, framealpha=1, frameon=True)

    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5, -5, 5, 5])))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("Legend.svg", dpi=300, bbox_inches=bbox, transparent=True)


if __name__ == "__main__":
    num_repeats = 20
    resolutions = list(np.logspace(start=-5, stop=2, base=10, num=20))

    env_names = ["Four Rooms", "Hanoi"]
    rooms_results = get_cluster_sizes(DiscreteXuFourRooms, {}, num_repeats, resolutions)
    hanoi_results = get_cluster_sizes(HanoiEnvironment, {"num_disks": 4, "num_poles": 3}, num_repeats, resolutions)

    num_curves = max([len(rooms_results), len(hanoi_results)])
    colourmap = list(plt.cm.tab20(np.linspace(0, 0.9, max([10, num_curves]))))

    for k, results in enumerate([rooms_results, hanoi_results]):
        # Plot results.
        results_summary = {"resolutions": resolutions}
        fig = plt.figure(figsize=(3, 3), constrained_layout=True)
        for j, result in enumerate([results]):
            env_name = env_names[k]
            for i in range(len(result)):
                plt.plot(
                    resolutions[: len(result[i])],
                    [np.mean(runs) for runs in result[i]],
                    label=f"Level {i + 1}",
                    linestyle="-",
                    marker="o",
                    markersize=2,
                )
                plt.fill_between(
                    resolutions[: len(result[i])],
                    [np.mean(runs) - np.std(runs) for runs in result[i]],
                    [np.mean(runs) + np.std(runs) for runs in result[i]],
                    alpha=0.3,
                )
                results_summary[f"Level {i + 1}"] = [np.mean(runs) for runs in result[i]]

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Resolution Parameter (ρ)")
            plt.ylabel("Mean No. of Nodes Per Cluster")
            plt.grid("both")
            output_path = "./Training Results/Resolution Analysis/"
            Path(output_path).mkdir(parents=True, exist_ok=True)
            plt.legend()
            plt.show()
            plt.savefig(
                f"{output_path}/{env_name}.pdf",
                format="pdf",
                dpi=300,
                transparent=True,
                bbox_inches="tight",
            )
            plt.clf()
            plt.close()

        with open(f"{output_path}/{env_name}.json", "w") as f:
            json.dump(results_summary, f, indent=2)

    plot_legend(5)
