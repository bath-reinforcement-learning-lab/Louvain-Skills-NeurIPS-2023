import math
import random
import distinctipy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from typing import Tuple
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point, Polygon


from louvainskills.utils.graph_layouts import pinballlayout
from louvainskills.louvain import apply_louvain
from louvainskills.utils.graph_utils import convert_nx_to_ig, convert_ig_to_nx


# Define Colors.
GRAY = "#777777"
DARK_GRAY = "#444444"
GREEN = "#3BB143"
DARK_GREEN = "#0B6623"
RED = "#D30000"
DARK_RED = "#800000"

seed = 42

random.seed(seed)
np.random.seed(seed)


def points_to_graph(points, n_neighbours, sigma=0.25):
    # Create the k nearest neighbour model.
    knn = NearestNeighbors(n_neighbors=n_neighbours)
    knn.fit(points)

    # Construct the STG by linking each point to its nearest neighbours.
    stg = nx.Graph()
    for point in points:
        distances, neighbour_idxs = knn.kneighbors([point], n_neighbors=n_neighbours + 1)
        distances = distances.squeeze()[1:]
        neighbours = [points[neighbour_idx] for neighbour_idx in neighbour_idxs.squeeze()[1:]]

        for distance, neighbour in zip(distances, neighbours):
            if not stg.has_edge(point, neighbour):
                weight = math.exp(-math.pow(distance, 2) / sigma)
                stg.add_edge(point, neighbour, weight=weight)

    return stg


class PinballStateSampler(object):
    def __init__(self, config_path: str, velocity_range: Tuple[float, float] = (-0.1, 0.1)):
        self._load_config(config_path)
        self.vmin = velocity_range[0]
        self.vmax = velocity_range[1]

    def _load_config(self, config_path):
        start = None
        target = None
        ball_size = None
        obstacles = []

        with open(config_path, "r") as f:
            for line in f:
                # Trim trailing spaces and newline characters.
                line = line.strip().rstrip("\n").lower()

                # Ignore comments and blank lines.
                if line.startswith("#") or line == "":
                    continue
                elif line.startswith("polygon"):
                    coords = line.split(" ")[1:]
                    vertices = [tuple(coords[x : x + 2]) for x in range(0, len(coords), 2)]
                    vertices = [(float(x), 1 - float(y)) for x, y in vertices]
                    obstacle = Polygon(vertices)
                    obstacles.append(obstacle)
                elif line.startswith("start"):
                    start_string = line.split(" ")[1:]
                    start = (float(start_string[0]), 1 - float(start_string[1]))
                elif line.startswith("target"):
                    target_string = line.split(" ")[1:]
                    target = (float(target_string[0]), 1 - float(target_string[1]))
                    target_size = float(target_string[2])
                elif line.startswith("ball"):
                    ball_size = float(line.split(" ")[1])
                else:
                    illegal_entry = line.split(" ")[0]
                    raise ValueError(f"Found illegal entry '{illegal_entry}' in config file {config_path}.")

        self.start = start
        self.ball_size = ball_size
        self.target = target
        self.target_size = target_size
        self.obstacles = obstacles

    def draw(self, points=None):
        if points is None:
            points = []

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Add obstacles.
        for obstacle in self.obstacles:
            xs, ys = obstacle.exterior.xy
            ax.fill(xs, ys, fc=GRAY, ec=DARK_GRAY, zorder=3)

        # Add start circle.
        ax.add_patch(plt.Circle(self.start, self.ball_size, fc=GREEN, ec=DARK_GREEN, zorder=2))

        # Add target circle.
        ax.add_patch(plt.Circle(self.target, self.target_size, fc=RED, ec=DARK_RED, zorder=2))

        # Add points.
        for point in points:
            if len(point) == 2:
                x, y = point
                ax.plot(x, y, marker=".", zorder=1, ms=7)
            elif len(point) == 3:
                x, y, c = point
                ax.plot(x, y, marker=".", zorder=1, ms=7, c=c)
            elif len(point) == 4:
                x, y, _, _ = point
                ax.plot(x, y, marker=".", zorder=1, ms=7)
            elif len(point) == 5:
                x, y, _, _, c = point
                ax.plot(x, y, marker=".", zorder=1, ms=7, c=c)

        # Set figure limits.
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        plt.show()

    # def sample_points(self, num_points):
    #     points = [None for _ in range(num_points)]

    #     for i in range(num_points):
    #         # Generate valid (x, y) position.
    #         while True:
    #             x = random.random()
    #             y = random.random()

    #             if not self._is_point_in_obstacle((x, y)):
    #                 break

    #         v_x = random.uniform(self.vmin, self.vmax)
    #         v_y = random.uniform(self.vmin, self.vmax)

    #         points[i] = (x, y, v_x, v_y)

    #     return points

    def sample_points(
        self, radius: float, num_points: int, num_candidates: int = 30
    ) -> list[tuple[float, float, float, float]]:
        """
        Poisson-disk sampling (Bridson) over the unit square minus obstacles/start/goal.
        Returns up to `num_points` samples with min separation `radius`. If the pure
        Poisson stage yields fewer than `num_points`, it tops up using simple
        rejection sampling that still respects the same separation.

        Each returned point is (x, y, v_x, v_y), with velocities in [vmin, vmax].
        """

        # Helpers
        def valid_xy(x: float, y: float) -> bool:
            return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and not self._is_point_in_obstacle((x, y))

        def far_enough(x: float, y: float, pts: list[tuple[float, float]]) -> bool:
            if not pts:
                return True
            r2 = radius * radius
            for px, py in pts:
                dx = x - px
                dy = y - py
                if dx * dx + dy * dy < r2:
                    return False
            return True

        # Acceleration grid for neighbour checks
        cell_size = radius / math.sqrt(2.0)
        gw = max(1, int(math.ceil(1.0 / cell_size)))
        gh = max(1, int(math.ceil(1.0 / cell_size)))
        accel = [[-1 for _ in range(gw)] for __ in range(gh)]

        def to_accel(x: float, y: float) -> tuple[int, int]:
            c = min(gw - 1, max(0, int(x / cell_size)))
            r = min(gh - 1, max(0, int(y / cell_size)))
            return r, c

        def has_conflict(x: float, y: float, pts: list[tuple[float, float]]) -> bool:
            r, c = to_accel(x, y)
            nrad = 2  # search a small window around the cell
            rmin, rmax = max(0, r - nrad), min(gh - 1, r + nrad)
            cmin, cmax = max(0, c - nrad), min(gw - 1, c + nrad)
            r2 = radius * radius
            for rr in range(rmin, rmax + 1):
                row = accel[rr]
                for cc in range(cmin, cmax + 1):
                    idx = row[cc]
                    if idx >= 0:
                        px, py = pts[idx]
                        dx = x - px
                        dy = y - py
                        if dx * dx + dy * dy < r2:
                            return True
            return False

        # 1) Poisson stage
        pts_xy: list[tuple[float, float]] = []
        active: list[int] = []

        # First point
        placed = False
        for _ in range(10000):
            x0 = random.random()
            y0 = random.random()
            if valid_xy(x0, y0):
                pts_xy.append((x0, y0))
                r0, c0 = to_accel(x0, y0)
                accel[r0][c0] = 0
                active.append(0)
                placed = True
                break
        if not placed:
            return []  # domain is fully blocked

        while active and len(pts_xy) < num_points:
            ai = random.randrange(len(active))
            pi = active[ai]
            px, py = pts_xy[pi]

            found = False
            for _ in range(num_candidates):
                ang = random.random() * 2.0 * math.pi
                rad = radius + (2.0 * radius - radius) * random.random()
                x = px + rad * math.cos(ang)
                y = py + rad * math.sin(ang)
                if not valid_xy(x, y):
                    continue
                if has_conflict(x, y, pts_xy):
                    continue
                # accept
                pts_xy.append((x, y))
                rr, cc = to_accel(x, y)
                accel[rr][cc] = len(pts_xy) - 1
                active.append(len(pts_xy) - 1)
                found = True
                if len(pts_xy) >= num_points:
                    break
            if not found:
                active.pop(ai)

        # 2) Top-up (optional but simple): rejection that respects radius
        # Try a bounded number of extra attempts per missing point.
        missing = max(0, num_points - len(pts_xy))
        attempts = 0
        max_attempts = 2000 * missing if missing > 0 else 0
        while missing > 0 and attempts < max_attempts:
            attempts += 1
            x = random.random()
            y = random.random()
            if not valid_xy(x, y):
                continue
            if has_conflict(x, y, pts_xy):
                continue
            pts_xy.append((x, y))
            rr, cc = to_accel(x, y)
            accel[rr][cc] = len(pts_xy) - 1
            missing -= 1

        # Package with velocities
        points = []
        for x, y in pts_xy[:num_points]:
            v_x = random.uniform(self.vmin, self.vmax)
            v_y = random.uniform(self.vmin, self.vmax)
            points.append((x, y, v_x, v_y))
        return points

    def _is_point_in_obstacle(self, coords, include_start=True, include_goal=True):
        point = Point(coords)

        for obstacle in self.obstacles:
            if obstacle.contains(point):
                # print(f"Point is in obstacle: {obstacle}")
                return True

        if include_start:
            start_area = Point(self.start).buffer(self.ball_size)
            if start_area.contains(point):
                # print(f"Point is in start area.")
                return True

        if include_start:
            target_area = Point(self.target).buffer(self.target_size)
            if target_area.contains(point):
                # print(f"Point is in target area.")
                return True

        return False


if __name__ == "__main__":
    # Load Pinball Environment.
    config_path = "./Experiments/Chapter 3 - Next Steps/Continuous/pinball configs/pinball_simple.cfg"
    print("Initialising Pinball Domain...")
    state_sampler = PinballStateSampler(config_path, velocity_range=(0.0, 0.0))
    print("Initialised Pinball Domain.")

    # Sample Points.
    print("Sampling Continuous States...")
    target_num_points = 4000  # 150
    radius = 0.011  # 0.065
    points = state_sampler.sample_points(radius, target_num_points)
    print(f"Sampled {len(points)} states.")
    print("Sampled Continuous States...")

    num_levels = 0
    while num_levels < 3 or num_levels > 4:
        # Create STG.
        stg = points_to_graph(points, 10)
        print("Created STG Using KNN Method...")

        # Perform hierarchical graph clustering.
        stg_ig = convert_nx_to_ig(stg)
        stg_ig, aggs_ig = apply_louvain(
            stg_ig, resolution=0.05, first_levels_to_skip=0, return_aggregate_graphs=True, weights="weight"
        )
        stg = convert_ig_to_nx(stg_ig)

        aggs = [convert_ig_to_nx(agg_ig) for agg_ig in aggs_ig[1:]]
        num_levels = len(aggs)
        print(num_levels)

    print(f"Applied Louvain Algorithm, Found {len(aggs)} Levels...")

    # For each level of the hierarchy, plot the points and their cluster assignments.
    for i in range(len(aggs)):
        num_clusters = aggs[i].number_of_nodes()
        colours = distinctipy.get_colors(num_clusters, pastel_factor=0.2, rng=seed)
        coloured_points = [(x, y, colours[stg.nodes[(x, y, v_x, v_y)][f"cluster-{i}"]]) for x, y, v_x, v_y in points]
        state_sampler.draw(coloured_points)

    # Apply layout and save.
    pinballlayout(stg)
    # nx.write_gexf(stg, f"pinball.gexf", prettyprint=True)
    print("Finished!")
