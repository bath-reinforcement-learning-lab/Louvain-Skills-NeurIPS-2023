#!/usr/bin/env python3
"""
png_to_svg.py

Convert an NxM pixel PNG to an SVG where pixels of the same colour are joined
(flood-fill style) into rectilinear polygons. One <path> per colour, using
fill-rule="evenodd" so holes are handled correctly.

Usage:
    python png2blocks_svg.py input.png output.svg [--scale 1] [--alpha-threshold 0]

Notes:
- Connectivity is 4-neighbour (up/down/left/right).
- Colours are grouped by exact RGBA value (after loading).
- Fully or nearly-transparent pixels can be ignored via --alpha-threshold.
- Coordinates preserve the pixel grid exactly; use --scale to enlarge.
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
from PIL import Image

Point = Tuple[int, int]
Edge = Tuple[Point, Point]


def mask_to_boundary_cycles(mask: np.ndarray) -> List[List[Point]]:
    """
    Given a boolean mask (HxW), return a list of boundary cycles.
    Each cycle is a list of points (x, y) in pixel grid coordinates,
    forming a closed polygon around the union of unit squares for which mask==True.

    Algorithm:
      - For each filled pixel, conceptually add its 4 unit-square edges.
      - Remove shared edges between adjacent filled pixels.
      - Remaining edges form one or more disjoint rectilinear cycles.
      - Trace cycles by walking connected edges.
    """
    h, w = mask.shape

    # Pad with False to handle borders cleanly
    P = np.pad(mask, 1, mode="constant", constant_values=False)

    # Slices for neighbours in padded space
    C = P[1:-1, 1:-1]
    U = P[:-2, 1:-1]  # up
    D = P[2:, 1:-1]  # down
    L = P[1:-1, :-2]  # left
    R = P[1:-1, 2:]  # right

    # Booleans where a boundary edge exists
    top_edge = C & ~U
    bottom_edge = C & ~D
    left_edge = C & ~L
    right_edge = C & ~R

    edges: List[Edge] = []

    # Collect edges as segments between integer grid points
    ys, xs = np.nonzero(top_edge)
    for y, x in zip(ys, xs):
        # top of pixel at (y,x) => segment from (x, y) to (x+1, y)
        edges.append(((x, y), (x + 1, y)))

    ys, xs = np.nonzero(bottom_edge)
    for y, x in zip(ys, xs):
        # bottom => (x, y+1) to (x+1, y+1)
        edges.append(((x, y + 1), (x + 1, y + 1)))

    ys, xs = np.nonzero(left_edge)
    for y, x in zip(ys, xs):
        # left => (x, y) to (x, y+1)
        edges.append(((x, y), (x, y + 1)))

    ys, xs = np.nonzero(right_edge)
    for y, x in zip(ys, xs):
        # right => (x+1, y) to (x+1, y+1)
        edges.append(((x + 1, y), (x + 1, y + 1)))

    if not edges:
        return []

    # Build adjacency map for undirected edges between grid points
    adj: Dict[Point, List[Point]] = defaultdict(list)
    edge_set: Set[Tuple[Point, Point]] = set()

    def add_edge(a: Point, b: Point):
        # store both directions for easy traversal, but track visited by unordered pair
        adj[a].append(b)
        adj[b].append(a)
        key = (a, b) if a < b else (b, a)
        edge_set.add(key)

    for a, b in edges:
        add_edge(a, b)

    # Trace cycles. Each boundary vertex should have degree 2; cycles are well-defined.
    visited_edges: Set[Tuple[Point, Point]] = set()
    cycles: List[List[Point]] = []

    for start in list(adj.keys()):
        # Skip if all incident edges visited
        has_unvisited = any((((start, nb) if start < nb else (nb, start)) not in visited_edges) for nb in adj[start])
        if not has_unvisited:
            continue

        # Walk a cycle
        cycle: List[Point] = []
        current = start
        prev = None

        while True:
            cycle.append(current)
            # Choose next neighbour: one that is not the previous, and whose edge is unvisited
            candidates = adj[current]
            next_pt = None
            for nb in candidates:
                if nb == prev:
                    continue
                key = (current, nb) if current < nb else (nb, current)
                if key not in visited_edges:
                    next_pt = nb
                    visited_edges.add(key)
                    break

            if next_pt is None:
                # We might have reached the start; close the polygon if needed
                if len(cycle) > 1 and adj[current]:
                    # ensure the closing edge (current -> start) is marked visited
                    key = (current, cycle[0]) if current < cycle[0] else (cycle[0], current)
                    if key in edge_set and key not in visited_edges:
                        visited_edges.add(key)
                break

            prev, current = current, next_pt
            if current == start:
                cycle.append(current)  # explicitly close
                break

        # Only keep non-degenerate cycles (length >= 4 points including repeated start)
        if len(cycle) >= 4:
            # Remove duplicated final point for SVG path convenience; we'll add Z to close
            if cycle[0] == cycle[-1]:
                cycle = cycle[:-1]
            cycles.append(cycle)

    return cycles


def cycles_to_svg_path(cycles: List[List[Point]], scale: float = 1.0) -> str:
    """Convert cycles (lists of (x,y)) to a single SVG path 'd' string with even-odd fill."""
    parts: List[str] = []
    for cyc in cycles:
        if not cyc:
            continue
        # Move to first point
        x0, y0 = cyc[0]
        parts.append(f"M {x0 * scale:.6g} {y0 * scale:.6g}")
        # Line through subsequent points
        for x, y in cyc[1:]:
            parts.append(f"L {x * scale:.6g} {y * scale:.6g}")
        parts.append("Z")
    return " ".join(parts)


def rgba_to_svg_fill(rgba: Tuple[int, int, int, int]) -> Tuple[str, float]:
    r, g, b, a = rgba
    return f"rgb({r},{g},{b})", a / 255.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorise a pixel PNG into blocky SVG shapes per colour.")
    parser.add_argument("input", help="Input PNG path")
    parser.add_argument("output", help="Output SVG path")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for SVG coordinates (default: 1.0)")
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=0,
        help="Ignore pixels with alpha <= threshold (0..255). Default 0 keeps all nonzero alpha.",
    )
    args = parser.parse_args()

    im = Image.open(args.input).convert("RGBA")
    arr = np.array(im, dtype=np.uint8)  # H×W×4
    H, W, _ = arr.shape

    # Flatten to list of unique colours (RGBA)
    flat = arr.reshape(-1, 4)
    # Optionally drop (nearly) transparent
    if args.alpha_threshold > 0:
        keep = flat[:, 3] > args.alpha_threshold
        flat = flat[keep]

    if flat.size == 0:
        # Nothing to draw
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * args.scale}" height="{H * args.scale}" viewBox="0 0 {W * args.scale} {H * args.scale}"/>'
            )
        return

    # Unique colours present (rows are RGBA)
    unique_cols = np.unique(flat, axis=0)

    # Prepare SVG pieces
    svg_parts: List[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W * args.scale:.6g}" height="{H * args.scale:.6g}" '
        f'viewBox="0 0 {W * args.scale:.6g} {H * args.scale:.6g}">'
    )
    svg_parts.append('<g shape-rendering="crispEdges">')  # keep blocky edges

    # For each colour: build a mask, extract boundary cycles, and emit a <path>
    for rgba in unique_cols:
        r, g, b, a = map(int, rgba.tolist())
        if a <= args.alpha_threshold:
            continue
        mask = np.all(arr == rgba, axis=2)

        cycles = mask_to_boundary_cycles(mask)
        if not cycles:
            continue

        d_attr = cycles_to_svg_path(cycles, scale=args.scale)
        fill, fill_opacity = rgba_to_svg_fill((r, g, b, a))
        svg_parts.append(
            f'<path d="{d_attr}" fill="{fill}" fill-opacity="{fill_opacity:.6g}" stroke="none" fill-rule="evenodd"/>'
        )

    svg_parts.append("</g>")
    svg_parts.append("</svg>")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))
