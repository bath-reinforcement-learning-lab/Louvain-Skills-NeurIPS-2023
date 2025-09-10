import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def get_cmap_colours_hex(cmap_name: str, n: int, start_frac: float = 0.0, end_frac: float = 0.0):
    """
    Return a list of n equally spaced colours from the given matplotlib colourmap as hex strings.

    Args:
        cmap_name (str): Name of the colourmap.
        n (int): Number of colours to return.
        start_frac (float, optional): Fraction to trim from the start (low end) of the colourmap (0 ≤ start_frac < 1). Defaults to 0.0.
        end_frac (float, optional): Fraction to trim from the end (high end) of the colourmap (0 ≤ end_frac < 1). Defaults to 0.0.

    Raises:
        ValueError: If start_frac or end_frac are out of bounds.
        ValueError: If start_frac + end_frac >= 1.
        ValueError: If n is less than 1.

    Returns:
        list: A list of hex colour strings.
    """
    if not 0 <= start_frac < 1:
        raise ValueError("start_frac must be between 0 and 1.")
    if not 0 <= end_frac < 1:
        raise ValueError("end_frac must be between 0 and 1.")
    if start_frac + end_frac >= 1:
        raise ValueError("start_frac + end_frac must be < 1.")

    cmap = plt.get_cmap(cmap_name)

    # linspace between trimmed boundaries
    lower = start_frac
    upper = 1 - end_frac
    values = np.linspace(lower, upper, n)
    return [mcolors.to_hex(cmap(v)) for v in values]


def show_colourmap_colours(colours_hex):
    """
    Display a sequence of hex colours as adjacent squares with their hex codes beneath.

    Args:
        colours_hex (list): List of hex colour strings (e.g. ["#440154", "#482777", ...]).

    Raises:
        ValueError: If colours_hex is empty.

    Returns:
        None
    """
    if not colours_hex:
        raise ValueError("colours_hex must contain at least one colour.")

    n = len(colours_hex)
    fig, ax = plt.subplots(figsize=(n * 1.5, 2.5))
    for i, colour in enumerate(colours_hex):
        rect = plt.Rectangle((i, 0.5), 1, 1, color=colour)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.4, colour, ha="center", va="top", fontsize=10, family="monospace")
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.7)
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    cmap_args = [
        ("viridis", 6, 0.0, 0.0),  # Viridis
        ("hot", 6, 0.1, 0.25),  # Hot
        ("plasma", 6, 0.0, 0.0),  # Plasma
        ("cividis", 6, 0.0, 0.0),  # Cividis
        ("gist_rainbow", 6, 0.05, 0.05),  # GIST Rainbow
        ("rainbow", 6, 0.0, 0.0),  # Rainbow
    ]

    for cmap_name, n, start, end in cmap_args:
        # Get colours.
        colours_hex = get_cmap_colours_hex(cmap_name, n, start_frac=start, end_frac=end)

        show_colourmap_colours(colours_hex)
