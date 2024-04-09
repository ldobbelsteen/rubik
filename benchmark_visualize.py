import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from benchmark import BENCHMARK_RESULTS_DIR
from tools import closest_factors


def visualize_benchmarks():
    """Visualize the benchmark results onto plots and export them to image files."""
    for filename in os.listdir(BENCHMARK_RESULTS_DIR):
        if not filename.endswith(".csv"):
            continue

        path = os.path.join(BENCHMARK_RESULTS_DIR, filename)

        parameter_name = Path(path).stem
        df = pd.read_csv(path)
        names = df["puzzle_name"].unique()
        rows, cols = closest_factors(len(names))
        fig, subplots = plt.subplots(
            rows, cols, squeeze=False, figsize=(cols * 6, rows * 4)
        )

        def coords_to_idx(x, y):
            return x * cols + y

        width = 0.35

        for xi in range(rows):
            for yi in range(cols):
                subplot = subplots[xi, yi]
                name = names[coords_to_idx(xi, yi)]
                puzzle_df = df[df["puzzle_name"] == name]
                x = np.array(range(len(puzzle_df[parameter_name])))
                subplot.bar(
                    x - (width / 2),
                    puzzle_df["prep_time"],
                    label="Prep time",
                    width=width,
                )
                subplot.bar(
                    x + (width / 2),
                    puzzle_df["solve_time"],
                    label="Solve time",
                    width=width,
                )
                subplot.set_xticks(x, puzzle_df[parameter_name])
                subplot.set_xlabel(parameter_name)
                subplot.set_ylabel("Time (s)")
                subplot.set_title(name)
                subplot.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(BENCHMARK_RESULTS_DIR, f"{parameter_name}.png"))


if __name__ == "__main__":
    visualize_benchmarks()
