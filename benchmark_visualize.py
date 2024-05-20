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


def visualize_ours_vs_florians():
    """Visualize ours vs florians benchmark."""
    df_us = pd.read_csv("benchmark_results/move_size.csv")
    # Take only move_size = 1 and n2 puzzles
    df_us = df_us[df_us['move_size'] == 1]
    df_us = df_us[df_us['puzzle_name'].str.contains('n2')]
    # Take average prep_time, solve_time for each puzzle
    df_us = df_us.groupby(["puzzle_name", 'move_size', 'k']).mean().reset_index()
    # Set all values in column move_size to 'ours'
    df_us['move_size'] = 'ours'
    # Rename move_size to parameter_name
    df_us = df_us.rename(columns={'move_size': 'parameter_name'})
    # Drop the message column
    df_us.drop(columns=["message"], inplace=True)

    df_florians = pd.read_csv("benchmark_results/florian.csv")
    # Take average prep_time, solve_time for each puzzle
    df_florians = df_florians.groupby(["puzzle_name", 'florian', 'k']).mean().reset_index()
    # Set all values in column move_size to 'ours'
    df_florians['florian'] = 'florians'
    # Rename florian to parameter_name
    df_florians = df_florians.rename(columns={'florian': 'parameter_name'})
    # Drop the message column
    df_florians.drop(columns=["message"], inplace=True)

    # Concatenate the two dataframes
    df = pd.concat([df_us, df_florians])
    # Add a column that combines prep_time and solve_time in combined_time
    df['combined_time'] = df['prep_time'] + df['solve_time']

    parameter_name = "parameter_name"
    names = df_florians["puzzle_name"].unique()
    rows, cols = closest_factors(len(names))
    fig, subplots = plt.subplots(
        rows, cols, squeeze=False, figsize=(cols * 6, rows * 4)
    )

    def coords_to_idx(x, y):
        return x * cols + y

    width = 0.25

    for xi in range(rows):
        for yi in range(cols):
            subplot = subplots[xi, yi]
            name = names[coords_to_idx(xi, yi)]
            puzzle_df = df[df["puzzle_name"] == name]
            x = np.array(range(len(puzzle_df[parameter_name])))
            subplot.bar(
                x - width,
                puzzle_df["prep_time"],
                label="Prep time",
                width=width,
            )
            subplot.bar(
                x,
                puzzle_df["solve_time"],
                label="Solve time",
                width=width,
            )
            subplot.bar(
                x + width,
                puzzle_df["combined_time"],
                label="Combined time",
                width=width,
            )
            subplot.set_xticks(x, puzzle_df[parameter_name])
            subplot.set_xlabel("ours vs florians")
            subplot.set_ylabel("Time (s)")
            subplot.set_title(name)
            subplot.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(BENCHMARK_RESULTS_DIR, "ours_vs_florians.png"))


if __name__ == "__main__":
    visualize_benchmarks()
    visualize_ours_vs_florians()
