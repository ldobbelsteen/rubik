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
    df_us = pd.read_csv("benchmark_results/enable_minimal_moves_n2.csv")

    df_florians = pd.read_csv("benchmark_results/florian.csv")
    # Take average prep_time, solve_time for each puzzle
    df_florians = df_florians.groupby(["puzzle_name", 'florian', 'k']).mean().reset_index()
    # Set all values in column move_size to 'ours'
    df_florians['florian'] = 'florians'
    # Rename florian to parameter_name
    df_florians = df_florians.rename(columns={'florian': 'parameter_name'})
    # Drop the message column
    df_florians.drop(columns=["message"], inplace=True)

    for df_ours, file_name in zip(
            [df_us.copy(), df_us[df_us['enable_minimal_moves_n2']].copy(),
             df_us[~df_us['enable_minimal_moves_n2']]].copy(),
            ["ours_vs_florians.png", "ours_with_minimal_moves_n2_vs_florians.png",
             "ours_without_minimal_moves_n2_vs_florians.png"]
    ):
        df_ours_used = df_ours.copy()
        # Set value in enable_minimal_moves_n2 to 'with_minimal_moves' or 'without_minimal_moves' dependent on whether
        #  enable_minimal_moves_n2 is True or False, respectively
        df_ours_used['enable_minimal_moves_n2'] = df_ours_used['enable_minimal_moves_n2'].apply(
            lambda x: 'with_minimal_moves' if x else 'without_minimal_moves')

        # Take average prep_time, solve_time for each puzzle
        df_ours_used = df_ours_used.groupby(["puzzle_name", 'enable_minimal_moves_n2', 'k']).mean().reset_index()

        # Rename move_size to parameter_name
        df_ours_used = df_ours_used.rename(columns={'enable_minimal_moves_n2': 'parameter_name'})
        # Drop the message column
        df_ours_used.drop(columns=["message"], inplace=True)

        # Concatenate the two dataframes
        df = pd.concat([df_ours_used, df_florians])
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
        fig.savefig(os.path.join(BENCHMARK_RESULTS_DIR, file_name))


if __name__ == "__main__":
    # visualize_benchmarks()
    visualize_ours_vs_florians()
