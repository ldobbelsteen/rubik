"""Script for benchmarking the performance of the solver."""

import os

import pandas as pd
from tqdm import tqdm

from puzzle import Puzzle
from solve import solve
from solve_config import SolveConfig

BENCHMARK_DIR = "./benchmarks"
BENCHMARK_PUZZLES = ["n2-random5.txt"]


def load_benchmark_puzzles() -> list[Puzzle]:
    """Load all puzzles in the benchmark list."""
    return [Puzzle.from_file(name) for name in BENCHMARK_PUZZLES]


def benchmark_move_sizes(base_config: SolveConfig, move_sizes: list[int]):
    """Benchmark the solver for different move sizes."""
    results = []
    for puzzle in tqdm(load_benchmark_puzzles()):
        for move_size in tqdm(move_sizes):
            base_config.move_size = move_size
            stats = solve(puzzle, base_config, False)
            results.append(
                (
                    puzzle.name,
                    move_size,
                    stats.total_prep_time().total_seconds(),
                    stats.total_solve_time().total_seconds(),
                )
            )

    df = pd.DataFrame(
        data=results,
        columns=["puzzle_name", "move_size", "prep_time", "solve_time"],
    )

    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    df.to_csv(os.path.join(BENCHMARK_DIR, "move_sizes_results.csv"), index=False)


if __name__ == "__main__":
    benchmark_move_sizes(SolveConfig.default(), [1, 2, 3, 4, 5])
