"""Benchmarks for the solver given different configuration parameters."""

import os

import pandas as pd
from tqdm import tqdm

from puzzle import Puzzle
from solve import solve
from solve_config import SolveConfig
from tools import print_stamped

BENCHMARK_DIR = "./benchmarks"
BENCHMARK_PUZZLES = [
    # "n2-k7-0.txt",
    # "n2-k8-0.txt",
    # "n2-k9-0.txt",
    "n3-k7-0.txt",
    "n3-k8-1.txt",
    "n3-k9-0.txt",
]


def load_benchmark_puzzles() -> list[Puzzle]:
    """Load all benchmark puzzles from the list of puzzles.."""
    return [Puzzle.from_file(name) for name in BENCHMARK_PUZZLES]


def benchmark_move_sizes(config: SolveConfig, move_sizes: list[int]):
    """Benchmark the solve function for a list of move sizes."""
    print_stamped("benchmarking move sizes...")
    results = []
    for puzzle in tqdm(load_benchmark_puzzles()):
        for move_size in tqdm(move_sizes):
            config.move_size = move_size
            stats = solve(puzzle, config, False)
            results.append(
                (
                    puzzle.name,
                    move_size,
                    stats.total_prep_time().total_seconds(),
                    stats.total_solve_time().total_seconds(),
                    stats.k(),
                )
            )

    pd.DataFrame(
        data=results,
        columns=["puzzle_name", "move_size", "prep_time", "solve_time", "k"],
    ).to_csv(os.path.join(BENCHMARK_DIR, "move_sizes.csv"), index=False)


def benchmark_thread_count(config: SolveConfig, thread_counts: list[int]):
    """Benchmark the solve function for a list of thread counts."""
    print_stamped("benchmarking thread counts...")
    results = []
    for puzzle in tqdm(load_benchmark_puzzles()):
        for thread_count in tqdm(thread_counts):
            config.max_solver_threads = thread_count
            stats = solve(puzzle, config, False)
            results.append(
                (
                    puzzle.name,
                    thread_count,
                    stats.total_prep_time().total_seconds(),
                    stats.total_solve_time().total_seconds(),
                    stats.k(),
                )
            )

    pd.DataFrame(
        data=results,
        columns=["puzzle_name", "thread_count", "prep_time", "solve_time", "k"],
    ).to_csv(os.path.join(BENCHMARK_DIR, "thread_counts.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    benchmark_thread_count(SolveConfig.default(), [1, 2, 4, 7])
    benchmark_move_sizes(SolveConfig.default(), [1, 2, 3, 4])
