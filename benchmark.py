"""Benchmarks for the solver given different configuration parameters."""

import os
import signal

from puzzle import Puzzle
from solve import solve
from solve_config import SolveConfig
from tools import print_stamped

BENCHMARK_DIR = "./benchmarks"
BENCHMARK_PUZZLES = [
    "n2-k7-0.txt",
    "n2-k8-0.txt",
    "n2-k9-0.txt",
    "n3-k7-0.txt",
    "n3-k8-0.txt",
    "n3-k8-1.txt",
    "n3-k9-0.txt",
]


def load_benchmark_puzzles() -> list[Puzzle]:
    """Load all benchmark puzzles from the list of puzzles.."""
    return [Puzzle.from_file(name) for name in BENCHMARK_PUZZLES]


def raise_(ex):
    """Raise an exception."""
    raise ex


def benchmark_move_sizes(config: SolveConfig, move_sizes: list[int]):
    """Benchmark the solve function for a list of move sizes."""
    print_stamped("benchmarking move sizes...")
    signal.signal(
        signal.SIGALRM,
        lambda signum, frame: raise_(Exception(f"timeout {signum} {frame}")),
    )

    path = os.path.join(BENCHMARK_DIR, "move_sizes.csv")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("puzzle_name,move_size,prep_time,solve_time,k\n")

    with open(path, "a") as f:
        for puzzle in load_benchmark_puzzles():
            print_stamped(f"puzzle {puzzle.name}...")
            fastest: float | None = None

            for move_size in move_sizes:
                print_stamped(f"move size {move_size}...")
                config.move_size = move_size

                if fastest is not None:
                    signal.alarm(int(fastest * 3))

                try:
                    stats = solve(puzzle, config, False)
                except Exception as e:
                    print_stamped(f"exception occurred: {e}")
                    f.write(
                        ",".join([puzzle.name, str(move_size), "-1", "-1", "-1"]) + "\n"
                    )
                else:
                    signal.alarm(0)
                    f.write(
                        ",".join(
                            [
                                puzzle.name,
                                str(move_size),
                                str(stats.total_prep_time().total_seconds()),
                                str(stats.total_solve_time().total_seconds()),
                                str(stats.k()),
                            ]
                        )
                    )
                    total = (
                        stats.total_prep_time().total_seconds()
                        + stats.total_solve_time().total_seconds()
                    )
                    if fastest is None or total < fastest:
                        fastest = total


def benchmark_thread_counts(config: SolveConfig, thread_counts: list[int]):
    """Benchmark the solve function for a list of thread counts."""
    print_stamped("benchmarking thread counts...")
    signal.signal(
        signal.SIGALRM,
        lambda signum, frame: raise_(Exception(f"timeout {signum} {frame}")),
    )

    path = os.path.join(BENCHMARK_DIR, "thread_counts.csv")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("puzzle_name,thread_count,prep_time,solve_time,k\n")

    with open(path, "a") as f:
        for puzzle in load_benchmark_puzzles():
            print_stamped(f"puzzle {puzzle.name}...")
            fastest: float | None = None

            for thread_count in thread_counts:
                print_stamped(f"thread count {thread_count}...")
                config.max_solver_threads = thread_count

                if fastest is not None:
                    signal.alarm(int(fastest * 3))

                try:
                    stats = solve(puzzle, config, False)
                except Exception as e:
                    print_stamped(f"exception occurred: {e}")
                    f.write(
                        ",".join([puzzle.name, str(thread_count), "-1", "-1", "-1"])
                        + "\n"
                    )
                else:
                    signal.alarm(0)
                    f.write(
                        ",".join(
                            [
                                puzzle.name,
                                str(thread_count),
                                str(stats.total_prep_time().total_seconds()),
                                str(stats.total_solve_time().total_seconds()),
                                str(stats.k()),
                            ]
                        )
                    )
                    total = (
                        stats.total_prep_time().total_seconds()
                        + stats.total_solve_time().total_seconds()
                    )
                    if fastest is None or total < fastest:
                        fastest = total


if __name__ == "__main__":
    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    benchmark_thread_counts(SolveConfig.default(), [1, 2, 4, 7])
    benchmark_move_sizes(SolveConfig.default(), [1, 2, 3, 4])
