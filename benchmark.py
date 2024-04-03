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


def benchmark_parameter(
    base_config: SolveConfig,
    parameter_name: str,
    parameter_values: list[int],
):
    """Benchmark the solve function for a list of parameter values."""
    print_stamped(f"benchmarking '{parameter_name}' parameter...")
    path = os.path.join(BENCHMARK_DIR, f"{parameter_name}.csv")
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    def raise_(e: Exception):
        """Raise an exception."""
        raise e

    # Raise a timeout exception when a SIGALRM signal is received.
    signal.signal(
        signal.SIGALRM,
        lambda *_: raise_(Exception("timeout")),
    )

    # Add CSV file headers if new file.
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(
                ",".join(
                    ["puzzle_name", parameter_name, "prep_time", "solve_time", "k"]
                )
            )

    # Run the solver for each puzzle and parameter value.
    with open(path, "a") as f:
        for puzzle in load_benchmark_puzzles():
            print_stamped(f"puzzle {puzzle.name}...")
            fastest_time_seconds: float | None = None

            for parameter_value in parameter_values:
                print_stamped(f"value {parameter_value}...")
                setattr(base_config, parameter_name, parameter_value)

                # Set timeout for 3 times the fastest time so far.
                if fastest_time_seconds is not None:
                    signal.alarm(int(fastest_time_seconds * 3))

                try:
                    stats = solve(puzzle, base_config, False)
                except Exception:
                    print_stamped("timeout occurred, skipping...")

                    # Write result with values -1 for timeout.
                    f.write(
                        ",".join(
                            map(
                                str,
                                [puzzle.name, parameter_value, -1, -1, -1],
                            )
                        )
                        + "\n"
                    )
                else:
                    # Cancel timeout.
                    signal.alarm(0)

                    # Write result to CSV file.
                    f.write(
                        ",".join(
                            map(
                                str,
                                [
                                    puzzle.name,
                                    parameter_value,
                                    stats.total_prep_time().total_seconds(),
                                    stats.total_solve_time().total_seconds(),
                                    stats.k(),
                                ],
                            )
                        )
                    )

                    # Update fastest time if faster.
                    time_seconds = (
                        stats.total_prep_time().total_seconds()
                        + stats.total_solve_time().total_seconds()
                    )
                    if (
                        fastest_time_seconds is None
                        or time_seconds < fastest_time_seconds
                    ):
                        fastest_time_seconds = time_seconds


if __name__ == "__main__":
    benchmark_parameter(SolveConfig.default(), "move_size", [1, 2, 3, 4])
    benchmark_parameter(SolveConfig.default(), "max_solver_threads", [1, 2, 4, 7])
    benchmark_parameter(SolveConfig.default(), "apply_theorem_11a", [True, False])
    benchmark_parameter(SolveConfig.default(), "apply_theorem_11b", [True, False])
