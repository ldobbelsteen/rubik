import os

from config import SolveConfig, Tactics
from puzzle import Puzzle
from solve import solve
from tools import print_stamped

TIMEOUT_FACTOR = 3
MIN_TIMEOUT_SECS = 60
BENCHMARK_RESULTS_DIR = "./benchmark_results"
BENCHMARK_PUZZLES = [
    "n2-k5-0",
    "n2-k7-0",
    "n2-k8-0",
    "n2-k8-1",
    "n2-k9-0",
    "n2-k9-1",
    "n3-k7-0",
    "n3-k8-0",
    "n3-k8-1",
    "n3-k9-0",
]


def load_benchmark_puzzles() -> list[Puzzle]:
    """Load all benchmark puzzles from the list of puzzles.."""
    return [Puzzle.from_file(name) for name in BENCHMARK_PUZZLES]


def benchmark_param(parameter_name: str, parameter_values: list):
    """Benchmark the solve function for a list of parameter values. This can be
    used to determine which parameters are best.
    """
    print_stamped(f"benchmarking '{parameter_name}' parameter...")
    path = os.path.join(BENCHMARK_RESULTS_DIR, f"{parameter_name}.csv")
    os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)

    # If file does not exist, create it and add CSV column headers.
    if not os.path.exists(path):
        with open(path, "w") as file:
            file.write(
                ",".join(
                    [
                        "puzzle_name",
                        parameter_name,
                        "prep_time",
                        "solve_time",
                        "k",
                        "message",
                    ]
                )
                + "\n"
            )

    # Run the solver for each puzzle and parameter value.
    with open(path, "a", buffering=1) as file:

        def write_line(puzzle_name, parameter_value, prep_time, solve_time, k, message):
            """Write a line to the CSV file."""
            file.write(
                ",".join(
                    map(
                        str,
                        [
                            puzzle_name,
                            parameter_value,
                            prep_time,
                            solve_time,
                            k,
                            message,
                        ],
                    )
                )
                + "\n"
            )

        for puzzle in load_benchmark_puzzles():
            print_stamped(f"puzzle {puzzle.name}...")
            time_range_secs: tuple[float, float] | None = None

            for parameter_value in parameter_values:
                print_stamped(f"value {parameter_value}...")

                # Set the parameter value in the config.
                config = SolveConfig(**{parameter_name: parameter_value})

                # Set timeout to max of: 3x the fastest run,
                # 1x the slowest run, or a minimum timeout.
                timeout_secs = (
                    None
                    if time_range_secs is None
                    else max(
                        time_range_secs[1],
                        time_range_secs[0] * TIMEOUT_FACTOR,
                        MIN_TIMEOUT_SECS,
                    )
                )

                # Solve and write the result to the CSV file.
                stats = solve(puzzle, config, timeout_secs, False, False)
                if stats is None:
                    write_line(
                        puzzle.name,
                        parameter_value,
                        "",
                        "",
                        "",
                        f"timeout after {timeout_secs}s",
                    )
                elif stats.k() is None:
                    write_line(
                        puzzle.name,
                        parameter_value,
                        stats.total_prep_time().total_seconds(),
                        stats.total_solve_time().total_seconds(),
                        "",
                        "no solution found",
                    )
                else:
                    write_line(
                        puzzle.name,
                        parameter_value,
                        stats.total_prep_time().total_seconds(),
                        stats.total_solve_time().total_seconds(),
                        stats.k(),
                        "",
                    )

                # Update the time range.
                if stats is not None:
                    duration_secs = stats.total_time().total_seconds()
                    if time_range_secs is None:
                        time_range_secs = (duration_secs, duration_secs)
                    else:
                        time_range_secs = (
                            min(time_range_secs[0], duration_secs),
                            max(time_range_secs[1], duration_secs),
                        )


if __name__ == "__main__":
    benchmark_param("move_size", [1, 2, 3, 4])
    benchmark_param("max_solver_threads", [0, 1, 2, 4, 7])
    benchmark_param("enable_n2_move_filters_1_and_2", [True, False])
    benchmark_param("enable_n3_move_filters_1_and_2", [True, False])
    benchmark_param("enable_n3_move_filters_3_and_4", [True, False])
    benchmark_param(
        "tactics",
        [
            Tactics.from_str(s)
            for s in [
                "se;s;ds;sp",
                "se;s;ds;a;sp",
                "se;s;ds;bti;sp",
                "se;s;ds;c2b;sp",
                "se;s;ds;cs;sp",
                "se;s;ds;css;sp",
                "se;s;ds;eti;sp",
                "se;s;ds;pi;sp",
                "se;s;ds;pv;sp",
            ]
        ],
    )
    # benchmark_param("apply_theorem_11a", [False, True])
    # benchmark_param("apply_theorem_11b", [False, True])
    benchmark_param("ban_repeated_states", [False, True])
    benchmark_param("enable_corner_min_patterns", [False, True])
    benchmark_param("enable_edge_min_patterns", [False, True])
