import os
import time
from multiprocessing import Manager, Process
from multiprocessing.managers import ValueProxy

from config import SolveConfig, Tactics
from puzzle import Puzzle
from solve import solve
from stats import SolveStats
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
    with Manager() as manager:
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

            def write_line(
                puzzle_name, parameter_value, prep_time, solve_time, k, message
            ):
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
                time_range: tuple[float, float] | None = None

                for parameter_value in parameter_values:
                    print_stamped(f"value {parameter_value}...")

                    # Set the parameter value in the config.
                    config = SolveConfig(**{parameter_name: parameter_value})

                    # Set timeout to max of: 3x the fastest run,
                    # 1x the slowest run, or a minimum timeout.
                    timeout_secs = (
                        None
                        if time_range is None
                        else max(
                            time_range[1],
                            time_range[0] * TIMEOUT_FACTOR,
                            MIN_TIMEOUT_SECS,
                        )
                    )

                    def solve_wrapper(
                        puzzle: Puzzle,
                        config: SolveConfig,
                        output: ValueProxy[SolveStats | None],
                    ):
                        """Wrapper function to run solve in a separate process."""
                        result = solve(puzzle, config, False, False)
                        output.set(result)

                    result: ValueProxy[SolveStats | None] = manager.Value(
                        "result", None
                    )
                    process = Process(
                        target=solve_wrapper,
                        args=(puzzle, config, result),
                    )

                    # Start the solver and wait with timeout.
                    start = time.time()
                    process.start()
                    if timeout_secs is not None:
                        process.join(timeout_secs)
                    else:
                        process.join()
                    if process.is_alive():
                        process.kill()
                        process.join()
                    assert process.exitcode is not None
                    if process.exitcode > 0:
                        raise Exception(f"process exited with code {process.exitcode}")
                    process.close()

                    # Write the result to the CSV file.
                    if result.value is None:
                        write_line(
                            puzzle.name,
                            parameter_value,
                            "",
                            "",
                            "",
                            f"timeout after {timeout_secs}s",
                        )
                    elif result.value.k() is None:
                        write_line(
                            puzzle.name,
                            parameter_value,
                            result.value.total_prep_time().total_seconds(),
                            result.value.total_solve_time().total_seconds(),
                            "",
                            "no solution found",
                        )
                    else:
                        write_line(
                            puzzle.name,
                            parameter_value,
                            result.value.total_prep_time().total_seconds(),
                            result.value.total_solve_time().total_seconds(),
                            result.value.k(),
                            "",
                        )

                    # Update the time range.
                    duration = time.time() - start
                    if time_range is None:
                        time_range = (duration, duration)
                    else:
                        time_range = (
                            min(time_range[0], duration),
                            max(time_range[1], duration),
                        )


if __name__ == "__main__":
    benchmark_param("move_size", [1, 2, 3, 4])
    benchmark_param("max_solver_threads", [0, 1, 2, 4, 7])
    benchmark_param("enable_n2_move_filters_1_and_2", [False, True])
    benchmark_param("enable_n3_move_filters_1_and_2", [False, True])
    benchmark_param("enable_n3_move_filters_3_and_4", [False, True])
    benchmark_param(
        "tactics",
        [
            Tactics.from_str(s)
            for s in [
                "se;s",
                "se;s;sp",
                "se;s;cs",
                "se;s;cs;sp",
                "se;s;ds",
                "se;s;ds;sp",
                "se;s;ds;cs;sp",
                "se;s;pv;pi;sp",
                "pa;se;s;pv;pi;sp",
            ]
        ],
    )
    # benchmark_param("apply_theorem_11a", [False, True])
    # benchmark_param("apply_theorem_11b", [False, True])
    benchmark_param("ban_repeated_states", [False, True])
    benchmark_param("enable_corner_min_patterns", [False, True])
    benchmark_param("enable_edge_min_patterns", [False, True])
