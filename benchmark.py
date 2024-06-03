import argparse
import os

import florian
from config import SolveConfig, Tactics
from puzzle import Puzzle
from solve import solve
from tools import log_stamped

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


def load_benchmark_puzzles(only_n2: bool = False, only_n3: bool = False) -> list[Puzzle]:
    """Load all benchmark puzzles from the list of puzzles.."""
    if only_n2:
        return [Puzzle.from_file(name) for name in BENCHMARK_PUZZLES if 'n2' in name]
    elif only_n3:
        return [Puzzle.from_file(name) for name in BENCHMARK_PUZZLES if 'n3' in name]
    return [Puzzle.from_file(name) for name in BENCHMARK_PUZZLES]


def benchmark_param(parameter_name: str, parameter_values: list | None = None, only_n2: bool = False, only_n3: bool = False):
    """Benchmark the solve function for a list of parameter values. This can be
    used to determine which parameters are best.
    """
    if parameter_values is None:
        parameter_values = []

    log_stamped(f"benchmarking '{parameter_name}' parameter...")
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

        if parameter_name.lower() == "florian":
            for puzzle in [puzzle for puzzle in BENCHMARK_PUZZLES if 'n2' in puzzle]:
                log_stamped(f"puzzle {puzzle}...")
                prep_time = 0
                solve_time = 0
                number_of_moves = 1

                while True:
                    florian_instance = florian.Florian(puzzle, number_of_moves)
                    time_prep, time_solve, solve_result = florian_instance.solve()

                    prep_time += time_prep
                    solve_time += time_solve

                    if solve_result.__str__() == "sat":
                        break

                    number_of_moves += 1

                write_line(
                    puzzle,
                    parameter_name,
                    prep_time,
                    solve_time,
                    number_of_moves,
                    "",
                )
        else:
            for puzzle in load_benchmark_puzzles(only_n2, only_n3):
                log_stamped(f"puzzle {puzzle.name}...")
                time_range_secs: tuple[float, float] | None = None

                for parameter_value in parameter_values:
                    log_stamped(f"value {parameter_value}...")

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


def run(
    number_of_times: int,
    all_benchmarks: bool = False,
    move_size: bool = False,
    max_solver_threads: bool = False,
    enable_n2_move_filters_1_and_2: bool = False,
    enable_n3_move_filters_1_and_2: bool = False,
    enable_n3_move_filters_3_and_4: bool = False,
    tactics: bool = False,
    ban_repeated_states: bool = False,
    enable_corner_min_patterns: bool = False,
    enable_edge_min_patterns: bool = False,
    enable_minimal_moves_n2: bool = False,
    florian_benchmark: bool = False,
) -> None:
    """Run the benchmarks.

    @param number_of_times: The number of times to run the benchmarks.
    @param all_benchmarks: Whether to perform all benchmarks.
    @param move_size: Whether to perform the move size benchmark.
    @param max_solver_threads: Whether to perform the max solver threads benchmark.
    @param enable_n2_move_filters_1_and_2: Whether to perform the enable n2 move filters 1 and 2 benchmark.
    @param enable_n3_move_filters_1_and_2: Whether to perform the enable n3 move filters 1 and 2 benchmark.
    @param enable_n3_move_filters_3_and_4: Whether to perform the enable n3 move filters 3 and 4 benchmark.
    @param tactics: Whether to perform the tactics benchmark.
    @param ban_repeated_states: Whether to perform the ban repeated states benchmark.
    @param enable_corner_min_patterns: Whether to perform the enable corner min patterns benchmark.
    @param enable_edge_min_patterns: Whether to perform the enable edge min patterns benchmark.
    @param enable_minimal_moves_n2: Whether to perform the enable minimal moves n2 benchmark.
    @param florian_benchmark: Whether to perform the florian benchmark.
    """
    if not any(
        [
            all_benchmarks,
            move_size,
            max_solver_threads,
            enable_n2_move_filters_1_and_2,
            enable_n3_move_filters_1_and_2,
            enable_n3_move_filters_3_and_4,
            tactics,
            ban_repeated_states,
            enable_corner_min_patterns,
            enable_edge_min_patterns,
            enable_minimal_moves_n2,
            florian_benchmark,
        ]
    ):
        raise ValueError("No benchmarks selected.")

    for iteration in range(number_of_times):
        log_stamped(f"Starting iteration {iteration + 1}")
        if all_benchmarks or move_size:
            log_stamped("Running move size benchmark...")
            benchmark_param("move_size", [1, 2, 3, 4])
            log_stamped("Finished move size benchmark")

        if all_benchmarks or max_solver_threads:
            log_stamped("Running max solver threads benchmark...")
            benchmark_param("max_solver_threads", [0, 1, 2, 4, 7])
            log_stamped("Finished max solver threads benchmark")

        if all_benchmarks or enable_n2_move_filters_1_and_2:
            log_stamped("Running enable n2 move filters 1 and 2 benchmark...")
            benchmark_param("enable_n2_move_filters_1_and_2", [True, False], True)
            log_stamped("Finished enable n2 move filters 1 and 2 benchmark")

        if all_benchmarks or enable_n3_move_filters_1_and_2:
            log_stamped("Running enable n3 move filters 1 and 2 benchmark...")
            benchmark_param("enable_n3_move_filters_1_and_2", [True, False], only_n3=True)
            log_stamped("Finished enable n3 move filters 1 and 2 benchmark")

        if all_benchmarks or enable_n3_move_filters_3_and_4:
            log_stamped("Running enable n3 move filters 3 and 4 benchmark...")
            benchmark_param("enable_n3_move_filters_3_and_4", [True, False], only_n3=True)
            log_stamped("Finished enable n3 move filters 3 and 4 benchmark")

        if all_benchmarks or tactics:
            log_stamped("Running tactics benchmark...")
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
                        "se;s;ds;eti;sp",
                        "se;s;ds;pi;sp",
                        "se;s;ds;pv;sp",
                    ]
                ],
            )
            log_stamped("Finished tactics benchmark")

        if all_benchmarks or ban_repeated_states:
            log_stamped("Running ban repeated states benchmark...")
            benchmark_param("ban_repeated_states", [False, True])
            log_stamped("Finished ban repeated states benchmark")

        if all_benchmarks or enable_corner_min_patterns:
            log_stamped("Running enable corner min patterns benchmark...")
            benchmark_param("enable_corner_min_patterns", [False, True])
            log_stamped("Finished enable corner min patterns benchmark")

        if all_benchmarks or enable_edge_min_patterns:
            log_stamped("Running enable edge min patterns benchmark...")
            benchmark_param("enable_edge_min_patterns", [False, True])
            log_stamped("Finished enable edge min patterns benchmark")

        if all_benchmarks or enable_minimal_moves_n2:
            log_stamped("Running enable minimal moves n2 benchmark...")
            benchmark_param("enable_minimal_moves_n2", [False, True], True)
            log_stamped("Finished enable minimal moves n2 benchmark")

        if all_benchmarks or florian_benchmark:
            log_stamped("Running florian benchmark...")
            benchmark_param("florian")
            log_stamped("Finished florian benchmark")

        log_stamped(f"Finished iteration {iteration + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_times", type=int, default=10,
                        help="The number of times to run the benchmarks.")
    parser.add_argument("--all", type=bool, default=False,
                        help="Whether to perform all benchmarks.")
    parser.add_argument("--move_size", type=bool, default=False,
                        help="Whether to perform the move size benchmark.")
    parser.add_argument("--max_solver_threads", type=bool, default=False,
                        help="Whether to perform the max solver threads benchmark.")
    parser.add_argument("--enable_n2_move_filters_1_and_2", type=bool, default=False,
                        help="Whether to perform the enable n2 move filters 1 and 2 benchmark.")
    parser.add_argument("--enable_n3_move_filters_1_and_2", type=bool, default=False,
                        help="Whether to perform the enable n3 move filters 1 and 2 benchmark.")
    parser.add_argument("--enable_n3_move_filters_3_and_4", type=bool, default=False,
                        help="Whether to perform the enable n3 move filters 3 and 4 benchmark.")
    parser.add_argument("--tactics", type=bool, default=False,
                        help="Whether to perform the tactics benchmark.")
    parser.add_argument("--ban_repeated_states", type=bool, default=False,
                        help="Whether to perform the ban repeated states benchmark.")
    parser.add_argument("--enable_corner_min_patterns", type=bool, default=False,
                        help="Whether to perform the enable corner min patterns benchmark.")
    parser.add_argument("--enable_edge_min_patterns", type=bool, default=False,
                        help="Whether to perform the enable edge min patterns benchmark.")
    parser.add_argument("--enable_minimal_moves_n2", type=bool, default=False,
                        help="Whether to perform the enable minimal moves n2 benchmark.")
    parser.add_argument("--florian", type=bool, default=False,
                        help="Whether to perform the florian benchmark.")
    args = parser.parse_args()

    run(
        args.number_of_times,
        args.all,
        args.move_size,
        args.max_solver_threads,
        args.enable_n2_move_filters_1_and_2,
        args.enable_n3_move_filters_1_and_2,
        args.enable_n3_move_filters_3_and_4,
        args.tactics,
        args.ban_repeated_states,
        args.enable_corner_min_patterns,
        args.enable_edge_min_patterns,
        args.enable_minimal_moves_n2,
        args.florian,
    )
