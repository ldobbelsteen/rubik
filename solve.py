import json
import sys
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue
import z3
from misc import print_with_stamp, State, move_name
import move_mapping
import pattern_database


def k_upperbound(n: int):
    match n:
        case 1:
            return 0  # is always already solved
        case 2:
            return 11  # God's Number
        case 3:
            return 20  # God's Number
        case _:
            raise Exception(f"upperbound of {n} not set")


def solve_for_k(puzzle: State, k: int, pattern_depth: int):
    """Solve a puzzle with a maximum number of moves. Return list of move names or nothing if not possible.
    Also returns, in both cases, the time it took to prepare the SAT model and the time it took to solve it."""
    prep_start = datetime.now()
    solver = z3.Optimize()

    def cell_idx(f: int, y: int, x: int):
        """Convert the coordinates on a cube to a flat index."""
        return x + y * puzzle.n + f * puzzle.n * puzzle.n

    # Variables indicating the colors of the cube cells at each state.
    colors = [
        [z3.Int(f"c({s},{i})") for i in range(6 * puzzle.n * puzzle.n)]
        for s in range(k + 1)
    ]

    # Variables which together indicate the move at each state.
    move_axes = [z3.Int(f"ma({s})") for s in range(len(colors) - 1)]
    move_indices = [z3.Int(f"mi({s})") for s in range(len(colors) - 1)]
    move_directions = [z3.Int(f"md({s})") for s in range(len(colors) - 1)]

    # Restrict color domains to the six colors.
    for s in range(len(colors)):
        solver.add(
            z3.And(
                [
                    z3.And(
                        colors[s][cell_idx(f, y, x)] >= 0,
                        colors[s][cell_idx(f, y, x)] < 6,
                    )
                    for f in range(6)
                    for y in range(puzzle.n)
                    for x in range(puzzle.n)
                ]
            )
        )

    # Restrict colors of first state to starting state.
    solver.add(
        z3.And(
            [
                colors[0][cell_idx(f, y, x)] == puzzle.get_color(f, y, x)
                for f in range(6)
                for y in range(puzzle.n)
                for x in range(puzzle.n)
            ]
        )
    )

    def is_complete(s: int):
        """Return restriction on whether a state is complete."""
        return z3.And(
            [
                colors[s][cell_idx(f, y, x)] == f
                for f in range(6)
                for y in range(puzzle.n)
                for x in range(puzzle.n)
            ]
        )

    # Restrict cube to be complete at the end.
    solver.add(is_complete(-1))

    # Restrict moves to valid moves.
    for s in range(len(colors) - 1):
        solver.add(z3.And(move_axes[s] >= 0, move_axes[s] <= 2))
        solver.add(z3.And(move_indices[s] >= 0, move_indices[s] <= puzzle.n))
        solver.add(z3.And(move_directions[s] >= 0, move_directions[s] <= 2))

    # Restrict color states when move is nothing.
    for s in range(len(colors) - 1):
        solver.add(
            z3.Or(
                move_indices[s] != puzzle.n,
                z3.And(
                    [
                        colors[s][cell_idx(f, y, x)] == colors[s + 1][cell_idx(f, y, x)]
                        for f in range(6)
                        for y in range(puzzle.n)
                        for x in range(puzzle.n)
                    ]
                ),
            )
        )

    # Only allow nothing move when complete.
    for s in range(len(colors) - 1):
        solver.add(z3.Or(move_indices[s] != puzzle.n, is_complete(s)))

    # Restrict color states using pre-generated move mappings file.
    mappings = move_mapping.load(puzzle.n)
    for s in range(len(colors) - 1):
        for ma in mappings:
            for mi in mappings[ma]:
                for md in mappings[ma][mi]:
                    conditions = []
                    if ma is not None:
                        conditions.append(move_axes[s] == ma)
                    if mi is not None:
                        conditions.append(move_indices[s] == mi)
                    if md is not None:
                        conditions.append(move_directions[s] == md)

                    consequences = [
                        colors[s + 1][cell_idx(f_in, y_in, x_in)]
                        == colors[s][cell_idx(f_out, y_out, x_out)]
                        for (f_in, y_in, x_in), (f_out, y_out, x_out) in mappings[ma][
                            mi
                        ][md]
                    ]

                    if len(consequences) > 0:
                        solver.add(
                            z3.Or(
                                z3.Or([z3.Not(cond) for cond in conditions]),
                                z3.And(consequences),
                            )
                        )

    # If between 1 and n moves ago we made a turn at an index and axis, a different axis has to have been turned in the meantime.
    for s in range(1, len(colors) - 1):
        for r in range(1, min(puzzle.n + 1, s + 1)):
            solver.add(
                z3.Or(
                    move_axes[s - r] != move_axes[s],
                    move_indices[s - r] != move_indices[s],
                    z3.Or([move_axes[s] != move_axes[p] for p in range(s - r + 1, s)]),
                )
            )

    # All subsequent moves in the same axis happen in ascending order of index.
    for s in range(1, len(colors) - 1):
        solver.add(
            z3.Or(
                move_axes[s - 1] != move_axes[s],
                z3.And(
                    [
                        z3.And(
                            [
                                z3.Or(move_indices[s - 1] != b, move_indices[s] != c)
                                for c in range(1, b)
                            ]
                        )
                        for b in range(puzzle.n, 1, -1)
                    ]
                ),
            )
        )

    # States cannot be repeated.
    for s1 in range(len(colors)):
        for s2 in range(s1 + 1, len(colors)):
            solver.add(
                z3.Not(
                    z3.And(
                        [
                            colors[s1][cell_idx(f, y, x)]
                            == colors[s2][cell_idx(f, y, x)]
                            for f in range(6)
                            for y in range(puzzle.n)
                            for x in range(puzzle.n)
                        ]
                    )
                )
            )

    # Add restrictions for pattern database.
    patterns = pattern_database.load(puzzle.n, pattern_depth)
    for state, minimum_remaining in patterns:
        for s in range(max(0, len(colors) - minimum_remaining), len(colors)):
            solver.add(
                z3.Not(
                    z3.And(
                        [
                            colors[s][cell_idx(f, y, x)] == state.get_color(f, y, x)
                            for f in range(6)
                            for y in range(puzzle.n)
                            for x in range(puzzle.n)
                        ]
                    )
                )
            )

    # Check model and return moves if sat.
    prep_time = datetime.now() - prep_start
    solve_start = datetime.now()
    res = solver.check()
    solve_time = datetime.now() - solve_start
    moves: list[str] | None = None

    if res == z3.sat:
        moves = []
        model = solver.model()
        for s in range(len(colors) - 1):
            ma = model.get_interp(move_axes[s]).as_long()
            mi = model.get_interp(move_indices[s]).as_long()
            md = model.get_interp(move_directions[s]).as_long()
            moves.append(move_name(puzzle.n, ma, mi, md))
        assert len(moves) == k

    return moves, prep_time, solve_time


def solve(files: list[str], process_count: int):
    """Solve a list of puzzles, efficiently distributing tasks among multiple processes."""

    pattern_depth = 0

    with Manager() as manager:
        # List of puzzles to solve.
        puzzles = [State(open(file, "r").read()) for file in files]

        # List of n values for each of the puzzles.
        ns = [puzzles[i].n for i in range(len(puzzles))]

        # Generate any missing move mappings.
        for n in ns:
            move_mapping.generate(n)

        # Generate any missing pattern databases.
        for n in ns:
            pattern_database.generate(n, pattern_depth)

        # List of upperbounds for k for each of the puzzles.
        k_upperbounds = [k_upperbound(ns[i]) for i in range(len(puzzles))]

        # Lists of prospects for k for each of the puzzles.
        k_prospects = [list(range(k_upperbounds[i] + 1)) for i in range(len(puzzles))]

        # List of currently found minimum size solutions for each of the puzzles.
        k_minima: list[list[str] | None] = [None for _ in range(len(puzzles))]

        # List of prep times for each tried prospect for each puzzle.
        prep_times: list[dict[int, timedelta]] = [{} for _ in range(len(puzzles))]

        # List of solving times for each tried prospect for each puzzle.
        solve_times: list[dict[int, timedelta]] = [{} for _ in range(len(puzzles))]

        # List of processes and their current tasks.
        processes: list[tuple[int, int, Process]] = []

        # Queue to output results onto.
        output: Queue[
            tuple[int, int, list[str] | None, timedelta, timedelta]
        ] = manager.Queue()

        def spawn_new_process():
            def solve_wrapper(
                puzzle: State,
                k: int,
                i: int,
                output: Queue[tuple[int, int, list[str] | None, timedelta, timedelta]],
            ):
                solution, prep_time, solve_time = solve_for_k(puzzle, k, pattern_depth)
                output.put((i, k, solution, prep_time, solve_time))

            for i in range(len(puzzles)):
                if len(k_prospects[i]) > 0:
                    median = (
                        len(k_prospects[i]) // 2
                    )  # Take median prospect, resulting in a kind of binary search.
                    k = k_prospects[i].pop(median)
                    process = Process(
                        target=solve_wrapper,
                        args=(
                            puzzles[i],
                            k,
                            i,
                            output,
                        ),
                    )
                    processes.append((i, k, process))
                    process.start()
                    break

        for _ in range(process_count):
            spawn_new_process()

        while len(processes) > 0:
            i, k, solution, prep_time, solve_time = output.get()
            prep_times[i][k] = prep_time
            solve_times[i][k] = solve_time

            if solution is None:
                print_with_stamp(
                    f"{files[i]}: unsat for k = {k} found in {solve_time} with {prep_time} prep time..."
                )
                k_prospects[i] = [p for p in k_prospects[i] if p > k]
                killed = 0
                for pi in range(len(processes) - 1, -1, -1):
                    if processes[pi][1] <= k:
                        processes.pop(pi)[2].kill()
                        killed += 1
                for _ in range(killed):
                    spawn_new_process()
            else:
                print_with_stamp(
                    f"{files[i]}: sat for k = {k} found in {solve_time} with {prep_time} prep time..."
                )
                current_minimum = k_minima[i]
                if current_minimum is None or k < len(current_minimum):
                    k_minima[i] = solution
                k_prospects[i] = [p for p in k_prospects[i] if p < k]
                killed = 0
                for pi in range(len(processes) - 1, -1, -1):
                    if processes[pi][1] >= k:
                        processes.pop(pi)[2].kill()
                        killed += 1
                for _ in range(killed):
                    spawn_new_process()

            if all([i != pi for pi, _, _ in processes]):
                minimum = k_minima[i]
                total_solve_time = sum(solve_times[i].values(), timedelta())
                total_prep_time = sum(prep_times[i].values(), timedelta())

                if minimum is None:
                    print_with_stamp(
                        f"{files[i]}: found no solution with k ≤ {k_upperbounds[i]} to be possible in {total_solve_time} with {total_prep_time} prep time"
                    )
                else:
                    print_with_stamp(
                        f"{files[i]}: minimum k = {len(minimum)} found in {total_solve_time} with {total_prep_time} prep time"
                    )

                result = {
                    "k": len(minimum) if minimum is not None else "n/a",
                    "pattern_depth": pattern_depth,
                    "moves": minimum if minimum is not None else "impossible",
                    "total_solve_time": str(total_solve_time),
                    "total_prep_time": str(total_prep_time),
                    "prep_time_per_k": {
                        k: str(t) for k, t in sorted(prep_times[i].items())
                    },
                    "solve_time_per_k": {
                        k: str(t) for k, t in sorted(solve_times[i].items())
                    },
                    "k_upperbound": k_upperbounds[i],
                }

                with open(files[i] + ".solution", "w") as solution_file:
                    solution_file.write(json.dumps(result, indent=4))


# e.g. python solve.py ./puzzles/n2-random10.txt ...
if __name__ == "__main__":
    solve(sys.argv[1:], cpu_count())
