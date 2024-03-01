import json
import sys
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue
import z3
from misc import print_stamped, State, move_name
import move_mapping
import end_pattern_database


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


def z3_int_with_range(solver: z3.Optimize, name: str, low: int, high: int):
    """Create Z3 integer and add its value range to the solver. The range is
    inclusive on low and exclusive on high."""
    var = z3.Int(name)
    solver.add(var >= low)
    solver.add(var < high)
    return var


def solve_for_k(puzzle: State, k: int, pattern_depth: int):
    """Solve a puzzle with a maximum number of moves. Return list of move names or nothing if not possible.
    Also returns, in both cases, the time it took to prepare the SAT model and the time it took to solve it."""
    prep_start = datetime.now()
    solver = z3.Optimize()

    # Variables which together represent the state of the cube at each step.
    corner_locations = [
        [z3_int_with_range(solver, f"corner_loc({c}) s({s})", 0, 8) for c in range(8)]
        for s in range(k + 1)
    ]
    corner_rotations = [
        [z3_int_with_range(solver, f"corner_rot({c}) s({s})", 0, 3) for c in range(8)]
        for s in range(k + 1)
    ]
    edge_locations = [
        [z3_int_with_range(solver, f"edge_loc({e}) s({s})", 0, 12) for e in range(12)]
        for s in range(k + 1)
    ]
    edge_rotations = [
        [z3_int_with_range(solver, f"edge_rot({e}) s({s})", 0, 2) for e in range(12)]
        for s in range(k + 1)
    ]
    center_locations = [
        [z3_int_with_range(solver, f"center_loc({f}) s({s})", 0, 6) for f in range(6)]
        for s in range(k + 1)
    ]

    # Variables which together indicate the move at each step.
    move_axes = [z3_int_with_range(solver, f"ma({s})", 0, 2) for s in range(k)]
    move_indices = [
        z3_int_with_range(solver, f"mi({s})", 0, puzzle.n) for s in range(k)
    ]
    move_directions = [z3_int_with_range(solver, f"md({s})", 0, 2) for s in range(k)]

    # Restrict colors of first state to starting state.
    solver.add(
        z3.And(
            [
                states[0][puzzle.cell_idx(f, y, x)] == puzzle.get_color(f, y, x)
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
                states[s][puzzle.cell_idx(f, y, x)] == f
                for f in range(6)
                for y in range(puzzle.n)
                for x in range(puzzle.n)
            ]
        )

    # Restrict cube to be complete at the end.
    solver.add(is_complete(-1))

    # Restrict moves to valid moves.
    for s in range(len(states) - 1):
        solver.add(z3.And(move_axes[s] >= 0, move_axes[s] <= 2))
        solver.add(z3.And(move_indices[s] >= 0, move_indices[s] <= puzzle.n))
        solver.add(z3.And(move_directions[s] >= 0, move_directions[s] <= 2))

    # Restrict color states when move is nothing.
    for s in range(len(states) - 1):
        solver.add(
            z3.Or(
                move_indices[s] != puzzle.n,
                z3.And(
                    [
                        states[s][puzzle.cell_idx(f, y, x)]
                        == states[s + 1][puzzle.cell_idx(f, y, x)]
                        for f in range(6)
                        for y in range(puzzle.n)
                        for x in range(puzzle.n)
                    ]
                ),
            )
        )

    # Only allow nothing move when complete.
    for s in range(len(states) - 1):
        solver.add(z3.Or(move_indices[s] != puzzle.n, is_complete(s)))

    # Restrict color states using pre-generated move mappings file.
    mappings = move_mapping.load(puzzle.n)
    for s in range(len(states) - 1):
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
                        states[s + 1][puzzle.cell_idx(f_in, y_in, x_in)]
                        == states[s][puzzle.cell_idx(f_out, y_out, x_out)]
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
    for s in range(1, len(states) - 1):
        for r in range(1, min(puzzle.n + 1, s + 1)):
            solver.add(
                z3.Or(
                    move_axes[s - r] != move_axes[s],
                    move_indices[s - r] != move_indices[s],
                    z3.Or([move_axes[s] != move_axes[p] for p in range(s - r + 1, s)]),
                )
            )

    # All subsequent moves in the same axis happen in ascending order of index.
    for s in range(1, len(states) - 1):
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
    for s1 in range(len(states)):
        for s2 in range(s1 + 1, len(states)):
            solver.add(
                z3.Not(
                    z3.And(
                        [
                            states[s1][puzzle.cell_idx(f, y, x)]
                            == states[s2][puzzle.cell_idx(f, y, x)]
                            for f in range(6)
                            for y in range(puzzle.n)
                            for x in range(puzzle.n)
                        ]
                    )
                )
            )

    # Add restrictions for end pattern database.
    patterns = end_pattern_database.load(puzzle.n, pattern_depth)
    for state, remaining in patterns:
        for s in range(max(0, len(states) - remaining), len(states) - 1):
            solver.add(
                z3.Or(
                    [
                        states[s][puzzle.cell_idx(f, y, x)] != state.get_color(f, y, x)
                        for f in range(6)
                        for y in range(puzzle.n)
                        for x in range(puzzle.n)
                        if not state.is_unset(f, y, x)
                    ]
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
        for s in range(len(states) - 1):
            ma = model.get_interp(move_axes[s]).as_long()
            mi = model.get_interp(move_indices[s]).as_long()
            md = model.get_interp(move_directions[s]).as_long()
            moves.append(move_name(puzzle.n, ma, mi, md))
        assert len(moves) == k

    return moves, prep_time, solve_time


def solve(files: list[str], process_count: int, pattern_depth: int):
    """Solve a list of puzzles, efficiently distributing tasks among multiple processes."""

    with Manager() as manager:
        # List of puzzles to solve.
        puzzles = [State.from_str(open(file, "r").read()) for file in files]

        # List of n values for each of the puzzles.
        ns = [puzzles[i].n for i in range(len(puzzles))]

        # Generate any missing move mappings.
        for n in ns:
            move_mapping.generate(n)

        # Generate any missing end pattern databases.
        for n in ns:
            end_pattern_database.generate(n, pattern_depth)

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
        output: Queue[tuple[int, int, list[str] | None, timedelta, timedelta]] = (
            manager.Queue()
        )

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
                    k = k_prospects[i].pop(0)
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
                print_stamped(
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
                print_stamped(
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
                    print_stamped(
                        f"{files[i]}: found no solution with k ≤ {k_upperbounds[i]} to be possible in {total_solve_time} with {total_prep_time} prep time"
                    )
                else:
                    print_stamped(
                        f"{files[i]}: minimum k = {len(minimum)} found in {total_solve_time} with {total_prep_time} prep time"
                    )

                result = {
                    "k": len(minimum) if minimum is not None else "n/a",
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
                    "process_count": process_count,
                }

                with open(
                    f"{files[i]}.d{pattern_depth}.solution", "w"
                ) as solution_file:
                    solution_file.write(json.dumps(result, indent=4))


# e.g. python solve.py ./puzzles/n2-random10.txt {pattern_depth}
if __name__ == "__main__":
    solve([sys.argv[1]], cpu_count(), int(sys.argv[2]))
