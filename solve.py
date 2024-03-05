import json
import sys
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue
import z3
from logic import State, move_name, list_edges, list_centers, list_corners
from misc import print_stamped
import move_mapping


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


def z3_int(solver: z3.Optimize, name: str, low: int, high: int):
    """Create Z3 integer and add its value range to the solver. The range is
    inclusive on low and exclusive on high."""
    var = z3.Int(name)
    solver.add(var >= low)
    solver.add(var < high)
    return var


def solve_for_k(puzzle: State, k: int):
    """Solve a puzzle with a maximum number of moves. Return list of move names or nothing if not possible.
    Also returns, in both cases, the time it took to prepare the SAT model and the time it took to solve it."""
    prep_start = datetime.now()
    solver = z3.Optimize()

    # Nested list representing an n × n × n cube. The variables represent
    # the current location (or rotation if relevant) of that cubie in a state.
    states: list[
        list[
            list[
                list[
                    tuple[
                        tuple[z3.ArithRef, z3.ArithRef, z3.ArithRef],  # coordinate
                        z3.ArithRef | None,  # rotation
                    ]
                    | None  # internal cubie
                ]
            ]
        ]
    ] = [
        [
            [[None for _ in range(puzzle.n)] for _ in range(puzzle.n)]
            for _ in range(puzzle.n)
        ]
        for _ in range(k + 1)
    ]

    # Populate the states with Z3 variables.
    corners = list_corners(puzzle.n)
    centers = list_centers(puzzle.n)
    edges = list_edges(puzzle.n)
    for s in range(k + 1):
        for x, y, z in corners:
            states[s][x][y][z] = (
                (
                    z3_int(solver, f"corner({x},{y},{z}) s({s}) x", 0, puzzle.n),
                    z3_int(solver, f"corner({x},{y},{z}) s({s}) y", 0, puzzle.n),
                    z3_int(solver, f"corner({x},{y},{z}) s({s}) z", 0, puzzle.n),
                ),
                z3_int(solver, f"corner({x}{y}{z}) s({s}) r", 0, 3),
            )
        for x, y, z in centers:
            states[s][x][y][z] = (
                (
                    z3_int(solver, f"center({x},{y},{z}) s({s}) x", 0, puzzle.n),
                    z3_int(solver, f"center({x},{y},{z}) s({s}) y", 0, puzzle.n),
                    z3_int(solver, f"center({x},{y},{z}) s({s}) z", 0, puzzle.n),
                ),
                None,
            )
        for x, y, z in edges:
            states[s][x][y][z] = (
                (
                    z3_int(solver, f"edge({x},{y},{z}) s({s}) x", 0, puzzle.n),
                    z3_int(solver, f"edge({x},{y},{z}) s({s}) y", 0, puzzle.n),
                    z3_int(solver, f"edge({x},{y},{z}) s({s}) z", 0, puzzle.n),
                ),
                z3_int(solver, f"edge({x}{y}{z}) s({s}) r", 0, 3),
            )

    # Variables which together indicate the move at each step.
    ma = [z3_int(solver, f"s({s}) ma", 0, 2) for s in range(k)]
    mi = [z3_int(solver, f"s({s}) mi", 0, puzzle.n + 1) for s in range(k)]
    md = [z3_int(solver, f"s({s}) md", 0, 2) for s in range(k)]

    def fix_state(s: int, state: State):
        """Return condition of a state being equal to a state object."""
        conditions = []

        for x in range(puzzle.n):
            for y in range(puzzle.n):
                for z in range(puzzle.n):
                    v = states[s][x][y][z]
                    if v is not None:
                        coords, rot = v
                        conditions.append(coords[0] == state.coords[x][y][z][0])
                        conditions.append(coords[1] == state.coords[x][y][z][1])
                        conditions.append(coords[2] == state.coords[x][y][z][2])
                        if rot is not None:
                            conditions.append(rot == state.rots[x][y][z])

        return z3.And(conditions)

    # Fix the first state to the puzzle state.
    solver.add(fix_state(0, puzzle))

    # Fix the last state to a finished state.
    finished = State.finished(puzzle.n)
    solver.add(fix_state(0, finished))

    def identical_states(s1: int, s2: int):
        """Return condition of two states being equal."""
        conditions = []

        for x in range(puzzle.n):
            for y in range(puzzle.n):
                for z in range(puzzle.n):
                    v1 = states[s1][x][y][z]
                    v2 = states[s2][x][y][z]
                    if v1 is not None:
                        assert v2 is not None
                        coords1, rot1 = v1
                        coords2, rot2 = v2
                        conditions.append(coords1[0] == coords2[0])
                        conditions.append(coords1[1] == coords2[1])
                        conditions.append(coords1[2] == coords2[2])
                        if rot1 is not None:
                            assert rot2 is not None
                            conditions.append(rot1 == rot2)

        return z3.And(conditions)

    # Restrict color states when move is nothing.
    for s in range(k):
        solver.add(z3.Or(mi[s] != puzzle.n, identical_states(s, s + 1)))

    # Only allow nothing move when complete.
    for s in range(len(states) - 1):
        solver.add(z3.Or(mi[s] != puzzle.n, fix_state(s, finished)))

    # # Restrict color states using pre-generated move mappings file.
    # mappings = move_mapping.load(puzzle.n)
    # for s in range(len(states) - 1):
    #     for ma in mappings:
    #         for mi in mappings[ma]:
    #             for md in mappings[ma][mi]:
    #                 conditions = []
    #                 if ma is not None:
    #                     conditions.append(ma[s] == ma)
    #                 if mi is not None:
    #                     conditions.append(mi[s] == mi)
    #                 if md is not None:
    #                     conditions.append(md[s] == md)

    #                 consequences = [
    #                     states[s + 1][puzzle.cell_idx(f_in, y_in, x_in)]
    #                     == states[s][puzzle.cell_idx(f_out, y_out, x_out)]
    #                     for (f_in, y_in, x_in), (f_out, y_out, x_out) in mappings[ma][
    #                         mi
    #                     ][md]
    #                 ]

    #                 if len(consequences) > 0:
    #                     solver.add(
    #                         z3.Or(
    #                             z3.Or([z3.Not(cond) for cond in conditions]),
    #                             z3.And(consequences),
    #                         )
    #                     )

    # If between 1 and n moves ago we made a turn at an index and axis, a different axis has to have been turned in the meantime.
    for s in range(1, len(states) - 1):
        for r in range(1, min(puzzle.n + 1, s + 1)):
            solver.add(
                z3.Or(
                    ma[s - r] != ma[s],
                    mi[s - r] != mi[s],
                    z3.Or([ma[s] != ma[p] for p in range(s - r + 1, s)]),
                )
            )

    # All subsequent moves in the same axis happen in ascending order of index.
    for s in range(1, len(states) - 1):
        solver.add(
            z3.Or(
                ma[s - 1] != ma[s],
                z3.And(
                    [
                        z3.And([z3.Or(mi[s - 1] != b, mi[s] != c) for c in range(1, b)])
                        for b in range(puzzle.n, 1, -1)
                    ]
                ),
            )
        )

    # States cannot be repeated.
    for s1 in range(len(states)):
        for s2 in range(s1 + 1, len(states)):
            solver.add(z3.Not(identical_states(s1, s2)))

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
            ma = model.get_interp(ma[s]).as_long()
            mi = model.get_interp(mi[s]).as_long()
            md = model.get_interp(md[s]).as_long()
            moves.append(move_name(puzzle.n, ma, mi, md))
        assert len(moves) == k

    return moves, prep_time, solve_time


def solve(files: list[str], process_count: int):
    """Solve a list of puzzles, efficiently distributing tasks among multiple processes."""

    with Manager() as manager:
        # List of puzzles to solve.
        puzzles = [State.from_str(open(file, "r").read()) for file in files]

        # List of n values for each of the puzzles.
        ns = [puzzles[i].n for i in range(len(puzzles))]

        # Generate any missing move mappings.
        for n in ns:
            move_mapping.generate(n)

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
                solution, prep_time, solve_time = solve_for_k(puzzle, k)
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

                with open(f"{files[i]}.solution", "w") as solution_file:
                    solution_file.write(json.dumps(result, indent=4))


# e.g. python solve.py ./puzzles/n2-random10.txt
if __name__ == "__main__":
    solve([sys.argv[1]], cpu_count())
