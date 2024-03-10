import json
import sys
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue
import z3
from puzzle import Puzzle, move_name
from misc import print_stamped


def k_upperbound(n: int):
    match n:
        case 2:
            return 11  # God's Number
        case 3:
            return 20  # God's Number
        case _:
            raise Exception(f"k upperbound of {n} not set")


def z3_int(solver: z3.Optimize, name: str, low: int, high: int):
    """Create Z3 integer and add its value range to the solver. The range is
    inclusive on low and exclusive on high."""
    var = z3.Int(name)
    solver.add(var >= low)
    solver.add(var < high)
    return var


def next_x_restriction(
    n: int,
    next_x: z3.ArithRef,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
):
    return z3.If(
        z3.And(ma == 0, mi == y),
        z3.If(
            md == 0,
            next_x == z,
            z3.If(md == 1, next_x == n - 1 - z, next_x == n - 1 - x),
        ),
        z3.If(
            z3.And(ma == 2, mi == z),
            z3.If(
                md == 0,
                next_x == y,
                z3.If(md == 1, next_x == n - 1 - y, next_x == n - 1 - x),
            ),
            next_x == x,
        ),
    )


def next_y_restriction(
    n: int,
    next_y: z3.ArithRef,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
):
    return z3.If(
        z3.And(ma == 1, mi == x),
        z3.If(
            md == 0,
            next_y == n - 1 - z,
            z3.If(md == 1, next_y == z, next_y == n - 1 - y),
        ),
        z3.If(
            z3.And(ma == 2, mi == z),
            z3.If(
                md == 0,
                next_y == n - 1 - x,
                z3.If(md == 1, next_y == x, next_y == n - 1 - y),
            ),
            next_y == y,
        ),
    )


def next_z_restriction(
    n: int,
    next_z: z3.ArithRef,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
):
    return z3.If(
        z3.And(ma == 0, mi == y),
        z3.If(
            md == 0,
            next_z == n - 1 - x,
            z3.If(md == 1, next_z == x, next_z == n - 1 - z),
        ),
        z3.If(
            z3.And(ma == 1, mi == x),
            z3.If(
                md == 0,
                next_z == next_z == y,
                z3.If(md == 1, next_z == n - 1 - y, next_z == n - 1 - z),
            ),
            next_z == z,
        ),
    )


def next_corner_r_restriction(
    next_r: z3.ArithRef,
    x: z3.ArithRef,
    z: z3.ArithRef,
    r: z3.ArithRef,
    c: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
):
    # Condition for next_r == (r + 1) % 3
    add_one = z3.If(r == 0, next_r == 1, z3.If(r == 1, next_r == 2, next_r == 0))

    # Condition for next_r == (r - 1) % 3
    minus_one = z3.If(r == 0, next_r == 2, z3.If(r == 1, next_r == 0, next_r == 1))

    return z3.If(
        md != 2,
        z3.If(
            z3.And(ma == 1, mi == x),
            z3.If(c, minus_one, add_one),
            z3.If(z3.And(ma == 2, mi == z), z3.If(c, add_one, minus_one), next_r == r),
        ),
        next_r == r,
    )


def next_corner_c_restriction(
    next_c: z3.BoolRef,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    c: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
):
    return z3.If(
        z3.And(
            md != 2,
            z3.Or(
                z3.And(ma == 0, mi == y),
                z3.And(ma == 1, mi == x),
                z3.And(ma == 2, mi == z),
            ),
        ),
        next_c != c,
        next_c == c,
    )


def next_edge_r_restriction(
    n: int,
    next_r: z3.BoolRef,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    r: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
):
    return z3.If(
        z3.And(
            md != 2,
            z3.Or(
                z3.And(ma == 2, mi == z),
                z3.And(
                    mi != 0,
                    mi != n - 1,
                    z3.Or(z3.And(ma == 0, mi == y), z3.And(ma == 1, mi == x)),
                ),
            ),
        ),
        next_r != r,
        next_r == r,
    )


def solve_for_k(puzzle: Puzzle, k: int):
    """Solve a puzzle with a maximum number of moves. Return list of move names or nothing if not possible.
    Also returns, in both cases, the time it took to prepare the SAT model and the time it took to solve it."""
    prep_start = datetime.now()
    solver = z3.Optimize()

    n = puzzle.n
    finished = Puzzle.finished(n)
    cubies = finished.cubies

    # Nested lists representing the n × n × n cube for each state.
    corners = [
        [
            (
                z3_int(solver, f"corner({x},{y},{z}) s({s}) x", 0, n),
                z3_int(solver, f"corner({x},{y},{z}) s({s}) y", 0, n),
                z3_int(solver, f"corner({x},{y},{z}) s({s}) z", 0, n),
                z3_int(solver, f"corner({x},{y},{z}) s({s}) r", 0, 3),
                z3.Bool(f"corner({x},{y},{z}) s({s}) c"),
            )
            for x, y, z in cubies.corners
        ]
        for s in range(k + 1)
    ]
    centers = [
        [
            (
                z3_int(solver, f"center({x},{y},{z}) s({s}) x", 0, n),
                z3_int(solver, f"center({x},{y},{z}) s({s}) y", 0, n),
                z3_int(solver, f"center({x},{y},{z}) s({s}) z", 0, n),
            )
            for x, y, z in cubies.centers
        ]
        for s in range(k + 1)
    ]
    edges = [
        [
            (
                z3_int(solver, f"edge({x},{y},{z}) s({s}) x", 0, n),
                z3_int(solver, f"edge({x},{y},{z}) s({s}) y", 0, n),
                z3_int(solver, f"edge({x},{y},{z}) s({s}) z", 0, n),
                z3.Bool(f"edge({x},{y},{z}) s({s}) r"),
            )
            for x, y, z in cubies.edges
        ]
        for s in range(k + 1)
    ]

    # Variables which together indicate the move at each state.
    mas = [z3_int(solver, f"s({s}) ma", 0, 3) for s in range(k)]
    mis = [z3_int(solver, f"s({s}) mi", 0, n + 1) for s in range(k)]
    mds = [z3_int(solver, f"s({s}) md", 0, 3) for s in range(k)]

    def fix_state(s: int, puzzle: Puzzle):
        """Return conditions of a state being equal to a puzzle object."""
        conds: list[z3.BoolRef | bool] = []
        for i, (x, y, z, r, c) in enumerate(corners[s]):
            px, py, pz, pr, pc = puzzle.corners[i]
            conds.extend([x == px, y == py, z == pz, r == pr, c == pc])
        for i, (x, y, z) in enumerate(centers[s]):
            px, py, pz = puzzle.centers[i]
            conds.extend([x == px, y == py, z == pz])
        for i, (x, y, z, r) in enumerate(edges[s]):
            px, py, pz, pr = puzzle.edges[i]
            conds.extend([x == px, y == py, z == pz, r == pr])
        return conds

    def identical_states(s1: int, s2: int):
        """Return conditions of two states being equal."""
        conds: list[z3.BoolRef | bool] = []
        for i, (x1, y1, z1, r1, c1) in enumerate(corners[s1]):
            x2, y2, z2, r2, c2 = corners[s2][i]
            conds.extend([x1 == x2, y1 == y2, z1 == z2, r1 == r2, c1 == c2])
        for i, (x1, y1, z1) in enumerate(centers[s1]):
            x2, y2, z2 = centers[s2][i]
            conds.extend([x1 == x2, y1 == y2, z1 == z2])
        for i, (x1, y1, z1, r1) in enumerate(edges[s1]):
            x2, y2, z2, r2 = edges[s2][i]
            conds.extend([x1 == x2, y1 == y2, z1 == z2, r1 == r2])
        return conds

    # Fix the first state to the puzzle state.
    solver.add(z3.And(fix_state(0, puzzle)))

    # Fix the last state to the finished state.
    solver.add(z3.And(fix_state(-1, finished)))

    # Restrict color states when move is nothing.
    for s in range(k):
        solver.add(z3.Or(mis[s] != n, z3.And(identical_states(s, s + 1))))

    # Only allow nothing move when complete.
    for s in range(k):
        solver.add(z3.Or(mis[s] != n, z3.And(fix_state(s, finished))))

    # Restrict cubicle states according to moves.
    for s in range(k):
        ma, mi, md = mas[s], mis[s], mds[s]

        for i, (x, y, z, r, c) in enumerate(corners[s]):
            next_x, next_y, next_z, next_r, next_c = corners[s + 1][i]
            solver.add(next_x_restriction(n, next_x, x, y, z, ma, mi, md))
            solver.add(next_y_restriction(n, next_y, x, y, z, ma, mi, md))
            solver.add(next_z_restriction(n, next_z, x, y, z, ma, mi, md))
            solver.add(next_corner_r_restriction(next_r, x, z, r, c, ma, mi, md))
            solver.add(next_corner_c_restriction(next_c, x, y, z, c, ma, mi, md))

        for i, (x, y, z) in enumerate(centers[s]):
            next_x, next_y, next_z = centers[s + 1][i]
            solver.add(next_x_restriction(n, next_x, x, y, z, ma, mi, md))
            solver.add(next_y_restriction(n, next_y, x, y, z, ma, mi, md))
            solver.add(next_z_restriction(n, next_z, x, y, z, ma, mi, md))

        for i, (x, y, z, r) in enumerate(edges[s]):
            next_x, next_y, next_z, next_r = edges[s + 1][i]
            solver.add(next_x_restriction(n, next_x, x, y, z, ma, mi, md))
            solver.add(next_y_restriction(n, next_y, x, y, z, ma, mi, md))
            solver.add(next_z_restriction(n, next_z, x, y, z, ma, mi, md))
            solver.add(next_edge_r_restriction(n, next_r, x, y, z, r, ma, mi, md))

    # If between 1 and n moves ago we made a turn at an index and axis, a different axis has to have been turned in the meantime.
    for s in range(1, k):
        for rs in range(1, min(n + 1, s + 1)):
            solver.add(
                z3.Or(
                    mas[s - rs] != mas[s],
                    mis[s - rs] != mis[s],
                    z3.Or([mas[s] != mas[p] for p in range(s - rs + 1, s)]),
                )
            )

    # All subsequent moves in the same axis happen in ascending order of index.
    for s in range(1, k):
        solver.add(
            z3.Or(
                mas[s - 1] != mas[s],
                z3.And(
                    [
                        z3.And(
                            [z3.Or(mis[s - 1] != b, mis[s] != c) for c in range(1, b)]
                        )
                        for b in range(n, 1, -1)
                    ]
                ),
            )
        )

    # States cannot be repeated.
    for s1 in range(k + 1):
        for s2 in range(s1 + 1, k + 1):
            solver.add(z3.Not(z3.And(identical_states(s1, s2))))

    # Check model and return moves if sat.
    prep_time = datetime.now() - prep_start
    solve_start = datetime.now()
    res = solver.check()
    solve_time = datetime.now() - solve_start
    moves: list[str] | None = None

    if res == z3.sat:
        moves = []
        model = solver.model()
        for s in range(k):
            moves.append(
                move_name(
                    n,
                    model.get_interp(mas[s]).as_long(),
                    model.get_interp(mis[s]).as_long(),
                    model.get_interp(mds[s]).as_long(),
                )
            )
        assert len(moves) == k

    return moves, prep_time, solve_time


def solve(files: list[str], process_count: int):
    """Solve a list of puzzles, efficiently distributing tasks among multiple processes."""

    with Manager() as manager:
        # List of puzzles to solve.
        puzzles = [Puzzle.from_file(file) for file in files]

        # List of upperbounds for k for each of the puzzles.
        k_upperbounds = [k_upperbound(puzzles[i].n) for i in range(len(puzzles))]

        # Lists of prospects for k for each of the puzzles (starting with [0, k]).
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
                puzzle: Puzzle,
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


# e.g. python solve.py ./puzzles/n2-random7.txt
if __name__ == "__main__":
    solve([sys.argv[1]], cpu_count())
