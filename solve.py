import argparse
import json
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue

import z3

from misc import print_stamped
from puzzle import Puzzle, default_k_upperbound, move_name
from sym_move_seqs import MoveSequence, load


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
            z3.If(md == 1, next_x == (n - 1) - z, next_x == (n - 1) - x),
        ),
        z3.If(
            z3.And(ma == 2, mi == z),
            z3.If(
                md == 0,
                next_x == y,
                z3.If(md == 1, next_x == (n - 1) - y, next_x == (n - 1) - x),
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
            next_y == (n - 1) - z,
            z3.If(md == 1, next_y == z, next_y == (n - 1) - y),
        ),
        z3.If(
            z3.And(ma == 2, mi == z),
            z3.If(
                md == 0,
                next_y == (n - 1) - x,
                z3.If(md == 1, next_y == x, next_y == (n - 1) - y),
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
            next_z == (n - 1) - x,
            z3.If(md == 1, next_z == x, next_z == (n - 1) - z),
        ),
        z3.If(
            z3.And(ma == 1, mi == x),
            z3.If(
                md == 0,
                next_z == y,
                z3.If(md == 1, next_z == (n - 1) - y, next_z == (n - 1) - z),
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
                    mi == 1,
                    z3.Or(z3.And(ma == 0, mi == y), z3.And(ma == 1, mi == x)),
                ),
            ),
        ),
        next_r != r,
        next_r == r,
    )


def solve_for_k(
    puzzle: Puzzle, k: int, sym_move_depth: int, disallowed: list[MoveSequence] = []
):
    """Compute the optimal solution for a puzzle with a maximum number of moves k.
    Returns list of moves or nothing if impossible. In both cases, also returns the time
    it took to prepare the SAT model and the time it took to solve it."""
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
    mis = [z3_int(solver, f"s({s}) mi", 0, n) for s in range(k)]
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
            solver.add(next_edge_r_restriction(next_r, x, y, z, r, ma, mi, md))

    # If we make a move at an index and axis, we cannot make a move at the same index
    # and axis for the next n moves, unless a different axis has been turned in the
    # meantime.
    for s in range(k - 1):
        solver.add(
            z3.And(
                [
                    z3.Or(
                        mas[f] != mas[s],
                        mis[f] != mis[s],
                        z3.Or([mas[s] != mas[b] for b in range(s + 1, f)]),
                    )
                    for f in range(s + 1, min(s + n + 1, k))
                ]
            )
        )

    # All subsequent moves in the same axis happen in ascending order of index.
    for s in range(k - 1):
        solver.add(z3.Or(mas[s] != mas[s + 1], mis[s] < mis[s + 1]))

    # Two subsequent center half moves happen in ascending order of axis.
    if n == 3:
        for s in range(k - 1):
            solver.add(
                z3.Or(
                    mis[s] != 1,
                    mis[s + 1] != 1,
                    mds[s] != 2,
                    mds[s + 1] != 2,
                    mas[s] < mas[s + 1],
                )
            )

    # States cannot be repeated.
    for s1 in range(k + 1):
        for s2 in range(s1 + 1, k + 1):
            solver.add(z3.Not(z3.And(identical_states(s1, s2))))

    def disallow_move_sequence(ms: MoveSequence):
        """Return conditions of a move sequence not being allowed."""
        return z3.And(
            [
                z3.Not(
                    z3.And(
                        [
                            z3.And(
                                mas[start + i] == ma,
                                mis[start + i] == mi,
                                mds[start + i] == md,
                            )
                            for i, (ma, mi, md) in enumerate(ms)
                        ]
                    )
                )
                for start in range(k - len(ms) + 1)
            ]
        )

    # Disallow the move sequences from the parameters.
    for seq in disallowed:
        solver.add(disallow_move_sequence(seq))

    # Disallow symmetric move sequences up to the specified depth.
    for seq, syms in load(n, sym_move_depth).items():
        for sym in syms:
            solver.add(disallow_move_sequence(sym))

    # Check model and return moves if sat.
    prep_time = datetime.now() - prep_start
    solve_start = datetime.now()
    res = solver.check()
    solve_time = datetime.now() - solve_start
    moves: MoveSequence | None = None

    if res == z3.sat:
        model = solver.model()
        moves = tuple(
            (
                model.get_interp(mas[s]).as_long(),
                model.get_interp(mis[s]).as_long(),
                model.get_interp(mds[s]).as_long(),
            )
            for s in range(k)
        )
        assert len(moves) == k

    return moves, prep_time, solve_time


def solve(
    path: str,
    sym_move_depth: int,
    max_processes: int,
    disable_stats_file: bool,
) -> tuple[MoveSequence | None, dict]:
    """Compute the optimal solution for a puzzle in parallel for all possible values
    of k within the upperbound. Returns the solution and a dict containing statistics
    of the solving process."""
    print_stamped(f"solving '{path}'...")

    puzzle = Puzzle.from_file(path)
    k_upperbound = default_k_upperbound(puzzle.n)

    with Manager() as manager:
        k_prospects = list(range(k_upperbound + 1))
        optimal_solution: MoveSequence | None = None

        # Times for each tried k.
        prep_times: dict[int, timedelta] = {}
        solve_times: dict[int, timedelta] = {}

        # List of running processes and their k.
        processes: list[tuple[Process, int]] = []

        # Queue for the processes to output results onto.
        output: Queue[tuple[int, MoveSequence | None, timedelta, timedelta]] = (
            manager.Queue()
        )

        def spawn_new_process():
            def solve_for_k_wrapper(
                puzzle: Puzzle,
                k: int,
                output: Queue[tuple[int, MoveSequence | None, timedelta, timedelta]],
            ):
                solution, prep_time, solve_time = solve_for_k(puzzle, k, sym_move_depth)
                output.put((k, solution, prep_time, solve_time))

            if len(k_prospects) > 0:
                k = k_prospects.pop(0)
                process = Process(target=solve_for_k_wrapper, args=(puzzle, k, output))
                processes.append((process, k))
                process.start()

        for _ in range(max_processes):
            spawn_new_process()

        while len(processes) > 0:
            k, stats, prep_time, solve_time = output.get()
            prep_times[k] = prep_time
            solve_times[k] = solve_time

            if stats is None:
                print_stamped(
                    f"k = {k}: UNSAT found in {solve_time} with {prep_time} prep..."
                )

                # Kill the process that returned this result and replace it.
                for i in reversed(range(len(processes))):
                    if processes[i][1] == k:
                        process, _ = processes.pop(i)
                        process.kill()
                        spawn_new_process()
            else:
                print_stamped(
                    f"k = {k}: SAT found in {solve_time} with {prep_time} prep..."
                )

                # Update the optimal solution if it is better than the current.
                if optimal_solution is None or k < len(optimal_solution):
                    optimal_solution = stats

                # Filter out larger prospects, since we now know they are also SAT.
                k_prospects = [kp for kp in k_prospects if kp < k]

                # Kill the process that returned this result and any processes solving
                # larger prospects, and replace them.
                for i in reversed(range(len(processes))):
                    if processes[i][1] >= k:
                        process, _ = processes.pop(i)
                        process.kill()
                        spawn_new_process()

    if optimal_solution is None:
        k = "n/a"
        total_solve_time = sum(solve_times.values(), timedelta())
        total_prep_time = sum(prep_times.values(), timedelta())
        print_stamped(
            f"foud no k ≤ {k_upperbound} to be possible in {total_solve_time} with {total_prep_time} prep"  # noqa: E501
        )
    else:
        k = len(optimal_solution)
        total_solve_time = sum(
            [v for kp, v in solve_times.items() if kp <= k], timedelta()
        )
        total_prep_time = sum(
            [v for kp, v in prep_times.items() if kp <= k], timedelta()
        )
        print_stamped(
            f"minimum k = {k} found in {total_solve_time} with {total_prep_time} prep"  # noqa: E501
        )

    stats = {
        "k": k,
        "moves": "impossible"
        if optimal_solution is None
        else [move_name(ma, mi, md) for ma, mi, md in optimal_solution],
        "total_solve_time": str(total_solve_time),
        "total_prep_time": str(total_prep_time),
        "prep_times": {k: str(t) for k, t in sorted(prep_times.items())},
        "solve_times": {k: str(t) for k, t in sorted(solve_times.items())},
        "max_processes": max_processes,
        "k_upperbound": k_upperbound,
    }

    if not disable_stats_file:
        with open(f"{path}.stats", "w") as file:
            file.write(json.dumps(stats, indent=4))

    return optimal_solution, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--sym-moves-dep", default=0, type=int)
    parser.add_argument("--max-processes", default=cpu_count() - 1, type=int)
    parser.add_argument("--disable-stats-file", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    solve(
        args.path,
        args.sym_moves_dep,
        args.max_processes,
        args.disable_stats_file,
    )
