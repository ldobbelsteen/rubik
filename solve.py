import argparse
import json
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue

import z3

import mappers
from misc import gods_number, print_stamped
from puzzle import Puzzle, move_name
from sym_move_seqs import MoveSequence, load


def z3_int(solver: z3.Optimize, name: str, low: int, high: int):
    """Create Z3 integer and add its value range to the solver. The range is
    inclusive on low and exclusive on high."""
    var = z3.Int(name)
    solver.add(var >= low)
    solver.add(var < high)
    return var


def solve_for_k(
    puzzle: Puzzle,
    k: int,
    move_skipping: bool,
    sym_move_depth: int,
    disallowed: list[MoveSequence] = [],
):
    """Compute the optimal solution for a puzzle with a maximum number of moves k.
    Returns list of moves or nothing if impossible. In both cases, also returns the time
    it took to prepare the SAT model and the time it took to solve it."""
    prep_start = datetime.now()
    solver = z3.Optimize()

    n = puzzle.n
    finished = Puzzle.finished(n)
    cubicles = finished.cubicles

    # Nested lists representing the n × n × n cube for each state.
    corners = [
        [
            (
                z3.Bool(f"corner({x},{y},{z}) s({s}) x"),
                z3.Bool(f"corner({x},{y},{z}) s({s}) y"),
                z3.Bool(f"corner({x},{y},{z}) s({s}) z"),
                z3_int(solver, f"corner({x},{y},{z}) s({s}) r", 0, 3),
                z3.Bool(f"corner({x},{y},{z}) s({s}) c"),
            )
            for x, y, z, _, _ in cubicles.corners
        ]
        for s in range(k + 1)
    ]
    centers = [
        [
            (
                z3_int(solver, f"center({a},{h}) s({s}) a", 0, 3),
                z3.Bool(f"center({a},{h}) s({s}) s"),
            )
            for a, h in cubicles.centers
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
            for x, y, z, _ in cubicles.edges
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
        for i, (a, h) in enumerate(centers[s]):
            pa, ph = puzzle.centers[i]
            conds.extend([a == pa, h == ph])
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
        for i, (a1, h1) in enumerate(centers[s1]):
            a2, h2 = centers[s2][i]
            conds.extend([a1 == a2, h1 == h2])
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
        move_skipping_single = k % 2 == 1 and s == (k - 1)

        for i, (x, y, z, r, c) in enumerate(corners[s]):
            if not move_skipping or move_skipping_single:
                next_x, next_y, next_z, next_r, next_c = corners[s + 1][i]
                solver.add(mappers.z3_corner_x(n, x, y, z, ma, mi, md, next_x))
                solver.add(mappers.z3_corner_y(n, x, y, z, ma, mi, md, next_y))
                solver.add(mappers.z3_corner_z(n, x, y, z, ma, mi, md, next_z))
                solver.add(mappers.z3_corner_r(n, x, z, r, c, ma, mi, md, next_r))
                solver.add(mappers.z3_corner_c(n, x, y, z, c, ma, mi, md, next_c))
            elif s % 2 == 0:
                next_x, next_y, next_z, next_r, next_c = corners[s + 2][i]
                ma2, mi2, md2 = mas[s + 1], mis[s + 1], mds[s + 1]
                # TODO

        for i, (a, h) in enumerate(centers[s]):
            if not move_skipping or move_skipping_single:
                next_a, next_h = centers[s + 1][i]
                solver.add(mappers.z3_center_a(a, ma, mi, md, next_a))
                solver.add(mappers.z3_center_h(a, h, ma, mi, md, next_h))
            elif s % 2 == 0:
                next_a, next_h = centers[s + 2][i]
                ma2, mi2, md2 = mas[s + 1], mis[s + 1], mds[s + 1]
                # TODO

        for i, (x, y, z, r) in enumerate(edges[s]):
            if not move_skipping or move_skipping_single:
                next_x, next_y, next_z, next_r = edges[s + 1][i]
                solver.add(mappers.z3_edge_x(n, x, y, z, ma, mi, md, next_x))
                solver.add(mappers.z3_edge_y(n, x, y, z, ma, mi, md, next_y))
                solver.add(mappers.z3_edge_z(n, x, y, z, ma, mi, md, next_z))
                solver.add(mappers.z3_edge_r(x, y, z, r, ma, mi, md, next_r))
            elif s % 2 == 0:
                next_x, next_y, next_z, next_r = edges[s + 2][i]
                ma2, mi2, md2 = mas[s + 1], mis[s + 1], mds[s + 1]
                # TODO

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
    move_skipping: bool,
    sym_move_depth: int,
    max_processes: int,
    disable_stats_file: bool,
) -> tuple[MoveSequence | None, dict]:
    """Compute the optimal solution for a puzzle in parallel for all possible values
    of k within the upperbound. Returns the solution and a dict containing statistics
    of the solving process."""
    print_stamped(f"solving '{path}'...")

    puzzle = Puzzle.from_file(path)
    k_upperbound = gods_number(puzzle.n)

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
                solution, prep_time, solve_time = solve_for_k(
                    puzzle, k, move_skipping, sym_move_depth
                )
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
    parser.add_argument("--move-skipping", action=argparse.BooleanOptionalAction)
    parser.add_argument("--sym-moves-dep", default=0, type=int)
    parser.add_argument("--max-processes", default=cpu_count() - 1, type=int)
    parser.add_argument("--disable-stats-file", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    solve(
        args.path,
        args.move_skipping,
        args.sym_moves_dep,
        args.max_processes,
        args.disable_stats_file,
    )
