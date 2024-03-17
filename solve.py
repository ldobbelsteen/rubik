import argparse
import json
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue

import z3

import move_mappers
from move_symmetries import load
from puzzle import MoveSeq, Puzzle, move_name
from tools import gods_number, print_stamped


def z3_int(solver: z3.Optimize, name: str, low: int, high: int) -> z3.ArithRef:
    """Create Z3 integer and add its value range to the solver. The range is
    inclusive on low and exclusive on high."""
    var = z3.Int(name)
    solver.add(var >= low)
    solver.add(var < high)
    return var


def solve_for_k(
    puzzle: Puzzle,
    k: int,
    move_stacking: bool,
    sym_move_depth: int,
    banned: list[MoveSeq] = [],
):
    """Compute the optimal solution for a puzzle with a maximum number of moves k.
    Returns list of moves or nothing if impossible. In both cases, also returns the time
    it took to prepare the SAT model and the time it took to solve it."""
    prep_start = datetime.now()
    solver = z3.Optimize()
    n = puzzle.n

    # Nested lists representing the cube at each state.
    corners = [
        [
            (
                z3.Bool(f"corner({x},{y},{z}) s({s}) x"),
                z3.Bool(f"corner({x},{y},{z}) s({s}) y"),
                z3.Bool(f"corner({x},{y},{z}) s({s}) z"),
                z3_int(solver, f"corner({x},{y},{z}) s({s}) r", 0, 3),
                z3.Bool(f"corner({x},{y},{z}) s({s}) c"),
            )
            for x, y, z, _, _ in puzzle.finished_state.corners
        ]
        for s in range(k + 1)
    ]
    edges = [
        [
            (
                z3_int(solver, f"edge({x},{y},{z}) s({s}) a", 0, 3),
                z3.Bool(f"edge({x},{y},{z}) s({s}) x_hi"),
                z3.Bool(f"edge({x},{y},{z}) s({s}) y_hi"),
                z3.Bool(f"edge({x},{y},{z}) s({s}) r"),
            )
            for x, y, z, _ in puzzle.finished_state.edges
        ]
        for s in range(k + 1)
    ]

    # Variables which together indicate the move at each state.
    axs = [z3_int(solver, f"s({s}) ax", 0, 3) for s in range(k)]
    his = [z3.Bool(f"s({s}) hi") for s in range(k)]
    drs = [z3_int(solver, f"s({s}) dr", 0, 3) for s in range(k)]

    def fix_state(s: int, puzzle: Puzzle):
        """Return conditions of a state being equal to a puzzle object."""
        conditions: list[z3.BoolRef | bool] = []
        for c1, c2 in zip(corners[s], puzzle.corner_states):
            for v1, v2 in zip(c1, c2):
                conditions.append(v1 == v2)
        for e1, e2 in zip(edges[s], puzzle.edge_states):
            for v1, v2 in zip(e1, e2):
                conditions.append(v1 == v2)
        return conditions

    def identical_states(s1: int, s2: int):
        """Return conditions of two states being equal."""
        conditions: list[z3.BoolRef | bool] = []
        for c1, c2 in zip(corners[s1], corners[s2]):
            for v1, v2 in zip(c1, c2):
                conditions.append(v1 == v2)
        for e1, e2 in zip(edges[s1], edges[s2]):
            for v1, v2 in zip(e1, e2):
                conditions.append(v1 == v2)
        return conditions

    # Fix the first state to the puzzle state.
    solver.add(z3.And(fix_state(0, puzzle)))

    # Fix the last state to the finished state.
    finished = Puzzle.finished(n, puzzle.center_colors)
    solver.add(z3.And(fix_state(-1, finished)))

    # Restrict cubie states according to moves.
    for s in range(k):
        ax, hi, dr = axs[s], his[s], drs[s]
        move_stacking_single = k % 2 == 1 and s == (k - 1)

        # Add restrictions for the corner cubies.
        for i, (x_hi, y_hi, z_hi, r, cw) in enumerate(corners[s]):
            if not move_stacking or move_stacking_single:
                next_x_hi, next_y_hi, next_z_hi, next_r, next_cw = corners[s + 1][i]
                solver.add(
                    move_mappers.z3_corner_x_hi(x_hi, y_hi, z_hi, ax, hi, dr, next_x_hi)
                )
                solver.add(
                    move_mappers.z3_corner_y_hi(x_hi, y_hi, z_hi, ax, hi, dr, next_y_hi)
                )
                solver.add(
                    move_mappers.z3_corner_z_hi(x_hi, y_hi, z_hi, ax, hi, dr, next_z_hi)
                )
                solver.add(
                    move_mappers.z3_corner_r(x_hi, z_hi, r, cw, ax, hi, dr, next_r)
                )
                solver.add(
                    move_mappers.z3_corner_cw(x_hi, y_hi, z_hi, cw, ax, hi, dr, next_cw)
                )
            elif s % 2 == 0:
                next_x_hi, next_y_hi, next_z_hi, next_r, next_cw = corners[s + 2][i]
                ax2, hi2, dr2 = axs[s + 1], his[s + 1], drs[s + 1]
                # TODO: implement once stacked mappers are ready

        # Add restrictions for the edge cubies.
        for i, (a, x_hi, y_hi, r) in enumerate(edges[s]):
            if not move_stacking or move_stacking_single:
                next_a, next_x_hi, next_y_hi, next_r = edges[s + 1][i]
                solver.add(move_mappers.z3_edge_a(a, x_hi, y_hi, ax, hi, dr, next_a))
                solver.add(
                    move_mappers.z3_edge_x_hi(a, x_hi, y_hi, ax, hi, dr, next_x_hi)
                )
                solver.add(
                    move_mappers.z3_edge_y_hi(a, x_hi, y_hi, ax, hi, dr, next_y_hi)
                )
                solver.add(move_mappers.z3_edge_r(a, next_a, r, next_r))
            elif s % 2 == 0:
                next_a, next_x_hi, next_y_hi, next_r = edges[s + 2][i]
                ax2, hi2, dr2 = axs[s + 1], his[s + 1], drs[s + 1]
                # TODO: implement once stacked mappers are ready

    # Subsequent moves in the same axis have fixed side order: first low, then high.
    for s in range(k - 1):
        solver.add(z3.Or(axs[s] != axs[s + 1], z3.And(z3.Not(his[s]), his[s + 1])))

    # If we make a move at an axis and side, we cannot make a move at the same axis and
    # side for two moves, unless a different axis has been turned in the meantime.
    for s in range(k - 1):
        solver.add(
            z3.And(
                [
                    z3.Or(
                        axs[f] != axs[s],
                        his[f] != his[s],
                        z3.Or([axs[s] != axs[b] for b in range(s + 1, f)]),
                    )
                    for f in range(s + 1, min(s + 3, k))
                ]
            )
        )

    # For four consecutive half moves, there are the following two requirements:
    # 1. If the moves have axis pattern XYYX, then the first and last moves have a fixed
    #    side order: first low, then high OR first high, then high.
    # 2. If the moves have axis pattern XXYY, then X < Y, since they are commutative.
    for s in range(k - 3):
        solver.add(
            z3.Implies(
                z3.And(drs[s] == 2, drs[s + 1] == 2, drs[s + 2] == 2, drs[s + 3] == 2),
                z3.And(
                    z3.Implies(
                        z3.And(
                            axs[s] == axs[s + 3],
                            axs[s + 1] == axs[s + 2],
                            axs[s] != axs[s + 1],
                        ),
                        his[s + 3],
                    ),
                    z3.Implies(
                        z3.And(
                            axs[s] == axs[s + 1],
                            axs[s + 2] == axs[s + 3],
                            axs[s + 1] != axs[s + 2],
                        ),
                        axs[s + 1] < axs[s + 2],
                    ),
                ),
            )
        )

    # States cannot be repeated.
    for s1 in range(k + 1):
        for s2 in range(s1 + 1, k + 1):
            solver.add(z3.Not(z3.And(identical_states(s1, s2))))

    def ban_move_sequence(ms: MoveSeq):
        """Return conditions of a move sequence not being allowed."""
        return z3.And(
            [
                z3.Not(
                    z3.And(
                        [
                            z3.And(
                                axs[start + i] == ma,
                                his[start + i] == mi,
                                drs[start + i] == md,
                            )
                            for i, (ma, mi, md) in enumerate(ms)
                        ]
                    )
                )
                for start in range(k - len(ms) + 1)
            ]
        )

    # Ban the move sequences from the parameters.
    for seq in banned:
        solver.add(ban_move_sequence(seq))

    # Ban computed symmetric move sequences up to the specified depth.
    for seq, syms in load(n, sym_move_depth).items():
        for sym in syms:
            solver.add(ban_move_sequence(sym))

    # Check model and return moves if sat.
    prep_time = datetime.now() - prep_start
    solve_start = datetime.now()
    res = solver.check()
    solve_time = datetime.now() - solve_start
    moves: MoveSeq | None = None

    if res == z3.sat:
        model = solver.model()
        moves = tuple(
            (
                model.get_interp(axs[s]).as_long(),
                z3.is_true(model.get_interp(his[s])),
                model.get_interp(drs[s]).as_long(),
            )
            for s in range(k)
        )
        assert len(moves) == k

    return moves, prep_time, solve_time


def solve(
    path: str,
    move_stacking: bool,
    sym_move_depth: int,
    max_processes: int,
    disable_stats_file: bool,
) -> tuple[MoveSeq | None, dict]:
    """Compute the optimal solution for a puzzle in parallel for all possible values
    of k within the upperbound. Returns the solution and a dict containing statistics
    of the solving process."""
    print_stamped(f"solving '{path}'...")

    puzzle = Puzzle.from_file(path)
    k_upperbound = gods_number(puzzle.n)

    with Manager() as manager:
        k_prospects = list(range(k_upperbound + 1))
        optimal_solution: MoveSeq | None = None

        # Times for each tried k.
        prep_times: dict[int, timedelta] = {}
        solve_times: dict[int, timedelta] = {}

        # List of running processes and their k.
        processes: list[tuple[Process, int]] = []

        # Queue for the processes to output results onto.
        output: Queue[tuple[int, MoveSeq | None, timedelta, timedelta]] = (
            manager.Queue()
        )

        def spawn_new_process():
            def solve_for_k_wrapper(
                puzzle: Puzzle,
                k: int,
                output: Queue[tuple[int, MoveSeq | None, timedelta, timedelta]],
            ):
                solution, prep_time, solve_time = solve_for_k(
                    puzzle, k, move_stacking, sym_move_depth
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
        else [move_name(move) for move in optimal_solution],
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
    parser.add_argument("--move-stacking", action=argparse.BooleanOptionalAction)
    parser.add_argument("--sym-moves-dep", default=0, type=int)
    parser.add_argument("--max-processes", default=cpu_count() - 1, type=int)
    parser.add_argument("--disable-stats-file", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    solve(
        args.path,
        args.move_stacking,
        args.sym_moves_dep,
        args.max_processes,
        args.disable_stats_file,
    )
