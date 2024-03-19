import argparse
from datetime import datetime
from multiprocessing import cpu_count

import z3

import move_mappers
from move_symmetries import load
from puzzle import MoveSeq, Puzzle
from stats import Stats
from tools import gods_number, print_stamped


def z3_int(
    solver: z3.Solver | z3.Optimize, name: str, low: int, high: int
) -> z3.ArithRef:
    """Create Z3 integer and add its value range to the solver. The range is
    inclusive on low and exclusive on high."""
    var = z3.Int(name)
    solver.add(var >= low)
    solver.add(var < high)
    return var


def solve_for_k(
    puzzle: Puzzle,
    k: int,
    max_threads: int,
    move_stacking: bool,
    sym_move_depth: int,
    banned: list[MoveSeq] = [],
):
    """Compute the optimal solution for a puzzle with a maximum number of moves k.
    Returns list of moves or nothing if impossible. In both cases, also returns the time
    it took to prepare the SAT model and the time it took to solve it."""
    prep_start = datetime.now()
    n = puzzle.n

    # Configure Z3.
    z3.set_param("parallel.enable", True)
    z3.set_param("parallel.threads.max", max_threads)
    solver = z3.Then(
        z3.Tactic("simplify"),
        z3.Tactic("solve-eqs"),
        z3.Tactic("aig"),
        # z3.Tactic("dom-simplify"),
        # z3.Tactic("purify-arith"),
        # z3.Tactic("lia2pb"),
        # z3.Tactic("pb2bv"),
        # z3.Tactic("lia2card"),
        # z3.Tactic("card2bv"),
        # z3.Tactic("bit-blast"),
        # z3.Tactic("propagate-bv-bounds"),
        # z3.Tactic("blast-term-ite"),
        # z3.Tactic("elim-term-ite"),
        z3.Tactic("pqffd"),
        # z3.Tactic("sat-preprocess"),
        # z3.Tactic("sat"),
    ).solver()

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
                # TODO: implement once stacked mappers are ready
                # next_x_hi, next_y_hi, next_z_hi, next_r, next_cw = corners[s + 2][i]
                # ax2, hi2, dr2 = axs[s + 1], his[s + 1], drs[s + 1]
                pass

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
                # TODO: implement once stacked mappers are ready
                # next_a, next_x_hi, next_y_hi, next_r = edges[s + 2][i]
                # ax2, hi2, dr2 = axs[s + 1], his[s + 1], drs[s + 1]
                pass

    # Symmetric move filter #1
    # Subsequent moves in the same axis have fixed side order: first low, then high.
    for s in range(k - 1):
        solver.add(z3.Implies(axs[s] == axs[s + 1], z3.And(z3.Not(his[s]), his[s + 1])))

    # Symmetric move filter #2
    # If we make a move at an axis and side, we cannot make a move at the same axis and
    # side for two moves, unless a different axis has been turned in the meantime.
    for s in range(k - 1):
        solver.add(
            z3.And(
                [
                    z3.Implies(
                        z3.And(axs[f] == axs[s], his[f] == his[s]),
                        z3.Or([axs[s] != axs[b] for b in range(s + 1, f)]),
                    )
                    for f in range(s + 1, min(s + 3, k))
                ]
            )
        )

    # Symmetric move filters #3 and #4
    # For four consecutive half moves, there are the following two requirements:
    # 1. If the moves have axis pattern XYYX, then the first and last moves have a fixed
    #    side order: first low, then high OR first high, then high.
    # 2. If the moves have axis pattern XXYY, then X < Y, since they are commutative.
    if n == 3:
        for s in range(k - 3):
            solver.add(
                z3.Implies(
                    z3.And(
                        drs[s] == 2, drs[s + 1] == 2, drs[s + 2] == 2, drs[s + 3] == 2
                    ),
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
    for b in banned:
        solver.add(ban_move_sequence(b))

    # Ban computed symmetric move sequences up to the specified depth.
    for _, syms in load(n, sym_move_depth).items():
        for sym in syms:
            solver.add(ban_move_sequence(sym))

    # States cannot be repeated.
    for s1 in range(k + 1):
        for s2 in range(s1 + 1, k + 1):
            solver.add(z3.Not(z3.And(identical_states(s1, s2))))

    # # Theorem 11.1a: sum x_i = 0 mod 3
    # for s in range(k + 1):
    #     corner_sum = reduce(operator.add, [r for _, _, _, r, _ in corners[s]])
    #     solver.add(corner_sum % 3 == 0)

    # # Theorem 11.1b: sum y_i = 0 mod 2
    # for s in range(k + 1):
    #     if len(edges[s]) > 0:
    #         edge_sum = reduce(operator.add, [r for _, _, _, r in edges[s]])
    #         solver.add(edge_sum % 2 == 0)

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
    puzzle: Puzzle,
    k_upperbound: int,
    max_threads: int,
    move_stacking: bool,
    sym_move_depth: int,
    print_info: bool,
) -> Stats:
    """Compute the optimal solution for a puzzle within an upperbound."""
    stats = Stats(max_threads, k_upperbound)

    for k in range(k_upperbound + 1):
        solution, prep_time, solve_time = solve_for_k(
            puzzle,
            k,
            max_threads,
            move_stacking,
            sym_move_depth,
        )

        stats.register_solution(k, solution, prep_time, solve_time)

        if solution is None:
            if print_info:
                print_stamped(
                    f"k = {k}: UNSAT found in {solve_time} with {prep_time} prep"
                )
        else:
            if print_info:
                print_stamped(
                    f"k = {k}: SAT found in {solve_time} with {prep_time} prep"
                )
            break

    if print_info:
        if stats.solution is None:
            print_stamped(
                f"foud no k ≤ {stats.k_upperbound} to be possible in {stats.total_solve_time()} with {stats.total_prep_time()} prep"  # noqa: E501
            )
        else:
            print_stamped(
                f"minimum k = {stats.k()} found in {stats.total_solve_time()} with {stats.total_prep_time()} prep"  # noqa: E501
            )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--move-stacking", action=argparse.BooleanOptionalAction)
    parser.add_argument("--sym-moves-dep", default=0, type=int)
    parser.add_argument("--max-threads", default=cpu_count() - 1, type=int)
    parser.add_argument("--disable-stats-file", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    puzzle = Puzzle.from_file(args.path)
    stats = solve(
        puzzle,
        gods_number(puzzle.n),
        args.max_threads,
        args.move_stacking,
        args.sym_moves_dep,
        True,
    )
    if not args.disable_stats_file:
        stats.write_to_file(args.path)
