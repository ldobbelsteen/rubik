"""Solve a puzzle using Z3."""

import argparse
import operator
import os
from datetime import datetime
from functools import reduce

import z3

import move_mappers.default
import move_mappers.stacked
import move_symmetries
from puzzle import (
    PUZZLE_DIR,
    CornerState,
    EdgeState,
    MoveSeq,
    Puzzle,
    finished_corner_states,
    finished_edge_states,
    move_names,
)
from solve_config import SolveConfig, gods_number
from stats import SolveStats
from tools import natural_sorted, print_stamped


def validate_solution(puzzle: Puzzle, solution: MoveSeq):
    """Check whether a solution is actually valid for a puzzle and whether the
    move sequence is allowed by the filters.
    """
    canon = move_names(solution)

    if not move_symmetries.allowed_by_filters(puzzle.n, solution):
        raise Exception(f"solution should have been filtered: {canon}")

    for move in solution:
        puzzle = puzzle.execute_move(move)
    if not puzzle.is_finished():
        raise Exception(f"solution is not actual solution: {canon}")


def z3_int(
    solver: z3.Solver | z3.Optimize, name: str, low: int, high: int
) -> z3.ArithRef:
    """Create Z3 integer and add its value range to the solver. The range is
    inclusive on both sides.
    """
    var = z3.Int(name)
    solver.add(z3.And(var >= low, var <= high))
    return var


def solve_for_k(
    puzzle: Puzzle,
    k: int,
    config: SolveConfig,
    banned: list[MoveSeq] = [],
):
    """Compute the optimal solution for a puzzle with a maximum number of moves k.
    Returns list of moves or nothing if impossible. In both cases, also returns the time
    it took to prepare the SAT model and the time it took to solve it.
    """
    fin_corner_states = finished_corner_states(puzzle.n)
    fin_edge_states = finished_edge_states(puzzle.n)
    prep_start = datetime.now()
    n = puzzle.n

    # Configure Z3.
    z3.set_param("parallel.enable", True)
    z3.set_param("parallel.threads.max", config.max_solver_threads)

    # Boil down to SAT and use SAT solver.
    tactics = [
        "normalize-bounds",
        "purify-arith",
        # "elim-term-ite",  # fails
        # "blast-term-ite",  # big negative impact
        "solve-eqs",
        "simplify",
        "dom-simplify",
        # "lia2pb",
        "lia2card",
        # "simplify",
        # "eq2bv",
        # "pb2bv",
        "card2bv",
        "propagate-bv-bounds",
        "bit-blast",
        # "aig",
        "sat-preprocess",
    ]
    solver = z3.Then(*tactics, "psat").solver()

    # # Use quantifier-free finite domain solver.
    # # TODO: it seems SAT results returned are not always valid. It does not seem
    # # to do with the specific tactics used, but rather with the QF_FD solver (since
    # # running it on just z3.SolverFor("QF_FD") also fails for some puzzles).
    # tactics = [
    #     "normalize-bounds",  # medium positive impact
    #     "purify-arith",  # tiny positive impact (could be variance)
    #     # "elim-term-ite",  # fails
    #     # "blast-term-ite",  # big negative impact
    #     "solve-eqs",  # medium positive impact
    #     "simplify",  # small positive impact
    #     "dom-simplify",  # small positive impact
    #     # "lia2pb",  # medium negative impact
    #     "lia2card",  # small positive impact
    #     # "eq2bv",  # big negative impact
    #     # "pb2bv",  # medium negative impact
    #     # "card2bv",  # big negative impact
    #     # "propagate-bv-bounds",  # no impact
    #     # "bit-blast",  # tiny negative impact (could be variance)
    #     # "aig",  # big negative impact
    #     # "sat-preprocess",  # big negative impact
    # ]
    # solver = z3.Then(*tactics, "pqffd").solver()

    # Nested lists representing the cube at each state.
    corners = [
        [
            (
                z3.Bool(f"corner({x},{y},{z}) s({s}) x"),
                z3.Bool(f"corner({x},{y},{z}) s({s}) y"),
                z3.Bool(f"corner({x},{y},{z}) s({s}) z"),
                z3_int(solver, f"corner({x},{y},{z}) s({s}) r", 0, 2),
                z3.Bool(f"corner({x},{y},{z}) s({s}) c"),
            )
            for x, y, z, _, _ in fin_corner_states
        ]
        for s in range(k + 1)
    ]
    edges = [
        [
            (
                z3_int(solver, f"edge({x},{y},{z}) s({s}) a", 0, 2),
                z3.Bool(f"edge({x},{y},{z}) s({s}) x_hi"),
                z3.Bool(f"edge({x},{y},{z}) s({s}) y_hi"),
                z3.Bool(f"edge({x},{y},{z}) s({s}) r"),
            )
            for x, y, z, _ in fin_edge_states
        ]
        for s in range(k + 1)
    ]

    # Variables which together indicate the move at each state.
    axs = [z3_int(solver, f"s({s}) ax", 0, 2) for s in range(k)]
    his = [z3.Bool(f"s({s}) hi") for s in range(k)]
    drs = [z3_int(solver, f"s({s}) dr", 0, 2) for s in range(k)]

    def fix_state(
        s: int,
        corner_states: tuple[CornerState, ...],
        edge_states: tuple[EdgeState, ...],
    ):
        """Return conditions of a state being equal to a state."""
        conditions: list[z3.BoolRef | bool] = []
        for c1, c2 in zip(corners[s], corner_states):
            for v1, v2 in zip(c1, c2):
                conditions.append(v1 == v2)
        for e1, e2 in zip(edges[s], edge_states):
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
    solver.add(z3.And(fix_state(0, puzzle.corner_states, puzzle.edge_states)))

    # Fix the last state to the finished state.
    solver.add(z3.And(fix_state(-1, fin_corner_states, fin_edge_states)))

    # Restrict cubie states according to moves.
    for s in range(k):
        if s % config.move_size != 0:
            continue
        move_size = min(k - s, config.move_size)

        # Add restrictions for the corner cubies.
        for i, (x_hi, y_hi, z_hi, r, cw) in enumerate(corners[s]):
            if move_size == 1:
                ax, hi, dr = axs[s], his[s], drs[s]
                (
                    next_x_hi,
                    next_y_hi,
                    next_z_hi,
                    next_r,
                    next_cw,
                ) = corners[s + 1][i]
                solver.add(
                    move_mappers.default.z3_corner_x_hi(
                        x_hi, y_hi, z_hi, ax, hi, dr, next_x_hi
                    )
                )
                solver.add(
                    move_mappers.default.z3_corner_y_hi(
                        x_hi, y_hi, z_hi, ax, hi, dr, next_y_hi
                    )
                )
                solver.add(
                    move_mappers.default.z3_corner_z_hi(
                        x_hi, y_hi, z_hi, ax, hi, dr, next_z_hi
                    )
                )
                solver.add(
                    move_mappers.default.z3_corner_r(
                        x_hi, z_hi, r, cw, ax, hi, dr, next_r
                    )
                )
                solver.add(
                    move_mappers.default.z3_corner_cw(
                        x_hi, y_hi, z_hi, cw, ax, hi, dr, next_cw
                    )
                )
            else:
                (
                    next_x_hi,
                    next_y_hi,
                    next_z_hi,
                    next_r,
                    next_cw,
                ) = corners[s + move_size][i]
                axl, hil, drl = (
                    axs[s : s + move_size],
                    his[s : s + move_size],
                    drs[s : s + move_size],
                )
                solver.add(
                    next_x_hi
                    == move_mappers.stacked.z3_corner_x_hi(
                        x_hi, y_hi, z_hi, axl, hil, drl
                    )
                )
                solver.add(
                    next_y_hi
                    == move_mappers.stacked.z3_corner_y_hi(
                        x_hi, y_hi, z_hi, axl, hil, drl
                    )
                )
                solver.add(
                    next_z_hi
                    == move_mappers.stacked.z3_corner_z_hi(
                        x_hi, y_hi, z_hi, axl, hil, drl
                    )
                )
                solver.add(
                    next_r
                    == move_mappers.stacked.z3_corner_r(
                        x_hi, y_hi, z_hi, r, cw, axl, hil, drl
                    )
                )
                solver.add(
                    next_cw
                    == move_mappers.stacked.z3_corner_cw(
                        x_hi, y_hi, z_hi, cw, axl, hil, drl
                    )
                )

        # Add restrictions for the edge cubies.
        for i, (a, x_hi, y_hi, r) in enumerate(edges[s]):
            if move_size == 1:
                ax, hi, dr = axs[s], his[s], drs[s]
                (
                    next_a,
                    next_x_hi,
                    next_y_hi,
                    next_r,
                ) = edges[s + 1][i]
                solver.add(
                    move_mappers.default.z3_edge_a(a, x_hi, y_hi, ax, hi, dr, next_a)
                )
                solver.add(
                    move_mappers.default.z3_edge_x_hi(
                        a, x_hi, y_hi, ax, hi, dr, next_x_hi
                    )
                )
                solver.add(
                    move_mappers.default.z3_edge_y_hi(
                        a, x_hi, y_hi, ax, hi, dr, next_y_hi
                    )
                )
                solver.add(move_mappers.default.z3_edge_r(a, next_a, r, next_r))
            else:
                (
                    next_a,
                    next_x_hi,
                    next_y_hi,
                    next_r,
                ) = edges[s + move_size][i]
                axl, hil, drl = (
                    axs[s : s + move_size],
                    his[s : s + move_size],
                    drs[s : s + move_size],
                )
                solver.add(
                    next_a
                    == move_mappers.stacked.z3_edge_a(a, x_hi, y_hi, axl, hil, drl)
                )
                solver.add(
                    next_x_hi
                    == move_mappers.stacked.z3_edge_x_hi(a, x_hi, y_hi, axl, hil, drl)
                )
                solver.add(
                    next_y_hi
                    == move_mappers.stacked.z3_edge_y_hi(a, x_hi, y_hi, axl, hil, drl)
                )
                solver.add(
                    next_r
                    == move_mappers.stacked.z3_edge_r(a, x_hi, y_hi, r, axl, hil, drl)
                )

    # Add symmetric move sequence filters for n = 2.
    if n == 2:
        # Move filter #1 and #2
        for s in range(k - 1):
            solver.add(
                z3.Implies(
                    axs[s] == axs[s + 1],
                    z3.Not(
                        z3.Or(
                            z3.Not(his[s + 1]),
                            his[s],
                        )
                    ),
                )
            )

    # Add symmetric move sequence filters for n = 3.
    if n == 3:
        # Move filter #1 and #2
        for s in range(k - 1):
            solver.add(
                z3.Implies(
                    axs[s] == axs[s + 1],
                    z3.Not(
                        z3.Or(
                            z3.Not(his[s]),
                            his[s + 1],
                        )
                    ),
                )
            )

        # Move filter #3 and #4
        for s in range(k - 3):
            solver.add(
                z3.Implies(
                    z3.And(
                        drs[s] == 2,
                        drs[s + 1] == 2,
                        drs[s + 2] == 2,
                        drs[s + 3] == 2,
                    ),
                    z3.Not(
                        z3.Or(
                            z3.And(
                                axs[s] == axs[s + 3],
                                axs[s + 1] == axs[s + 2],
                                z3.Not(his[s]),
                            ),
                            z3.And(
                                axs[s] == axs[s + 1],
                                axs[s + 1] > axs[s + 2],
                                axs[s + 2] == axs[s + 3],
                            ),
                        )
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

    # States cannot be repeated.
    for s1 in range(k + 1):
        for s2 in range(s1 + 1, k + 1):
            solver.add(z3.Not(z3.And(identical_states(s1, s2))))

    # Theorem 11.1a: sum x_i = 0 mod 3
    if config.apply_theorem_11a:
        for s in range(k + 1):
            corner_sum = reduce(operator.add, [r for _, _, _, r, _ in corners[s]])
            solver.add(corner_sum % 3 == 0)

    # Theorem 11.1b: sum y_i = 0 mod 2
    if config.apply_theorem_11b:
        for s in range(k + 1):
            if len(edges[s]) > 0:
                edge_sum = reduce(operator.add, [r for _, _, _, r in edges[s]])
                solver.add(edge_sum % 2 == 0)

    # Check model and return moves if sat.
    prep_time = datetime.now() - prep_start
    solve_start = datetime.now()
    res = solver.check()
    solve_time = datetime.now() - solve_start

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
        return moves, prep_time, solve_time
    elif res == z3.unsat:
        return None, prep_time, solve_time
    else:
        raise Exception(f"unexpected solver result: {res}")


def solve(puzzle: Puzzle, config: SolveConfig, print_info: bool) -> SolveStats:
    """Compute the optimal solution for a puzzle within an upperbound for the number
    of moves. If no upperbound is given, God's number is used.
    """
    stats = SolveStats(puzzle, config)
    k_upperbound = gods_number(puzzle.n)

    for k in range(k_upperbound + 1):
        solution, prep_time, solve_time = solve_for_k(puzzle, k, config)
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

    if stats.solution is None:
        if print_info:
            print_stamped(
                f"foud no k ≤ {k_upperbound} to be possible in {stats.total_solve_time()} with {stats.total_prep_time()} prep"  # noqa: E501
            )
    else:
        validate_solution(puzzle, stats.solution)
        if print_info:
            print_stamped(
                f"minimum k = {stats.k()} found in {stats.total_solve_time()} with {stats.total_prep_time()} prep"  # noqa: E501
            )

    return stats


def solve_one(name: str, config: SolveConfig, print_info: bool):
    """Helper function for solving a single puzzle."""
    puzzle = Puzzle.from_file(name)
    stats = solve(puzzle, config, print_info)
    stats.to_file()


def solve_all(config: SolveConfig, print_info: bool):
    """Helper function for solving all puzzles."""
    names = [filename for filename in os.listdir(PUZZLE_DIR)]
    names = natural_sorted(names)

    puzzles = [Puzzle.from_file(name) for name in names]
    for puzzle in puzzles:
        print_stamped(f"solving {puzzle.name}...")
        stats = solve(puzzle, config, print_info)
        stats.to_file()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, nargs="?")
    args = parser.parse_args()

    if args.name is None:
        solve_all(SolveConfig.default(), True)
    else:
        solve_one(args.name, SolveConfig.default(), True)
