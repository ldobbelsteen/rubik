import argparse
import operator
from datetime import datetime
from functools import reduce
from typing import cast

import z3

import move_mappers
from cubie_min_patterns import load_corner_min_patterns, load_edge_min_patterns
from puzzle import (
    CornerState,
    EdgeState,
    Puzzle,
    all_puzzles_names,
)
from solve_config import SolveConfig, gods_number
from state import CornerStateZ3, EdgeStateZ3, Move, MoveSeq, MoveZ3, TernaryZ3
from stats import SolveStats
from tools import print_stamped, str_to_file


def corner_stacked(c: CornerStateZ3, moves: list[MoveZ3]):
    """Return the corner state after applying the moves in the given list."""
    states: list[CornerStateZ3] = [c]
    for move in moves:
        states.append(
            CornerStateZ3(
                c.n,
                move_mappers.corner_x_hi_flat(states[-1], move),
                move_mappers.corner_y_hi_flat(states[-1], move),
                move_mappers.corner_z_hi_flat(states[-1], move),
                TernaryZ3(
                    move_mappers.corner_r_b1_flat(states[-1], move),
                    move_mappers.corner_r_b2_flat(states[-1], move),
                ),
                move_mappers.corner_cw_flat(states[-1], move),
            )
        )
    return states[-1]


def edge_stacked(e: EdgeStateZ3, moves: list[MoveZ3]):
    """Return the edge state after applying the moves in the given list."""
    states: list[EdgeStateZ3] = [e]
    for move in moves:
        next_a = TernaryZ3(
            move_mappers.edge_a_b1_flat(states[-1], move),
            move_mappers.edge_a_b2_flat(states[-1], move),
        )
        states.append(
            EdgeStateZ3(
                e.n,
                next_a,
                move_mappers.edge_x_hi_flat(states[-1], move),
                move_mappers.edge_y_hi_flat(states[-1], move),
                move_mappers.edge_r_flat(states[-1], next_a),
            )
        )
    return states[-1]


def solve_for_k(puzzle: Puzzle, k: int, config: SolveConfig):
    """Compute the optimal solution for a puzzle with a maximum number of moves k.
    Returns list of moves or nothing if impossible. In both cases, also returns the time
    it took to prepare the SAT model and the time it took to solve it.
    """
    goal = z3.Goal()
    prep_start = datetime.now()
    n = puzzle.n

    # Create the corner, edge and move states for each step.
    corners = [
        [
            CornerStateZ3.new(n, s, corner.x_hi, corner.y_hi, corner.z_hi, goal)
            for corner in CornerState.all_finished(n)
        ]
        for s in range(k + 1)
    ]
    edges = [
        [
            EdgeStateZ3.new(n, s, edge.a, edge.x_hi, edge.y_hi, goal)
            for edge in EdgeState.all_finished(n)
        ]
        for s in range(k + 1)
    ]
    moves = [MoveZ3(s, goal) for s in range(k)]

    def fix_state(
        s: int,
        corner_states: tuple[CornerState, ...],
        edge_states: tuple[EdgeState, ...],
    ):
        """Return conditions of a state having fixed corner and edge states."""
        conditions = []
        for c1, c2 in zip(corners[s], corner_states):
            conditions.append(c1 == c2)
        for e1, e2 in zip(edges[s], edge_states):
            conditions.append(e1 == e2)
        return conditions

    def identical_states(s1: int, s2: int):
        """Return conditions of two states being equal."""
        conditions = []
        for c1, c2 in zip(corners[s1], corners[s2]):
            conditions.append(c1 == c2)
        for e1, e2 in zip(edges[s1], edges[s2]):
            conditions.append(e1 == e2)
        return conditions

    # Fix the first state to the puzzle state.
    goal.add(z3.And(fix_state(0, puzzle.corners, puzzle.edges)))

    # Fix the last state to the finished state.
    goal.add(
        z3.And(
            fix_state(
                -1,
                CornerState.all_finished(n),
                EdgeState.all_finished(n),
            )
        )
    )

    # Restrict cubie states according to moves.
    for s in range(k):
        if config.move_size == 0:  # Use basic move mappers.
            m = moves[s]

            for i, c in enumerate(corners[s]):
                next = corners[s + 1][i]
                goal.add(move_mappers.corner_x_hi(c, m, next))
                goal.add(move_mappers.corner_y_hi(c, m, next))
                goal.add(move_mappers.corner_z_hi(c, m, next))
                goal.add(move_mappers.corner_r(c, m, next))
                goal.add(move_mappers.corner_cw(c, m, next))

            # Add restrictions for the edge cubies.
            for i, e in enumerate(edges[s]):
                next = edges[s + 1][i]
                goal.add(move_mappers.edge_a(e, m, next))
                goal.add(move_mappers.edge_x_hi(e, m, next))
                goal.add(move_mappers.edge_y_hi(e, m, next))
                goal.add(move_mappers.edge_r(e, next))

        else:  # Use stacked move mappers.
            if s % config.move_size != 0:
                continue  # Skip if not multiple of move size.
            move_size = min(k - s, config.move_size)
            moveset = moves[s : s + move_size]

            # Add restrictions for the corner cubies.
            for i, c in enumerate(corners[s]):
                next = corners[s + move_size][i]
                goal.add(next == corner_stacked(c, moveset))

            # Add restrictions for the edge cubies.
            for i, e in enumerate(edges[s]):
                next = edges[s + move_size][i]
                goal.add(next == edge_stacked(e, moveset))

    # Add symmetric move sequence filters for n = 2.
    if n == 2:
        if config.enable_n2_move_filters_1_and_2:
            # Move filter #1 and #2
            for s in range(k - 1):
                goal.add(
                    z3.Implies(
                        moves[s].ax == moves[s + 1].ax,
                        z3.Not(
                            z3.Or(
                                z3.Not(moves[s + 1].hi),
                                moves[s].hi,
                            )
                        ),
                    )
                )

    # Add symmetric move sequence filters for n = 3.
    if n == 3:
        if config.enable_n3_move_filters_1_and_2:
            # Move filter #1 and #2
            for s in range(k - 1):
                goal.add(
                    z3.Implies(
                        moves[s].ax == moves[s + 1].ax,
                        z3.Not(
                            z3.Or(
                                z3.Not(moves[s].hi),
                                moves[s + 1].hi,
                            )
                        ),
                    )
                )

        if config.enable_n3_move_filters_3_and_4:
            # Move filter #3 and #4
            for s in range(k - 3):
                goal.add(
                    z3.Implies(
                        z3.And(
                            moves[s].dr == 2,
                            moves[s + 1].dr == 2,
                            moves[s + 2].dr == 2,
                            moves[s + 3].dr == 2,
                        ),
                        z3.Not(
                            z3.Or(
                                z3.And(
                                    moves[s].ax == moves[s + 3].ax,
                                    moves[s + 1].ax == moves[s + 2].ax,
                                    z3.Not(moves[s].hi),
                                ),
                                z3.And(
                                    moves[s].ax == moves[s + 1].ax,
                                    moves[s + 1].ax > moves[s + 2].ax,
                                    moves[s + 2].ax == moves[s + 3].ax,
                                ),
                            )
                        ),
                    )
                )

    if config.ban_repeated_states:
        # States cannot be repeated.
        for s1 in range(k + 1):
            for s2 in range(s1 + 1, k + 1):
                goal.add(z3.Not(z3.And(identical_states(s1, s2))))

    if config.apply_theorem_11a:
        # Theorem 11.1a: sum x_i = 0 mod 3
        for s in range(k + 1):
            if len(corners[s]) > 0:
                corner_sum = reduce(
                    operator.add,
                    [
                        cast(
                            z3.ArithRef,
                            z3.If(c.cw, c.r.arith_value(), -1 * c.r.arith_value()),
                        )
                        for c in corners[s]
                    ],
                )
                goal.add(corner_sum % 3 == 0)

    if config.apply_theorem_11b:
        # Theorem 11.1b: sum y_i = 0 mod 2
        for s in range(k + 1):
            if len(edges[s]) > 0:
                edge_sum = reduce(
                    operator.add,
                    [
                        cast(
                            z3.ArithRef,
                            z3.If(e.r, 1, 0),
                        )
                        for e in edges[s]
                    ],
                )
                goal.add(edge_sum % 2 == 0)

    if config.enable_corner_min_patterns:
        for i, patterns in enumerate(load_corner_min_patterns(n)):
            for corner, depth in patterns.items():
                for s in range(max(0, k + 1 - depth), k + 1):
                    goal.add(corners[s][i] != corner)

    if config.enable_edge_min_patterns:
        for i, patterns in enumerate(load_edge_min_patterns(n)):
            for edge, depth in patterns.items():
                for s in range(max(0, k + 1 - depth), k + 1):
                    goal.add(edges[s][i] != edge)

    if config.max_solver_threads == 0:
        # Parallelism is disabled: use a single-threaded tactic.
        tactic = cast(z3.Tactic, z3.Then(*config.tactics.get(), "sat"))
    else:
        # Configure Z3 to use parallelism.
        z3.set_param("parallel.enable", True)
        z3.set_param("parallel.threads.max", config.max_solver_threads)
        tactic = cast(z3.Tactic, z3.Then(*config.tactics.get(), "psat"))
    solver = tactic.solver()
    solver.add(goal)

    # Check model and return moves if sat.
    prep_time = datetime.now() - prep_start
    solve_start = datetime.now()
    res = solver.check()
    solve_time = datetime.now() - solve_start

    if res == z3.sat:
        model = solver.model()
        solution = MoveSeq(
            tuple(
                Move(
                    model.eval(moves[s].ax.arith_value()).as_long(),  # type: ignore
                    z3.is_true(model.get_interp(moves[s].hi)),
                    model.eval(moves[s].dr.arith_value()).as_long(),  # type: ignore
                )
                for s in range(k)
            )
        )
        return solution, prep_time, solve_time
    elif res == z3.unsat:
        return None, prep_time, solve_time
    else:
        raise Exception(f"unexpected solver result: {res}")


def solve(
    puzzle: Puzzle,
    config: SolveConfig,
    print_info: bool,
    stats_to_file: bool,
) -> SolveStats:
    """Compute the optimal solution for a puzzle within an upperbound for the number
    of moves. Returns the statistics of the solve operation.
    """
    stats = SolveStats(puzzle, config)
    k_upperbound = gods_number(puzzle.n)

    for k in range(k_upperbound + 1):
        solution, prep_time, solve_time = solve_for_k(puzzle, k, config)
        stats.register_solution(k, solution, prep_time, solve_time, stats_to_file)

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
        assert puzzle.is_solution(stats.solution)
        if print_info:
            print_stamped(
                f"minimum k = {stats.k()} found in {stats.total_solve_time()} with {stats.total_prep_time()} prep"  # noqa: E501
            )

    if stats_to_file:
        stats.to_file()

    return stats


def solve_one(name: str, config: SolveConfig, print_info: bool, stats_to_file: bool):
    """Helper function for solving a single puzzle."""
    puzzle = Puzzle.from_file(name)
    solve(puzzle, config, print_info, stats_to_file)


def solve_all(config: SolveConfig, print_info: bool, stats_to_file: bool):
    """Helper function for solving all puzzles."""
    names = all_puzzles_names()
    puzzles = [Puzzle.from_file(name) for name in names]
    for puzzle in puzzles:
        print_stamped(f"solving {puzzle.name}...")
        solve(puzzle, config, print_info, stats_to_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, nargs="?")
    args = parser.parse_args()

    if args.name is None:
        solve_all(SolveConfig(), True, True)
    else:
        solve_one(args.name, SolveConfig(), True, True)
