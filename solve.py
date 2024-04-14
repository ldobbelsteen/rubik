import argparse
import operator
from datetime import datetime, timedelta
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
from tools import print_stamped


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


class SolveInstance:
    """An instance of a solve operation for a puzzle."""

    def __init__(self, puzzle: Puzzle, k: int, config: SolveConfig):
        """Create a new solve instance by preparing all constraints in a goal."""
        self.n = puzzle.n
        self.k = k
        self.puzzle = puzzle
        self.config = config
        self.goal = z3.Goal()
        prep_start = datetime.now()

        # Initialize corner, edge and move state variables.
        self.corners = [
            [
                CornerStateZ3.new(
                    self.n, s, corner.x_hi, corner.y_hi, corner.z_hi, self.goal
                )
                for corner in CornerState.all_finished(self.n)
            ]
            for s in range(k + 1)
        ]
        self.edges = [
            [
                EdgeStateZ3.new(self.n, s, edge.a, edge.x_hi, edge.y_hi, self.goal)
                for edge in EdgeState.all_finished(self.n)
            ]
            for s in range(k + 1)
        ]
        self.moves = [MoveZ3(s, self.goal) for s in range(k)]

        # Fix the first state to the puzzle state.
        self.goal.add(z3.And(self.fix_state(0, puzzle.corners, puzzle.edges)))

        # Fix the last state to the finished state.
        self.goal.add(
            z3.And(
                self.fix_state(
                    -1,
                    CornerState.all_finished(self.n),
                    EdgeState.all_finished(self.n),
                )
            )
        )

        # Restrict cubie states according to moves.
        for s in range(k):
            if config.move_size == 0:  # Use basic move mappers.
                m = self.moves[s]

                for i, c in enumerate(self.corners[s]):
                    next = self.corners[s + 1][i]
                    self.goal.add(move_mappers.corner_x_hi(c, m, next))
                    self.goal.add(move_mappers.corner_y_hi(c, m, next))
                    self.goal.add(move_mappers.corner_z_hi(c, m, next))
                    self.goal.add(move_mappers.corner_r(c, m, next))
                    self.goal.add(move_mappers.corner_cw(c, m, next))

                # Add restrictions for the edge cubies.
                for i, e in enumerate(self.edges[s]):
                    next = self.edges[s + 1][i]
                    self.goal.add(move_mappers.edge_a(e, m, next))
                    self.goal.add(move_mappers.edge_x_hi(e, m, next))
                    self.goal.add(move_mappers.edge_y_hi(e, m, next))
                    self.goal.add(move_mappers.edge_r(e, next))

            else:  # Use stacked move mappers.
                if s % config.move_size != 0:
                    continue  # Skip if not multiple of move size.
                move_size = min(k - s, config.move_size)
                moveset = self.moves[s : s + move_size]

                # Add restrictions for the corner cubies.
                for i, c in enumerate(self.corners[s]):
                    next = self.corners[s + move_size][i]
                    self.goal.add(next == corner_stacked(c, moveset))

                # Add restrictions for the edge cubies.
                for i, e in enumerate(self.edges[s]):
                    next = self.edges[s + move_size][i]
                    self.goal.add(next == edge_stacked(e, moveset))

        # Add symmetric move sequence filters for n = 2.
        if self.n == 2:
            if config.enable_n2_move_filters_1_and_2:
                # Move filter #1 and #2
                for s in range(k - 1):
                    self.goal.add(
                        z3.Implies(
                            self.moves[s].ax == self.moves[s + 1].ax,
                            z3.Not(
                                z3.Or(
                                    z3.Not(self.moves[s + 1].hi),
                                    self.moves[s].hi,
                                )
                            ),
                        )
                    )

        # Add symmetric move sequence filters for n = 3.
        if self.n == 3:
            if config.enable_n3_move_filters_1_and_2:
                # Move filter #1 and #2
                for s in range(k - 1):
                    self.goal.add(
                        z3.Implies(
                            self.moves[s].ax == self.moves[s + 1].ax,
                            z3.Not(
                                z3.Or(
                                    z3.Not(self.moves[s].hi),
                                    self.moves[s + 1].hi,
                                )
                            ),
                        )
                    )

            if config.enable_n3_move_filters_3_and_4:
                # Move filter #3 and #4
                for s in range(k - 3):
                    self.goal.add(
                        z3.Implies(
                            z3.And(
                                self.moves[s].dr == 2,
                                self.moves[s + 1].dr == 2,
                                self.moves[s + 2].dr == 2,
                                self.moves[s + 3].dr == 2,
                            ),
                            z3.Not(
                                z3.Or(
                                    z3.And(
                                        self.moves[s].ax == self.moves[s + 3].ax,
                                        self.moves[s + 1].ax == self.moves[s + 2].ax,
                                        z3.Not(self.moves[s].hi),
                                    ),
                                    z3.And(
                                        self.moves[s].ax == self.moves[s + 1].ax,
                                        self.moves[s + 1].ax > self.moves[s + 2].ax,
                                        self.moves[s + 2].ax == self.moves[s + 3].ax,
                                    ),
                                )
                            ),
                        )
                    )

        if config.ban_repeated_states:
            # States cannot be repeated.
            for s1 in range(k + 1):
                for s2 in range(s1 + 1, k + 1):
                    self.goal.add(z3.Not(z3.And(self.identical_states(s1, s2))))

        if config.apply_theorem_11a:
            # Theorem 11.1a: sum x_i = 0 mod 3
            for s in range(k + 1):
                if len(self.corners[s]) > 0:
                    corner_sum = reduce(
                        operator.add,
                        [
                            cast(
                                z3.ArithRef,
                                z3.If(c.cw, c.r.arith_value(), -1 * c.r.arith_value()),
                            )
                            for c in self.corners[s]
                        ],
                    )
                    self.goal.add(corner_sum % 3 == 0)

        if config.apply_theorem_11b:
            # Theorem 11.1b: sum y_i = 0 mod 2
            for s in range(k + 1):
                if len(self.edges[s]) > 0:
                    edge_sum = reduce(
                        operator.add,
                        [
                            cast(
                                z3.ArithRef,
                                z3.If(e.r, 1, 0),
                            )
                            for e in self.edges[s]
                        ],
                    )
                    self.goal.add(edge_sum % 2 == 0)

        if config.enable_corner_min_patterns:
            for i, patterns in enumerate(load_corner_min_patterns(self.n)):
                for corner, depth in patterns.items():
                    for s in range(max(0, k + 1 - depth), k + 1):
                        self.goal.add(self.corners[s][i] != corner)

        if config.enable_edge_min_patterns:
            for i, patterns in enumerate(load_edge_min_patterns(self.n)):
                for edge, depth in patterns.items():
                    for s in range(max(0, k + 1 - depth), k + 1):
                        self.goal.add(self.edges[s][i] != edge)

        # Stop the timer and store the preparation time.
        self.prep_time = datetime.now() - prep_start

    def fix_state(
        self,
        s: int,
        corner_states: tuple[CornerState, ...],
        edge_states: tuple[EdgeState, ...],
    ):
        """Return conditions of a state having fixed corner and edge states."""
        conditions = []
        for c1, c2 in zip(self.corners[s], corner_states):
            conditions.append(c1 == c2)
        for e1, e2 in zip(self.edges[s], edge_states):
            conditions.append(e1 == e2)
        return conditions

    def identical_states(self, s1: int, s2: int):
        """Return conditions of two states being equal."""
        conditions = []
        for c1, c2 in zip(self.corners[s1], self.corners[s2]):
            conditions.append(c1 == c2)
        for e1, e2 in zip(self.edges[s1], self.edges[s2]):
            conditions.append(e1 == e2)
        return conditions

    def solve(self) -> tuple[MoveSeq | None, timedelta, timedelta]:
        """Compute the optimal solution for this instance. Returns the solution,
        along with the time it took to prepare the SAT model and the time it took
        to solve it.
        """
        if self.config.max_solver_threads == 0:
            # Parallelism is disabled: use a single-threaded tactic.
            tactic = cast(z3.Tactic, z3.Then(*self.config.tactics.get(), "sat"))
        else:
            # Configure Z3 to use parallelism.
            z3.set_param("parallel.enable", True)
            z3.set_param("parallel.threads.max", self.config.max_solver_threads)
            tactic = cast(z3.Tactic, z3.Then(*self.config.tactics.get(), "psat"))

        solver = tactic.solver()
        solver.add(self.goal)

        solve_start = datetime.now()
        result = solver.check()
        solve_time = datetime.now() - solve_start

        if result == z3.sat:
            model = solver.model()
            solution = MoveSeq(
                tuple(
                    Move(
                        model.eval(self.moves[s].ax.arith_value()).as_long(),  # type: ignore
                        z3.is_true(model.get_interp(self.moves[s].hi)),
                        model.eval(self.moves[s].dr.arith_value()).as_long(),  # type: ignore
                    )
                    for s in range(self.k)
                )
            )
            return solution, self.prep_time, solve_time
        elif result == z3.unsat:
            return None, self.prep_time, solve_time
        else:
            raise Exception(f"unexpected solver result: {result}")


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
        solution, prep_time, solve_time = SolveInstance(puzzle, k, config).solve()
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
