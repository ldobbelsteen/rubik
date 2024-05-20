import argparse
import functools
import operator
import os
from datetime import datetime, timedelta
from typing import cast

import z3

from config import SolveConfig, gods_number
from cubie_min_patterns import load_corner_min_patterns, load_edge_min_patterns
from puzzle import (
    CornerState,
    EdgeState,
    Puzzle,
    all_puzzles_names,
)
from state import CornerStateZ3, EdgeStateZ3, Move, MoveSeq, MoveZ3
from stats import SolveStats
from tools import print_stamped


class SolveInstance:
    """An instance of a solve operation given a puzzle, number of moves
    and configuration. Contains Z3 variables representing the state. A Z3 goal
    can be built, which in turn can be solved to find a solution by Z3, or can be
    exported to a DIMACS file.
    """

    def __init__(self, puzzle: Puzzle, k: int, config: SolveConfig):
        """Initialize an instance given a puzzle, number of moves and config."""
        self.puzzle = puzzle
        self.k = k
        self.config = config
        self.n = puzzle.n

        # List of constraints which should always be present in the goal.
        self.base_constraints: list[z3.BoolRef] = []

        # Initialize the states of the corners for each step.
        self.corners = [
            [
                CornerStateZ3.new(
                    self.n,
                    s,
                    corner.x_hi,
                    corner.y_hi,
                    corner.z_hi,
                    self.base_constraints,
                    self.config.enable_minimal_moves_n2,
                )
                for corner in CornerState.all_finished(self.n)
            ]
            for s in range(self.k + 1)
        ]

        # Initialize the states of the edges for each step.
        self.edges = [
            [
                EdgeStateZ3.new(
                    self.n,
                    s,
                    edge.a,
                    edge.x_hi,
                    edge.y_hi,
                    self.base_constraints,
                    self.config.enable_minimal_moves_n2,
                )
                for edge in EdgeState.all_finished(self.n)
            ]
            for s in range(self.k + 1)
        ]

        # Initialize the move for each stpe.
        self.moves = [MoveZ3(s, self.base_constraints, self.n, self.config.enable_minimal_moves_n2) for s in range(k)]

    def corner_states_equal(self, s: int, states: tuple[CornerState, ...]):
        """Return a list of constraints that enforce the corner states to be equal."""
        return [c1 == c2 for c1, c2 in zip(self.corners[s], states)]

    def edge_states_equal(self, s: int, states: tuple[EdgeState, ...]):
        """Return a list of constraints that enforce the edge states to be equal."""
        return [e1 == e2 for e1, e2 in zip(self.edges[s], states)]

    def identical_corner_states(self, s1: int, s2: int):
        """Return a list of constraints that enforce the corner states
        to be identical.
        """
        return [c1 == c2 for c1, c2 in zip(self.corners[s1], self.corners[s2])]

    def identical_edge_states(self, s1: int, s2: int):
        """Return a list of constraints that enforce the edge states to be identical."""
        return [e1 == e2 for e1, e2 in zip(self.edges[s1], self.edges[s2])]

    def build_goal(self, timeout_secs: float | None) -> z3.Goal | None:
        """Build the goal for this instance by accumulating contraints. Returns
        a Z3 goal object (the tactics from the config are already applied).
        Applies a timeout to applying the tactics if specified and returns nothing
        if the timeout is reached.
        """
        start = datetime.now()
        goal = z3.Goal()

        # Add base constraints to the goal.
        goal.add(self.base_constraints)

        # Fix the first state to the puzzle state.
        goal.add(
            z3.And(
                *self.corner_states_equal(0, self.puzzle.corners),
                *self.edge_states_equal(0, self.puzzle.edges),
            )
        )

        # Fix the last state to the finished state.
        goal.add(
            z3.And(
                *self.corner_states_equal(-1, CornerState.all_finished(self.n)),
                *self.edge_states_equal(-1, EdgeState.all_finished(self.n)),
            )
        )

        # Restrict cubie states according to moves.
        for s in range(self.k):
            if s % self.config.move_size != 0:
                continue  # Skip if not multiple of move size.
            move_size = min(self.k - s, self.config.move_size)
            current_moves = self.moves[s : s + move_size]

            # Add restrictions for the corner cubies.
            for i, c in enumerate(self.corners[s]):
                next = self.corners[s + move_size][i]
                goal.add(
                    next
                    == functools.reduce(
                        lambda s, m: s.execute_move(m), current_moves, c
                    )
                )

            # Add restrictions for the edge cubies.
            for i, e in enumerate(self.edges[s]):
                next = self.edges[s + move_size][i]
                goal.add(
                    next
                    == functools.reduce(
                        lambda s, m: s.execute_move(m), current_moves, e
                    )
                )

        # Add symmetric move sequence filters for n = 2.
        if self.n == 2:
            if self.config.enable_n2_move_filters_1_and_2:
                # Move filter #1 and #2
                for s in range(self.k - 1):
                    if self.config.enable_minimal_moves_n2:
                        goal.add(
                            z3.Not(self.moves[s].ax == self.moves[s + 1].ax),
                        )
                    else:
                        goal.add(
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
            if self.config.enable_n3_move_filters_1_and_2:
                # Move filter #1 and #2
                for s in range(self.k - 1):
                    goal.add(
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

            if self.config.enable_n3_move_filters_3_and_4:
                # Move filter #3 and #4
                for s in range(self.k - 3):
                    goal.add(
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

        if self.config.ban_repeated_states:
            # States cannot be repeated.
            for s1 in range(self.k + 1):
                for s2 in range(s1 + 1, self.k + 1):
                    goal.add(
                        z3.Not(
                            z3.And(
                                *self.identical_corner_states(s1, s2),
                                *self.identical_edge_states(s1, s2),
                            )
                        )
                    )

        if self.config.apply_theorem_11a:
            # Theorem 11.1a: sum x_i = 0 mod 3
            for s in range(self.k + 1):
                if len(self.corners[s]) > 0:
                    corner_sum = functools.reduce(
                        operator.add,
                        [
                            cast(
                                z3.ArithRef,
                                z3.If(c.cw, c.r.arith_value(), -1 * c.r.arith_value()),
                            )
                            for c in self.corners[s]
                        ],
                    )
                    goal.add(corner_sum % 3 == 0)

        if self.config.apply_theorem_11b:
            # Theorem 11.1b: sum y_i = 0 mod 2
            for s in range(self.k + 1):
                if len(self.edges[s]) > 0:
                    edge_sum = functools.reduce(
                        operator.add,
                        [
                            cast(
                                z3.ArithRef,
                                z3.If(e.r, 1, 0),
                            )
                            for e in self.edges[s]
                        ],
                    )
                    goal.add(edge_sum % 2 == 0)

        if self.config.enable_corner_min_patterns:
            for i, patterns in enumerate(load_corner_min_patterns(self.n)):
                for corner, depth in patterns.items():
                    for s in range(max(0, self.k + 1 - depth), self.k + 1):
                        goal.add(self.corners[s][i] != corner)

        if self.config.enable_edge_min_patterns:
            for i, patterns in enumerate(load_edge_min_patterns(self.n)):
                for edge, depth in patterns.items():
                    for s in range(max(0, self.k + 1 - depth), self.k + 1):
                        goal.add(self.edges[s][i] != edge)

        tactic_strs = self.config.tactics.get()
        if len(tactic_strs) > 0:
            if len(tactic_strs) > 1:
                tactic = z3.Then(*tactic_strs)
            elif len(tactic_strs) == 1:
                tactic = z3.Tactic(tactic_strs[0])

            # Set timeout of remaining time.
            if timeout_secs is not None:
                elapsed = datetime.now() - start
                remaining = timeout_secs - elapsed.total_seconds()
                tactic = z3.TryFor(tactic, int(remaining * 1000))

            try:
                applied = tactic(goal)
            except z3.z3types.Z3Exception as e:
                # Catch error only if it was a timeout, else reraise.
                if e.value.decode("utf-8") == "canceled":
                    return None
                raise e

            assert len(applied) == 1
            return applied[0]
        else:
            return goal

    def solve(
        self,
        timeout_secs: float | None,
    ) -> tuple[MoveSeq | None, timedelta, timedelta] | None:
        """Compute the optimal solution for this instance. Returns the solution,
        along with the time it took to build the goal, and the time it took to solve it.
        If the instance is UNSAT, the solution is None. If the timeout was triggered,
        nothing is returned.
        """
        prep_start = datetime.now()
        goal = self.build_goal(timeout_secs)
        prep_time = datetime.now() - prep_start

        # Check if the prep operation timed out.
        if goal is None:
            return None

        if self.config.max_solver_threads == 0:
            # Parallelism is disabled; use a single-threaded solver.
            solver = z3.Tactic("sat").solver()
        else:
            # Configure Z3 to use parallelism and create parallel solver.
            z3.set_param("parallel.enable", True)
            z3.set_param("parallel.threads.max", self.config.max_solver_threads)
            solver = z3.Tactic("psat").solver()

        # Set timeout of the original time minus the time we already spent preparing.
        if timeout_secs is not None:
            remaining_timeout_secs = timeout_secs - prep_time.total_seconds()
            solver.set("timeout", int(remaining_timeout_secs * 1000))
        else:
            remaining_timeout_secs = None

        solver.add(goal)

        solve_start = datetime.now()
        result = solver.check()
        solve_time = datetime.now() - solve_start

        # Check if the solve operation timed out.
        if (
            remaining_timeout_secs is not None
            and remaining_timeout_secs <= solve_time.total_seconds()
        ):
            return None

        if result == z3.sat:
            model = solver.model()
            if self.config.enable_minimal_moves_n2 and self.n == 2:
                solution = MoveSeq(
                    tuple(
                        Move(
                            model.eval(self.moves[s].ax.arith_value()).as_long(),  # type: ignore
                            True,
                            model.eval(self.moves[s].dr.arith_value()).as_long(),  # type: ignore
                        )
                        for s in range(self.k)
                    )
                )
            else:
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
            return solution, prep_time, solve_time
        elif result == z3.unsat:
            return None, prep_time, solve_time
        else:
            raise Exception(f"unexpected solver result: {result}")

    def to_dimacs(self, path: str):
        """Export the instance to a DIMACS file at the given path."""
        goal = self.build_goal(None)
        goal_cnf = z3.Tactic("tseitin-cnf").apply(goal)[0]

        def child_map(child) -> int:
            nonlocal var_count
            assert isinstance(child, z3.BoolRef)
            if z3.is_not(child):
                assert child.num_args() == 1
                name = child.arg(0).decl().name()
                if name not in var_name_mapping:
                    var_count += 1
                    var_name_mapping[name] = var_count
                return -var_name_mapping[name]
            else:
                name = child.decl().name()
                if name not in var_name_mapping:
                    var_count += 1
                    var_name_mapping[name] = var_count
                return var_name_mapping[name]

        var_count: int = 1
        clauses: list[list[int]] = []
        var_name_mapping: dict[str, int] = {}
        for clause in goal_cnf:
            assert isinstance(clause, z3.BoolRef)
            if z3.is_or(clause):
                clauses.append([child_map(child) for child in clause.children()])
            else:
                clauses.append([child_map(clause)])

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"p cnf {var_count} {len(clauses)}\n")
            for c in clauses:
                f.write(" ".join(map(str, c)) + " 0\n")


def solve(
    puzzle: Puzzle,
    config: SolveConfig,
    timeout_secs: float | None,
    print_info: bool,
    stats_to_file: bool,
) -> SolveStats | None:
    """Compute the optimal solution for a puzzle within an upperbound for the number
    of moves. Returns the statistics of the solve operation. Allows for setting
    a timeout for the allowed time spent. If the timeout is reached, the function
    returns the stats up to that point.
    """
    stats = SolveStats(puzzle, config)
    k_upperbound = gods_number(puzzle.n)

    for k in range(k_upperbound + 1):
        instance = SolveInstance(puzzle, k, config)
        k_timeout_secs = (
            timeout_secs - stats.total_time().total_seconds()
            if timeout_secs is not None
            else None
        )
        result = instance.solve(k_timeout_secs)
        if result is None:
            if print_info:
                print_stamped(f"k = {k}: timeout reached, quitting...")
            return None

        solution, prep_time, solve_time = result
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


def solve_one(
    name: str,
    config: SolveConfig,
    timeout_secs: float | None,
    print_info: bool,
    stats_to_file: bool,
):
    """Helper function for solving a single puzzle."""
    puzzle = Puzzle.from_file(name)
    solve(puzzle, config, timeout_secs, print_info, stats_to_file)


def solve_all(
    config: SolveConfig,
    timeout_secs: float | None,
    print_info: bool,
    stats_to_file: bool,
):
    """Helper function for solving all puzzles."""
    names = all_puzzles_names()
    puzzles = [Puzzle.from_file(name) for name in names]
    for puzzle in puzzles:
        print_stamped(f"solving {puzzle.name}...")
        solve(puzzle, config, timeout_secs, print_info, stats_to_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, nargs="?")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--disable-info-print", action=argparse.BooleanOptionalAction)
    parser.add_argument("--disable-stats-file", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.name is None:
        solve_all(
            SolveConfig(),
            args.timeout,
            not args.disable_info_print,
            not args.disable_stats_file,
        )
    else:
        solve_one(
            args.name,
            SolveConfig(),
            args.timeout,
            not args.disable_info_print,
            not args.disable_stats_file,
        )
