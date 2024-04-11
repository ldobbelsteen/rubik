import json
import os
from datetime import timedelta

from puzzle import Puzzle
from solve_config import SolveConfig
from state import MoveSeq
from tools import str_to_file

SOLVE_RESULTS_DIR = "./solve_results"


class SolveStats:
    """Statistics about the solving process."""

    def __init__(self, puzzle: Puzzle, config: SolveConfig):
        """Initialize the statistics."""
        self.puzzle = puzzle
        self.config = config

        self.solution = None
        self.prep_times: dict[int, timedelta] = {}
        self.solve_times: dict[int, timedelta] = {}

    def register_solution(
        self,
        k: int,
        solution: MoveSeq | None,
        prep_time: timedelta,
        solve_time: timedelta,
    ):
        """Register a solution and its times for a specific k."""
        if solution is not None and (
            self.solution is None or len(solution) < len(self.solution)
        ):
            self.solution = solution

        assert k not in self.prep_times and k not in self.solve_times
        self.prep_times[k] = prep_time
        self.solve_times[k] = solve_time

    def k(self):
        """Return the current k if a solution has been found."""
        if self.solution is not None:
            return len(self.solution)

    def total_prep_time(self):
        """Return the total preparation time."""
        k = self.k()
        if k is None:
            return sum(self.prep_times.values(), timedelta())
        else:
            return sum([v for kp, v in self.prep_times.items() if kp <= k], timedelta())

    def total_solve_time(self):
        """Return the total solving time."""
        k = self.k()
        if k is None:
            return sum(self.solve_times.values(), timedelta())
        else:
            return sum(
                [v for kp, v in self.solve_times.items() if kp <= k], timedelta()
            )

    def to_dict(self):
        """Return the statistics as a dictionary."""
        result: dict[str, str | int | dict[int, str] | tuple[str, ...]] = {}

        k = self.k()
        if k:
            result["k"] = k

        if self.solution is not None:
            result["moves"] = tuple(map(str, self.solution))

        result["total_prep_time"] = str(self.total_prep_time())
        result["total_solve_time"] = str(self.total_solve_time())
        result["prep_times"] = {k: str(t) for k, t in sorted(self.prep_times.items())}
        result["solve_times"] = {k: str(t) for k, t in sorted(self.solve_times.items())}

        return result

    def to_file(self):
        """Write the statistics to a file."""
        os.makedirs(SOLVE_RESULTS_DIR, exist_ok=True)
        str_to_file(
            json.dumps(self.to_dict(), indent=4),
            os.path.join(SOLVE_RESULTS_DIR, self.puzzle.name + ".json"),
        )
