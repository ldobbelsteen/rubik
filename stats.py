import json
from datetime import timedelta

from puzzle import MoveSeq, move_name


class Stats:
    def __init__(self, max_processes: int, k_upperbound: int):
        self.max_processes = max_processes
        self.k_upperbound = k_upperbound

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
        if solution is not None and (
            self.solution is None or len(solution) < len(self.solution)
        ):
            self.solution = solution

        assert k not in self.prep_times and k not in self.solve_times
        self.prep_times[k] = prep_time
        self.solve_times[k] = solve_time

    def k(self):
        if self.solution is not None:
            return len(self.solution)

    def total_prep_time(self):
        k = self.k()
        if k is None:
            return sum(self.prep_times.values(), timedelta())
        else:
            return sum([v for kp, v in self.prep_times.items() if kp <= k], timedelta())

    def total_solve_time(self):
        k = self.k()
        if k is None:
            return sum(self.solve_times.values(), timedelta())
        else:
            return sum(
                [v for kp, v in self.solve_times.items() if kp <= k], timedelta()
            )

    def to_dict(self):
        result: dict[str, str | int | list[str] | dict[int, str]] = {}

        k = self.k()
        if k:
            result["k"] = k

        if self.solution is not None:
            result["moves"] = [move_name(move) for move in self.solution]

        result["total_prep_time"] = str(self.total_prep_time())
        result["total_solve_time"] = str(self.total_solve_time())
        result["prep_times"] = {k: str(t) for k, t in sorted(self.prep_times.items())}
        result["solve_times"] = {k: str(t) for k, t in sorted(self.solve_times.items())}
        result["k_upperbound"] = self.k_upperbound
        result["max_processes"] = self.max_processes

        return result

    def write_to_file(self, puzzle_path: str):
        with open(f"{puzzle_path}.stats", "w") as file:
            file.write(json.dumps(self.to_dict(), indent=4))
