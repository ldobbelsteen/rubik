"""Functions and classes pertaining to solver configuration."""

from multiprocessing import cpu_count


def gods_number(n: int):
    """Return the known God number for a specific n."""
    match n:
        case 1:
            return 0
        case 2:
            return 11
        case 3:
            return 20
        case _:
            raise Exception(f"God's number not known for n = {n}")


class SolveConfig:
    """Configuration for a solve operation that can be passed to the solver."""

    def __init__(
        self,
        move_size: int,
        max_solver_threads: int,
        apply_theorem_11a: bool,
        apply_theorem_11b: bool,
    ):
        """Create a new solve configuration."""
        self.move_size = move_size
        self.max_solver_threads = max_solver_threads
        self.apply_theorem_11a = apply_theorem_11a
        self.apply_theorem_11b = apply_theorem_11b

    @staticmethod
    def default():
        """Create a config with default values."""
        return SolveConfig(1, cpu_count() - 1, False, False)
