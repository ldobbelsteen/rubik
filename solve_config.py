"""Functions and classes pertaining to solver configuration."""

from multiprocessing import cpu_count


class SolveConfig:
    """Configuration for a solve operation that can be passed to the solver."""

    def __init__(self):
        """Create a new config with default values."""
        self.move_size = 1
        self.use_sat_solver = True
        self.max_solver_threads = cpu_count() - 1
        self.apply_theorem_11a = False
        self.apply_theorem_11b = False
        self.print_info = True
