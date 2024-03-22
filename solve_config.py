from multiprocessing import cpu_count


class SolveConfig:
    def __init__(self):
        self.move_stacking = False
        self.use_sat_solver = False
        self.symmetric_move_ban_depth = 0
        self.max_solver_threads = cpu_count() - 1
        self.apply_theorem_11a = False
        self.apply_theorem_11b = False
        self.print_info = True
