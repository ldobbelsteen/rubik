import os
from multiprocessing import cpu_count
from solve import solve
from misc import natural_sorted

# e.g. python solve_unsolved.py
if __name__ == "__main__":
    puzzles: list[str] = []
    puzzle_dir = "./puzzles/"
    for file in os.listdir(puzzle_dir):
        path = os.path.join(puzzle_dir, file)
        if (
            os.path.isfile(path)
            and path.endswith(".txt")
            and not os.path.isfile(f"{path}.solution")
        ):
            puzzles.append(path)
    solve(natural_sorted(puzzles), cpu_count())
