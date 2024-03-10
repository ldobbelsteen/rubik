import os
from multiprocessing import cpu_count

from misc import natural_sorted
from solve import solve

# e.g. python solve_unsolved.py
if __name__ == "__main__":
    dir = "./puzzles/"
    puzzles: list[str] = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        if (
            os.path.isfile(path)
            and path.endswith(".txt")
            and not os.path.isfile(f"{path}.solution")
        ):
            puzzles.append(path)
    solve(natural_sorted(puzzles), cpu_count())
