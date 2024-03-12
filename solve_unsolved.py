import os

from misc import natural_sorted
from solve import solve


def solve_unsolved():
    dir = "./puzzles/"
    puzzles: list[str] = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        if (
            os.path.isfile(path)
            and path.endswith(".txt")
            and not os.path.isfile(f"{path}.stats")
        ):
            puzzles.append(path)
    for puzzle in natural_sorted(puzzles):
        solve(puzzle)


# e.g. python solve_unsolved.py
if __name__ == "__main__":
    solve_unsolved()
