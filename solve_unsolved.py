import os
from multiprocessing import cpu_count
from solve import solve
import sys

# e.g. python solve_unsolved.py {pattern_depth}
if __name__ == "__main__":
    puzzles: list[str] = []
    puzzle_dir = "./puzzles/"
    for file in os.listdir(puzzle_dir):
        path = os.path.join(puzzle_dir, file)
        if (
            os.path.isfile(path)
            and path.endswith(".txt")
            and not os.path.isfile(path + ".solution")
        ):
            puzzles.append(path)
    solve(sorted(puzzles), cpu_count(), int(sys.argv[1]))