import os
import sys
from multiprocessing import cpu_count
from solve import solve_puzzles

# python solve_unsolved.py {puzzle_dir}
if __name__ == "__main__":
    puzzles: list[str] = []
    puzzle_dir = sys.argv[1]
    for file in os.listdir(puzzle_dir):
        path = os.path.join(puzzle_dir, file)
        if (
            os.path.isfile(path)
            and path.endswith(".cube")
            and not os.path.isfile(path + ".result")
        ):
            puzzles.append(path)
    solve_puzzles(sorted(puzzles), cpu_count())
