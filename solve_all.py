import argparse
import os

from generate_random import PUZZLE_DIR
from puzzle import Puzzle
from solve import solve
from solve_config import SolveConfig
from tools import natural_sorted, print_stamped


def solve_all(dir: str, skip_solved: bool, config=SolveConfig()):
    """Solve all puzzles in a directory. Already solved puzzles can be skipped.."""
    puzzle_paths: list[str] = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        if os.path.isfile(path) and path.endswith(".txt"):
            if not skip_solved or not os.path.isfile(f"{path}.stats"):
                puzzle_paths.append(path)

    for path in natural_sorted(puzzle_paths):
        if config.print_info:
            print_stamped(f"solving '{path}'")
        puzzle = Puzzle.from_file(path)
        stats = solve(puzzle, config)
        stats.write_to_file(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=PUZZLE_DIR, type=str)
    parser.add_argument("--skip-solved", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    solve_all(args.dir, args.skip_solved, args.sat_solver)
