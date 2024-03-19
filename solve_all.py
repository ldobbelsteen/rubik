import argparse
import os
from multiprocessing import cpu_count

from generate_random import PUZZLE_DIR
from puzzle import Puzzle
from solve import solve
from tools import gods_number, natural_sorted, print_stamped


def solve_all(
    dir: str,
    skip_solved: bool,
    move_stacking: bool,
    sym_move_depth: int,
    max_threads: int,
    disable_stats_file: bool,
):
    """Solve all puzzles in a directory. Already solved puzzles can be filtered out."""
    puzzle_paths: list[str] = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        if os.path.isfile(path) and path.endswith(".txt"):
            if not skip_solved or not os.path.isfile(f"{path}.stats"):
                puzzle_paths.append(path)

    for path in natural_sorted(puzzle_paths):
        print_stamped(f"solving '{path}'")
        puzzle = Puzzle.from_file(path)
        stats = solve(
            puzzle,
            gods_number(puzzle.n),
            max_threads,
            move_stacking,
            sym_move_depth,
            True,
        )
        if not disable_stats_file:
            stats.write_to_file(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=PUZZLE_DIR, type=str)
    parser.add_argument("--skip-solved", action=argparse.BooleanOptionalAction)
    parser.add_argument("--move-stacking", action=argparse.BooleanOptionalAction)
    parser.add_argument("--sym-moves-dep", default=0, type=int)
    parser.add_argument("--max-threads", default=cpu_count() - 1, type=int)
    parser.add_argument("--disable-stats-file", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    solve_all(
        args.dir,
        args.skip_solved,
        args.move_stacking,
        args.sym_moves_dep,
        args.max_threads,
        args.disable_stats_file,
    )
