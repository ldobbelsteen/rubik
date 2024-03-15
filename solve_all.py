import argparse
import os
from multiprocessing import cpu_count

from generate_random import PUZZLE_DIR
from solve import solve
from tools import natural_sorted


def solve_all(
    dir: str,
    skip_solved: bool,
    move_stacking: bool,
    sym_move_depth: int,
    max_processes: int,
    disable_stats_file: bool,
):
    """Solve all puzzles in a directory. Already solved puzzles can be filtered out."""
    puzzles: list[str] = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        if os.path.isfile(path) and path.endswith(".txt"):
            if not skip_solved or not os.path.isfile(f"{path}.stats"):
                puzzles.append(path)

    for puzzle in natural_sorted(puzzles):
        solve(
            puzzle,
            move_stacking,
            sym_move_depth,
            max_processes,
            disable_stats_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", default=PUZZLE_DIR, type=str)
    parser.add_argument("--skip-solved", action=argparse.BooleanOptionalAction)
    parser.add_argument("--move-stacking", action=argparse.BooleanOptionalAction)
    parser.add_argument("--sym-moves-dep", default=0, type=int)
    parser.add_argument("--max-processes", default=cpu_count() - 1, type=int)
    parser.add_argument("--disable-stats-file", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    solve_all(
        args.path,
        args.skip_solved,
        args.move_stacking,
        args.sym_moves_dep,
        args.max_processes,
        args.disable_stats_file,
    )
