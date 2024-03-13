import argparse
import os
from multiprocessing import cpu_count

from misc import natural_sorted
from solve import solve


def solve_unsolved(
    dir: str,
    sym_move_depth: int,
    max_processes: int,
    disable_stats_file: bool,
):
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
        solve(
            puzzle,
            sym_move_depth,
            max_processes,
            disable_stats_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("--sym-moves-dep", default=0, type=int)
    parser.add_argument("--max-processes", default=cpu_count() - 1, type=int)
    parser.add_argument("--disable-stats-file", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    solve_unsolved(
        args.path,
        args.sym_moves_dep,
        args.max_processes,
        args.disable_stats_file,
    )
