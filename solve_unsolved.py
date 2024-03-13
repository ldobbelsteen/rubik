import argparse
import os
from multiprocessing import cpu_count

from misc import natural_sorted
from solve import solve


def solve_unsolved(
    dir: str,
    sym_move_depth: int,
    only_larger_sym_moves: bool,
    max_processes: int,
    write_stats_file: bool,
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
            only_larger_sym_moves,
            max_processes,
            write_stats_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("--sym-moves-dep", default=0, type=int)
    parser.add_argument("--only-larger-sym-moves", default=True, type=bool)
    parser.add_argument("--max-processes", default=cpu_count() - 1, type=int)
    parser.add_argument("--write-stats-file", default=True, type=bool)
    args = parser.parse_args()
    solve_unsolved(
        args.dir,
        args.sym_moves_dep,
        args.only_larger_sym_moves,
        args.max_processes,
        args.write_stats_file,
    )
