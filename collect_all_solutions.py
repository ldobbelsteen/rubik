import argparse
from multiprocessing import cpu_count

from misc import print_stamped
from puzzle import Puzzle, move_name
from solve import solve, solve_for_k
from sym_move_seqs import MoveSequence


def collect_all_solutions(
    path: str,
    sym_move_depth: int,
    max_processes: int,
):
    puzzle = Puzzle.from_file(path)

    def canonicalize(solution: MoveSequence):
        return ", ".join([move_name(ma, mi, md) for ma, mi, md in solution])

    base_solution, base_result = solve(path, sym_move_depth, max_processes, True)
    if base_solution is None:
        raise Exception("puzzle has no solution")
    print_stamped(f"base solution: {base_solution}")
    print_stamped(f"base solution (canonical): {canonicalize(base_solution)}")
    k = base_result["k"]

    solutions = [base_solution]
    while True:
        solution, _, _ = solve_for_k(puzzle, k, sym_move_depth, solutions)
        if solution is None:
            break
        solutions.append(solution)
        print_stamped("-----------------------")
        print_stamped(f"new solution: {solution}")
        print_stamped(f"new solution (canonical): {canonicalize(solution)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--sym-moves-dep", default=0, type=int)
    parser.add_argument("--max-processes", default=cpu_count() - 1, type=int)
    args = parser.parse_args()
    collect_all_solutions(
        args.path,
        args.sym_moves_dep,
        args.max_processes,
    )
