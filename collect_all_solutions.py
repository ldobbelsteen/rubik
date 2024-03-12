import sys

from misc import print_stamped
from puzzle import Puzzle, move_name
from solve import solve, solve_for_k
from sym_move_seqs import MoveSequence


def collect_all_solutions(path: str):
    puzzle = Puzzle.from_file(path)

    def canonicalize(solution: MoveSequence):
        return ", ".join([move_name(ma, mi, md) for ma, mi, md in solution])

    base_solution, base_result = solve(path, write_stats_file=False)
    if base_solution is None:
        raise Exception("puzzle has no solution")
    print_stamped(f"base solution: {base_solution}")
    print_stamped(f"base solution (canonical): {canonicalize(base_solution)}")
    k = base_result["k"]

    solutions = [base_solution]
    while True:
        solution, _, _ = solve_for_k(puzzle, k, solutions)
        if solution is None:
            break
        solutions.append(solution)
        print_stamped("-----------------------")
        print_stamped(f"new solution: {solution}")
        print_stamped(f"new solution (canonical): {canonicalize(solution)}")


# e.g. python collect_all_solutions.py ./puzzles/n2-random7.txt
if __name__ == "__main__":
    collect_all_solutions(sys.argv[1])
