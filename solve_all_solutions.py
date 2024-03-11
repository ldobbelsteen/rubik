import sys

from misc import print_stamped
from puzzle import Puzzle, move_name
from solve import solve, solve_for_k

# e.g. python solve_all_solutions.py ./puzzles/n2-random7.txt
if __name__ == "__main__":
    path = sys.argv[1]
    puzzle = Puzzle.from_file(path)

    def canonicalize(solution: list[tuple[int, int, int]]):
        return [move_name(puzzle.n, ma, mi, md) for ma, mi, md in solution]

    base_solution, base_result = solve(path)
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
        print_stamped(f"new solution: {solution}")
        print_stamped(f"new solution (canonical): {canonicalize(solution)}")
