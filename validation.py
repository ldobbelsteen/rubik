from move_symmetries import allowed_by_filters
from puzzle import MoveSeq, Puzzle, move_names


def validate_solution(puzzle: Puzzle, solution: MoveSeq):
    canon = move_names(solution)

    if not allowed_by_filters(puzzle.n, solution):
        raise Exception(f"solution should have been filtered: {canon}")

    for move in solution:
        puzzle = puzzle.execute_move(move)
    if puzzle != Puzzle.finished(puzzle.n, puzzle.center_colors):
        raise Exception(f"solution is not actual solution: {canon}")
