"""Functions for validating the puzzle representation and the functions permuting it."""

from puzzle import DEFAULT_CENTER_COLORS, MoveSeq, Puzzle, all_moves, move_names
from solve import solve_for_k
from solve_config import SolveConfig
from tools import print_stamped


def validate_random_move_solvability_rec(
    config: SolveConfig,
    puzzle: Puzzle,
    moves: MoveSeq,
    max_d: int,
):
    """Recursively validate the solvability of a puzzle after a random move."""
    for k in range(len(moves) + 1):
        sol, _, _ = solve_for_k(puzzle, k, config)
        if sol is not None:
            break
    else:
        raise Exception(f"not solvable after: {move_names(moves)}")
    if len(moves) < max_d:
        for move in all_moves():
            validate_random_move_solvability_rec(
                config,
                puzzle.execute_move(move),
                (*moves, move),
                max_d,
            )


def validate_random_move_solvability(max_d: int):
    """Validate the solvability of a puzzle after a random move."""
    config = SolveConfig()
    config.print_info = False
    for n in [2, 3]:
        print_stamped(f"validating solvability for n = {n}")
        validate_random_move_solvability_rec(
            config,
            Puzzle.finished(n, DEFAULT_CENTER_COLORS),
            (),
            max_d,
        )


if __name__ == "__main__":
    validate_random_move_solvability(2)
