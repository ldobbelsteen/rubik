import argparse
import random

from puzzle import DEFAULT_CENTER_COLORS, Puzzle
from state import Move, MoveSeq


def generate_random(
    n: int,
    randomizations: int,
    write_to_file: bool,
) -> tuple[Puzzle, MoveSeq]:
    """Generate a random puzzle by taking a specified number of random moves
    on a finished puzzle. The result is optionally output to file. The puzzle
    is returned along with the set of moves that were taken.
    """
    if n not in (2, 3):
        raise Exception(f"n = {n} not supported")

    name = f"n{n}-random{randomizations}.txt"
    puzzle = Puzzle.finished(n, name, DEFAULT_CENTER_COLORS)
    all_moves = Move.list_all()

    moves = []
    for _ in range(randomizations):
        move = random.choice(all_moves)
        puzzle = puzzle.execute_move(move)
        moves.append(move)

    if write_to_file:
        puzzle.to_file()

    return puzzle, tuple(moves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("randomizations", type=int)
    args = parser.parse_args()
    generate_random(args.n, args.randomizations, True)
