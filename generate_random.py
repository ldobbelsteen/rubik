import argparse
import random

from puzzle import DEFAULT_CENTER_COLORS, Puzzle
from state import Move


def generate_random(n: int, randomizations: int, write_to_file: bool) -> Puzzle:
    """Generate a random puzzle by taking a specified number of random moves
    on a finished puzzle. The result is optionally output to file.
    """
    if n not in (2, 3):
        raise Exception(f"n = {n} not supported")

    name = f"n{n}-random{randomizations}.txt"
    puzzle = Puzzle.finished(n, name, DEFAULT_CENTER_COLORS)
    moves = Move.list_all()

    for _ in range(randomizations):
        move = random.choice(moves)
        puzzle = puzzle.execute_move(move)

    if write_to_file:
        puzzle.to_file()

    return puzzle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("randomizations", type=int)
    args = parser.parse_args()
    generate_random(args.n, args.randomizations, True)
