import argparse
import os
import random

from puzzle import DEFAULT_CENTER_COLORS, Puzzle, all_moves
from tools import create_parent_directory

PUZZLE_DIR = "./puzzles"


def generate_random(n: int, randomizations: int, write_to_file: bool) -> Puzzle:
    """Generate a random puzzle by taking a specified number of random moves
    on a finished puzzle. The result is optionally output to file."""
    if n != 2 and n != 3:
        raise Exception(f"n = {n} not supported")

    puzzle = Puzzle.finished(n, DEFAULT_CENTER_COLORS)
    moves = all_moves()

    for _ in range(randomizations):
        move = random.choice(moves)
        puzzle = puzzle.execute_move(move)

    if write_to_file:
        path = os.path.join(PUZZLE_DIR, f"n{n}-random{randomizations}.txt")
        create_parent_directory(path)
        with open(path, "w") as file:
            file.write(puzzle.to_str())

    return puzzle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("randomizations", type=int)
    args = parser.parse_args()
    generate_random(args.n, args.randomizations, True)
