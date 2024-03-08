import random
import sys
import os
from puzzle import Puzzle
from misc import create_parent_directory
import itertools


def moveset(n: int):
    return list(itertools.product(range(3), range(n), range(3)))


def generate(n: int, randomizations: int, overwrite=False):
    if n < 2 or n > 3:
        raise Exception(f"n = {n} not supported")

    path = f"./puzzles/n{n}-random{randomizations}.txt"
    if not overwrite and os.path.isfile(path):
        return
    create_parent_directory(path)

    p = Puzzle.finished(n)
    moves = moveset(n)

    for _ in range(randomizations):
        ma, mi, md = random.choice(moves)
        p.execute_move(ma, mi, md)

    with open(path, "w") as file:
        file.write(p.to_str())


# e.g. python generate.py {n} {randomization_count}
if __name__ == "__main__":
    generate(int(sys.argv[1]), int(sys.argv[2]), True)
