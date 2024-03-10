import itertools
import os
import random
import sys

from misc import create_parent_directory
from puzzle import SUPPORTED_NS, Puzzle


def moveset(n: int):
    return list(itertools.product(range(3), range(n), range(3)))


def generate(n: int, randomizations: int, overwrite=False):
    if n not in SUPPORTED_NS:
        raise Exception(f"n = {n} not supported")

    path = f"./puzzles/n{n}-random{randomizations}.txt"
    if not overwrite and os.path.isfile(path):
        return
    create_parent_directory(path)

    p = Puzzle.finished(n)
    m = moveset(n)

    for _ in range(randomizations):
        ma, mi, md = random.choice(m)
        p.execute_move(ma, mi, md)

    with open(path, "w") as file:
        file.write(p.to_str())


# e.g. python generate.py {n} {randomization_count}
if __name__ == "__main__":
    generate(int(sys.argv[1]), int(sys.argv[2]), True)
