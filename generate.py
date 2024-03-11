import itertools
import random
import sys

from misc import create_parent_directory
from puzzle import SUPPORTED_NS, Puzzle


def list_all_moves(n: int):
    """List all possible moves given n."""
    return list(itertools.product(range(3), range(n), range(3)))


def generate_puzzle(n: int, randomizations: int):
    if n not in SUPPORTED_NS:
        raise Exception(f"n = {n} not supported")

    path = f"./puzzles/n{n}-random{randomizations}.txt"
    create_parent_directory(path)

    p = Puzzle.finished(n)
    m = list_all_moves(n)

    for _ in range(randomizations):
        ma, mi, md = random.choice(m)
        p.execute_move(ma, mi, md)

    with open(path, "w") as file:
        file.write(p.to_str())


# e.g. python generate.py {n} {randomization_count}
if __name__ == "__main__":
    generate_puzzle(int(sys.argv[1]), int(sys.argv[2]))
