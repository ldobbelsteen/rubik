import argparse
import random

from misc import create_parent_directory
from puzzle import SUPPORTED_NS, Puzzle, list_all_moves


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("randomizations", type=int)
    args = parser.parse_args()
    generate_puzzle(args.n, args.randomizations)
