import random
import sys
import os
from misc import create_parent_directory, State
import itertools


def generate(n: int, randomizations: int, overwrite=False):
    path = f"./puzzles/n{n}-random{randomizations}.txt"
    if not overwrite and os.path.isfile(path):
        return
    create_parent_directory(path)

    state = State.finished(n)
    moves = list(itertools.product(range(3), range(n), range(3)))

    for _ in range(randomizations):
        ma, mi, md = random.choice(moves)
        state = state.execute_move(ma, mi, md)

    with open(path, "w") as file:
        file.write(state.to_str())


# e.g. python generate.py {n} {randomization_count}
if __name__ == "__main__":
    generate(int(sys.argv[1]), int(sys.argv[2]), True)
