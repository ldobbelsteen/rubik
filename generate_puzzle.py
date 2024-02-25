import random
import sys
import numpy as np
from misc import execute_move, create_parent_directory
import itertools
import os


def generate(n: int, randomizations: int):
    path = f"./puzzles/n{n}-random{randomizations}"
    if os.path.isfile(path):
        return  # already generated, so skip
    create_parent_directory(path)

    state = np.array(
        [
            np.array([np.array([f for _ in range(n)]) for _ in range(n)])
            for f in range(6)
        ]
    )

    moves = list(itertools.product(range(n), range(3), range(3)))

    for _ in range(randomizations):
        mi, ma, md = random.choice(moves)
        execute_move(n, state, mi, ma, md)

    with open(path, "w") as file:
        file.write(str(state.tolist()))


# e.g. python generate_puzzles.py {n} {randomization_count}
if __name__ == "__main__":
    generate(int(sys.argv[1]), int(sys.argv[2]))
