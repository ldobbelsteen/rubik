import random
import sys
from misc import create_parent_directory, State
import itertools


def generate(n: int, randomizations: int):
    path = f"./puzzles/n{n}-random{randomizations}.txt"
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
    generate(int(sys.argv[1]), int(sys.argv[2]))
