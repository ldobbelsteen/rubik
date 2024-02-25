import os
import sys
from datetime import datetime
from misc import execute_move, reverse_direction, create_parent_directory, state_to_str
import numpy as np


def file_path(n: int, d: int):
    return f"./pattern_databases/n{n}-d{d}.txt"


def generate(n: int, d: int):
    path = file_path(n, d)
    if os.path.isfile(path):
        return  # already generated, so skip
    create_parent_directory(path)

    patterns: dict[str, int] = {}

    def recurse(depth: int, state: np.ndarray):
        if depth == d:
            return
        else:
            for mi in range(n):
                for ma in range(3):
                    for md in range(3):
                        execute_move(n, state, mi, ma, md)
                        depth += 1

                        state_str = state_to_str(state)
                        if state_str not in patterns or depth < patterns[state_str]:
                            patterns[state_str] = depth
                        recurse(depth, state)

                        execute_move(n, state, mi, ma, reverse_direction(md))
                        depth -= 1

    recurse(
        0,
        np.array(
            [
                np.array([np.array([f for _ in range(n)]) for _ in range(n)])
                for f in range(6)
            ]
        ),
    )

    with open(path, "w") as file:
        for state, k in patterns.items():
            file.write(f"{state} {k}\n")


def load(n: int, d: int):
    with open(file_path(n, d), "r") as file:
        print(file)
        raise Exception("TODO")


# e.g. python pattern_database.py {n} {d}
if __name__ == "__main__":
    start = datetime.now()
    generate(int(sys.argv[1]), int(sys.argv[2]))
    print(f"took {datetime.now()-start} to complete!")
