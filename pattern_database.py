import os
import sys
from datetime import datetime
from misc import (
    create_parent_directory,
    print_stamped,
    State,
)


def file_path(n: int, d: int):
    return f"./pattern_databases/n{n}-d{d}.txt"


def generate(n: int, d: int):
    path = file_path(n, d)
    if os.path.isfile(path):
        return  # already generated, so skip
    create_parent_directory(path)

    print_stamped(f"generating pattern database for n = {n} and d = {d}...")

    patterns: dict[str, int] = {}

    def recurse(depth: int, state: State):
        s = state.to_str_only_corners()
        if s not in patterns or depth < patterns[s]:
            patterns[s] = depth

        if depth < d:
            for ma in range(3):
                for mi in range(n):
                    for md in range(3):
                        recurse(depth + 1, state.execute_move(ma, mi, md))

    recurse(0, State.finished(n))

    with open(path, "w") as file:
        for state, remaining in patterns.items():
            file.write(f"{state} {remaining}\n")


def load(n: int, d: int):
    with open(file_path(n, d), "r") as file:

        def parse_line(line: str):
            s, r = line.split(" ")
            state = State.from_str(s)
            remaining = int(r)
            return state, remaining

        return list(map(parse_line, file))


# e.g. python pattern_database.py {n} {d}
if __name__ == "__main__":
    start = datetime.now()
    generate(int(sys.argv[1]), int(sys.argv[2]))
    print(f"took {datetime.now()-start} to complete!")
