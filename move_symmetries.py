import argparse
import ast
import os

from puzzle import (
    DEFAULT_CENTER_COLORS,
    MoveSeq,
    Puzzle,
    all_moves,
    move_names,
    parse_move,
)
from tools import create_parent_directory, print_stamped


def symmetries_file_path(n: int, d: int):
    dir = os.path.dirname(__file__)
    filename = f"./generated_move_symmetries/n{n}-d{d}-symmetries.txt"
    return os.path.join(dir, filename)


def filtered_file_path(n: int, d: int):
    dir = os.path.dirname(__file__)
    filename = f"./generated_move_symmetries/n{n}-d{d}-filtered.txt"
    return os.path.join(dir, filename)


def unique_file_path(n: int, d: int):
    dir = os.path.dirname(__file__)
    filename = f"./generated_move_symmetries/n{n}-d{d}-unique.txt"
    return os.path.join(dir, filename)


def allowed_by_filters(n: int, seq: MoveSeq) -> bool:
    """Return whether a move sequence is allowed by applying filters."""
    k = len(seq)

    def axs(s: int):
        return seq[s][0]

    def his(s: int):
        return seq[s][1]

    def drs(s: int):
        return seq[s][2]

    if n == 2:
        # Move filter #1 and #2
        for s in range(k - 1):
            if axs(s) == axs(s + 1) and (not his(s + 1) or his(s)):
                return False

        # Move filter #3
        for s in range(k - 2):
            if axs(s) == axs(s + 2) and drs(s) == 0 and drs(s + 2) == 1:
                return False

        # Move filter #4
        for s in range(k - 2):
            if axs(s) == axs(s + 1) and drs(s) == drs(s + 1):
                return False

        return True

    if n == 3:
        # Move filter #1 and #2
        for s in range(k - 1):
            if axs(s) == axs(s + 1) and (not his(s) or his(s + 1)):
                return False

        for s in range(k - 3):
            if drs(s) == 2 and drs(s + 1) == 2 and drs(s + 2) == 2 and drs(s + 3) == 2:
                # Move filter #3
                if axs(s) == axs(s + 3) and axs(s + 1) == axs(s + 2) and not his(s):
                    return False

                # Move filter #4
                if (
                    axs(s) == axs(s + 1)
                    and axs(s + 1) > axs(s + 2)
                    and axs(s + 2) == axs(s + 3)
                ):
                    return False

        return True

    raise Exception("invalid n or misconfigured filters")

    if n == 3:
        for s in range(k - 3):
            if drs(s) == 2 and drs(s + 1) == 2 and drs(s + 2) == 2 and drs(s + 3) == 2:
                # Move filter #3
                if axs(s) == axs(s + 3) and axs(s + 1) == axs(s + 2) and his(s):
                    return False

                # Move filter #4
                if (
                    axs(s) == axs(s + 1)
                    and axs(s + 2) == axs(s + 3)
                    and axs(s) > axs(s + 3)
                ):
                    return False

    # Symmetric move filter #1
    for s in range(k - 1):
        if axs(s) == axs(s + 1) and his(s) and not his(s + 1):
            return False

    # Symmetric move filter #2
    for s in range(k - 1):
        for f in range(s + 1, min(s + 3, k)):
            if (
                axs(f) == axs(s)
                and his(f) == his(s)
                and all([axs(s) == axs(b) for b in range(s + 1, f)])
            ):
                return False

    if n == 3:
        for s in range(k - 3):
            if drs(s) == 2 and drs(s + 1) == 2 and drs(s + 2) == 2 and drs(s + 3) == 2:
                # Symmetric move filter #3
                if (
                    axs(s) == axs(s + 3)
                    and axs(s + 1) == axs(s + 2)
                    and axs(s) != axs(s + 1)
                ):
                    if not his(s + 3):
                        return False

                # Symmetric move filter #4
                if (
                    axs(s) == axs(s + 1)
                    and axs(s + 2) == axs(s + 3)
                    and axs(s + 1) != axs(s + 2)
                ):
                    if axs(s + 1) > axs(s + 2):
                        return False

    # Symmetric move filter #5
    if n == 3:
        for s in range(k - 4):
            if (
                (
                    (
                        axs(s) == axs(s + 1)
                        and axs(s + 2) == axs(s + 3)
                        and drs(s + 3) == 2
                        and (drs(s) == 2 or (drs(s + 1) == 2))
                    )
                    or (
                        axs(s) == axs(s + 3)
                        and axs(s + 1) == axs(s + 2)
                        and drs(s + 1) == 2
                        and (drs(s) == 2 or (drs(s + 3) == 2))
                    )
                )
                and (axs(s) != axs(s + 2))
                and drs(s + 2) == 2
            ):
                if axs(s) == axs(s + 4) and (
                    (
                        axs(s + 1) == axs(s + 4)
                        and ((drs(s) == 2) + (drs(s + 1) == 2) + (drs(s + 4) == 2)) >= 2
                    )
                    or (
                        axs(s + 3) == axs(s + 4)
                        and ((drs(s) == 2) + (drs(s + 3) == 2) + (drs(s + 4) == 2)) >= 2
                    )
                ):
                    return False

    # Symmetric move filter #6
    if n == 3:
        for s in range(k - 4):
            if (
                axs(s) == axs(s + 2)
                and axs(s + 1) == axs(s + 3)
                and axs(s + 3) == axs(s + 4)
                and drs(s) == drs(s + 2)
                and his(s) == his(s + 2)
                and drs(s) == 2
                and drs(s + 3) != 2
                and drs(s + 3) == drs(s + 4)
            ):
                return False

    # Symmetric move filter #7
    if n == 3:
        for s in range(k - 4):
            if (
                axs(s) == axs(s + 1)
                and axs(s + 1) == axs(s + 4)
                and (drs(s) == 2 or drs(s + 1) == 2 or drs(s + 4) == 2)
                and axs(s + 2) == axs(s + 3)
                and his(s + 2) != his(s + 3)
                and drs(s + 2) == drs(s + 3)
                and drs(s + 2) == 2
            ):
                if ((drs(s + 1) == 2 or drs(s + 4) == 2) and drs(s) == 1) or (
                    drs(s) == 2 and drs(s + 1) == 1
                ):
                    return False

    # Symmetric move filter #8
    if n == 3:
        for s in range(k - 4):
            if (
                axs(s + 1) == axs(s + 3)
                and drs(s + 1) == 2
                and drs(s + 1) == drs(s + 3)
                and his(s + 1) == his(s + 3)
            ):
                if axs(s) == axs(s + 2) and axs(s + 2) == axs(s + 4):
                    if his(s) == his(s + 4) and (
                        (drs(s) != drs(s + 4) and drs(s) != 2 and drs(s + 4) != 2)
                        or (drs(s) == 2 and drs(s + 4) == 2)
                    ):
                        if his(s) != 0:
                            return False

    return True


def generate(n: int, max_d: int):
    moves = all_moves()
    finished = Puzzle.finished(n, DEFAULT_CENTER_COLORS)
    paths: dict[Puzzle, MoveSeq] = {finished: tuple()}
    fresh: set[Puzzle] = {finished}

    for d in range(1, max_d + 1):
        next_fresh: set[Puzzle] = set()
        symmetries: dict[MoveSeq, set[MoveSeq]] = {}
        filtered: dict[Puzzle, set[MoveSeq]] = {}
        unique: set[MoveSeq] = set()

        for state in fresh:
            path = paths[state]
            for move in moves:
                new_path = path + (move,)

                new_state = state.execute_move(move)

                # Skip if the new path is not allowed by the filters.
                if not allowed_by_filters(n, new_path):
                    if new_state not in filtered:
                        filtered[new_state] = set()
                    filtered[new_state].add(new_path)
                    continue

                if new_state in paths:
                    prev_path = paths[new_state]
                    if prev_path not in symmetries:
                        symmetries[prev_path] = set()
                    symmetries[prev_path].add(new_path)

                    # Remove from unique, since it has now been observed more than once.
                    if prev_path in unique:
                        unique.remove(prev_path)

                else:
                    paths[new_state] = new_path
                    next_fresh.add(new_state)

                    # This state has not been seen before, so add to unique.
                    unique.add(new_path)

        # Check whether the filtered out move sequences' states are still reachable.
        for state, seqs in filtered.items():
            if state not in paths:
                canon = [move_names(seq) for seq in seqs]
                raise Exception(
                    f"following sequences should not all have been filtered:\n{canon}"
                )

        # Write found symmetrical move sequences to file.
        path = symmetries_file_path(n, d)
        create_parent_directory(path)
        output = [(k, sorted(v)) for k, v in symmetries.items()]
        output.sort(key=lambda x: (len(x[0]), len(x[1]), x[0], x[1]))
        with open(path, "w") as file:
            for seq, syms in output:
                seq_canon = move_names(seq)
                syms_canon = [move_names(seq) for seq in syms]
                file.write(f"{str(seq_canon)} -> {str(syms_canon)}\n")

        # Write found filtered move sequences to file.
        path = filtered_file_path(n, d)
        create_parent_directory(path)
        output = []
        for ftd in filtered.values():
            output.extend(ftd)
        output.sort()
        with open(path, "w") as file:
            for seq in output:
                seq_canon = move_names(seq)
                file.write(f"{str(seq_canon)}\n")

        # Write found unique move sequences to file.
        path = unique_file_path(n, d)
        create_parent_directory(path)
        output = list(unique)
        output.sort()
        with open(path, "w") as file:
            for seq in output:
                seq_canon = move_names(seq)
                file.write(f"{str(seq_canon)}\n")

        fil = sum(len(s) for s in filtered.values())
        pot = sum(len(s) for s in symmetries.values())
        print_stamped(f"d = {d}: filtered {fil}, with {pot} more filterable")
        fresh = next_fresh


def load_symmetries(
    n: int, d: int, include_lower: bool
) -> dict[MoveSeq, list[MoveSeq]]:
    if d <= 0:
        return {}

    path = symmetries_file_path(n, d)
    if not os.path.isfile(path):
        generate(n, d)

    result: dict[MoveSeq, list[MoveSeq]] = {}
    with open(path, "r") as file:
        for line in file:
            seq_raw, syms_raw = line.rstrip("\n").split(" -> ")
            seq_canon: tuple[str, ...] = ast.literal_eval(seq_raw)
            syms_canon: list[tuple[str, ...]] = ast.literal_eval(syms_raw)
            seq = tuple(parse_move(name) for name in seq_canon)
            syms = [tuple(parse_move(name) for name in sym) for sym in syms_canon]
            result[seq] = syms

    if include_lower:
        return result | load_symmetries(n, d - 1, include_lower)
    else:
        return result


def load_filtered(n: int, d: int, include_lower: bool) -> list[MoveSeq]:
    if d <= 0:
        return []

    path = filtered_file_path(n, d)
    if not os.path.isfile(path):
        generate(n, d)

    result: list[MoveSeq] = []
    with open(path, "r") as file:
        for line in file:
            seq_canon: tuple[str, ...] = ast.literal_eval(line.rstrip("\n"))
            seq = tuple(parse_move(name) for name in seq_canon)
            result.append(seq)

    if include_lower:
        return result + load_filtered(n, d - 1, include_lower)
    else:
        return result


def load_unique(n: int, d: int, include_lower: bool) -> list[MoveSeq]:
    if d <= 0:
        return []

    path = unique_file_path(n, d)
    if not os.path.isfile(path):
        generate(n, d)

    result: list[MoveSeq] = []
    with open(path, "r") as file:
        for line in file:
            seq_canon: tuple[str, ...] = ast.literal_eval(line.rstrip("\n"))
            seq = tuple(parse_move(name) for name in seq_canon)
            result.append(seq)

    if include_lower:
        return result + load_unique(n, d - 1, include_lower)
    else:
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    args = parser.parse_args()
    generate(args.n, args.d)
