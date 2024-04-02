"""Script for finding symmetric move sequences."""

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
from tools import print_stamped

GENERATED_DIR = "./generated_moves"


def symmetries_file_path(n: int, d: int):
    """Return the file path for found symmetrical move sequences."""
    dir = os.path.join(GENERATED_DIR, "symmetrical")
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, f"n{n}-d{d}.txt")


def filtered_file_path(n: int, d: int):
    """Return the file path for found filtered move sequences."""
    dir = os.path.join(GENERATED_DIR, "filtered")
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, f"n{n}-d{d}.txt")


def unique_file_path(n: int, d: int):
    """Return the file path for found unique move sequences."""
    dir = os.path.join(GENERATED_DIR, "unique")
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, f"n{n}-d{d}.txt")


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
            if axs(s) == axs(s + 1):
                if not his(s + 1):
                    return False
                if his(s):
                    return False

        return True

    if n == 3:
        # Move filter #1 and #2
        for s in range(k - 1):
            if axs(s) == axs(s + 1):
                if not his(s):
                    return False
                if his(s + 1):
                    return False

        # Move filter #3 and #4
        for s in range(k - 3):
            if drs(s) == 2 and drs(s + 1) == 2 and drs(s + 2) == 2 and drs(s + 3) == 2:
                if axs(s) == axs(s + 3) and axs(s + 1) == axs(s + 2) and not his(s):
                    return False
                if (
                    axs(s) == axs(s + 1)
                    and axs(s + 1) > axs(s + 2)
                    and axs(s + 2) == axs(s + 3)
                ):
                    return False

        # # Manual move filter #5
        # for s in range(k - 4):
        #     if (
        #         axs(s) != axs(s + 2)
        #         and drs(s + 2) == 2
        #         and axs(s) == axs(s + 4)
        #         and (
        #             (
        #                 axs(s) == axs(s + 1)
        #                 and axs(s + 2) == axs(s + 3)
        #                 and drs(s + 3) == 2
        #                 and (drs(s) == 2 or (drs(s + 1) == 2))
        #                 and (drs(s) == 2 + drs(s + 1) == 2 + drs(s + 4) == 2) >= 2
        #             )
        #             or (
        #                 axs(s) == axs(s + 3)
        #                 and axs(s + 1) == axs(s + 2)
        #                 and drs(s + 1) == 2
        #                 and (drs(s) == 2 or (drs(s + 3) == 2))
        #                 and (drs(s) == 2 + drs(s + 3) == 2 + drs(s + 4) == 2) >= 2
        #             )
        #         )
        #     ):
        #         return False

        # # Manual move filter #6
        # for s in range(k - 4):
        #     if (
        #         axs(s) == axs(s + 2)
        #         and axs(s + 1) == axs(s + 3)
        #         and axs(s + 3) == axs(s + 4)
        #         and drs(s) == drs(s + 2)
        #         and his(s) == his(s + 2)
        #         and drs(s) == 2
        #         and drs(s + 3) != 2
        #         and drs(s + 3) == drs(s + 4)
        #     ):
        #         return False

        # # Manual move filter #7
        # for s in range(k - 4):
        #     if (
        #         axs(s) == axs(s + 1)
        #         and axs(s + 1) == axs(s + 4)
        #         and (drs(s) == 2 or drs(s + 1) == 2 or drs(s + 4) == 2)
        #         and axs(s + 2) == axs(s + 3)
        #         and his(s + 2) != his(s + 3)
        #         and drs(s + 2) == drs(s + 3)
        #         and drs(s + 2) == 2
        #     ):
        #         if ((drs(s + 1) == 2 or drs(s + 4) == 2) and drs(s) == 1) or (
        #             drs(s) == 2 and drs(s + 1) == 1
        #         ):
        #             return False

        # # Manual move filter #8
        # for s in range(k - 4):
        #     if (
        #         axs(s + 1) == axs(s + 3)
        #         and drs(s + 1) == 2
        #         and drs(s + 1) == drs(s + 3)
        #         and his(s + 1) == his(s + 3)
        #     ):
        #         if axs(s) == axs(s + 2) and axs(s + 2) == axs(s + 4):
        #             if his(s) == his(s + 4) and (
        #                 (drs(s) != drs(s + 4) and drs(s) != 2 and drs(s + 4) != 2)
        #                 or (drs(s) == 2 and drs(s + 4) == 2)
        #             ):
        #                 if his(s) != 0:
        #                     return False

        return True

    raise Exception("invalid n or misconfigured filters")


def generate(n: int, d: int):
    """Generate files containing found symmetrical, filtered and unique move sequences
    given n and a depth.
    """
    moves = all_moves()
    finished = Puzzle.finished(n, "???", DEFAULT_CENTER_COLORS)
    paths: dict[Puzzle, MoveSeq] = {finished: tuple()}
    fresh: set[Puzzle] = {finished}

    for cd in range(1, d + 1):
        next_fresh: set[Puzzle] = set()
        symmetries: dict[MoveSeq, set[MoveSeq]] = {}
        filtered: dict[Puzzle, set[MoveSeq]] = {}
        unique: set[MoveSeq] = set()

        for state in fresh:
            path = paths[state]
            for move in moves:
                new_path = (*path, move)

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
        symmetrical_output = [(k, sorted(v)) for k, v in symmetries.items()]
        symmetrical_output.sort(key=lambda x: (len(x[0]), len(x[1]), x[0], x[1]))
        with open(symmetries_file_path(n, cd), "w") as file:
            for seq, syms in symmetrical_output:
                seq_canon = move_names(seq)
                syms_canon = [move_names(seq) for seq in syms]
                file.write(f"{seq_canon!s} -> {syms_canon!s}\n")

        # Write found filtered move sequences to file.
        filtered_output = []
        for vs in filtered.values():
            filtered_output.extend(vs)
        filtered_output.sort()
        with open(filtered_file_path(n, cd), "w") as file:
            for seq in filtered_output:
                seq_canon = move_names(seq)
                file.write(f"{seq_canon!s}\n")

        # Write found unique move sequences to file.
        unique_output = list(unique)
        unique_output.sort()
        with open(unique_file_path(n, cd), "w") as file:
            for seq in unique_output:
                seq_canon = move_names(seq)
                file.write(f"{seq_canon!s}\n")

        fil = len(filtered_output)
        pot = sum(len(s) for _, s in symmetrical_output)
        print_stamped(f"d = {cd}: filtered {fil}, with {pot} more filterable")
        fresh = next_fresh


def load_symmetries(
    n: int, d: int, include_lower: bool
) -> dict[MoveSeq, list[MoveSeq]]:
    """Load found symmetrical move sequences from file. Optionally also loads
    from lower depths.
    """
    if d <= 0:
        return {}

    path = symmetries_file_path(n, d)
    if not os.path.isfile(path):
        generate(n, d)

    result: dict[MoveSeq, list[MoveSeq]] = {}
    with open(path) as file:
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
    """Load found filtered move sequences from file. Optionally also loads
    from lower depths.
    """
    if d <= 0:
        return []

    path = filtered_file_path(n, d)
    if not os.path.isfile(path):
        generate(n, d)

    result: list[MoveSeq] = []
    with open(path) as file:
        for line in file:
            seq_canon: tuple[str, ...] = ast.literal_eval(line.rstrip("\n"))
            seq = tuple(parse_move(name) for name in seq_canon)
            result.append(seq)

    if include_lower:
        return result + load_filtered(n, d - 1, include_lower)
    else:
        return result


def load_unique(n: int, d: int, include_lower: bool) -> list[MoveSeq]:
    """Load found unique move sequences from file. Optionally also loads
    from lower depths.
    """
    if d <= 0:
        return []

    path = unique_file_path(n, d)
    if not os.path.isfile(path):
        generate(n, d)

    result: list[MoveSeq] = []
    with open(path) as file:
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
