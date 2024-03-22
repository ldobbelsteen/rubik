import argparse
import ast
import os

from puzzle import (
    DEFAULT_CENTER_COLORS,
    MoveSeq,
    Puzzle,
    all_moves,
    move_name,
    parse_move,
)
from tools import create_parent_directory, print_stamped


def file_path(n: int, d: int):
    dir = os.path.dirname(__file__)
    return os.path.join(dir, f"/generated_move_symmetries/n{n}-d{d}.txt")


def allowed_by_filters(n: int, seq: MoveSeq) -> bool:
    """Return whether a move sequence is allowed by applying filters."""
    k = len(seq)

    def axs(s: int):
        return seq[s][0]

    def his(s: int):
        return seq[s][1]

    def drs(s: int):
        return seq[s][2]

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

    return True


def generate(n: int, max_d: int):
    moves = all_moves()
    finished = Puzzle.finished(n, DEFAULT_CENTER_COLORS)
    paths: dict[Puzzle, MoveSeq] = {finished: tuple()}
    fresh: set[Puzzle] = {finished}
    banned: set[MoveSeq] = set()

    for d in range(1, max_d + 1):
        next_fresh: set[Puzzle] = set()
        filtered: dict[Puzzle, set[MoveSeq]] = {}
        symmetries: dict[MoveSeq, set[MoveSeq]] = {}

        for state in fresh:
            path = paths[state]
            for move in moves:
                new_path = path + (move,)

                # Skip if the new path contains symmetric move sequences
                # from lower depths.
                skip = False
                for ban in banned:
                    start = len(new_path) - len(ban)
                    if start >= 0 and ban == new_path[start:]:
                        skip = True
                        break
                if skip:
                    continue

                new_state = state.execute_move(move)

                # Skip if the new path is not allowed by the filters. Store
                # the filtered move sequence for later.
                if not allowed_by_filters(n, new_path):
                    if new_state not in filtered:
                        filtered[new_state] = set()
                    filtered[new_state].add(new_path)
                    banned.add(new_path)
                    continue

                if new_state in paths:
                    prev_path = paths[new_state]
                    assert len(prev_path) <= len(new_path)
                    if prev_path not in symmetries:
                        symmetries[prev_path] = set()
                    symmetries[prev_path].add(new_path)
                else:
                    paths[new_state] = new_path
                    next_fresh.add(new_state)

        # Check whether the filtered out move sequences' states are still reachable.
        for state, seqs in filtered.items():
            if state not in paths:
                canon = [tuple([move_name(m) for m in seq]) for seq in seqs]
                raise Exception(
                    f"following sequences should not all have been filtered:\n{canon}"
                )

        # Ban all symmetric move sequences for future iterations.
        for syms in symmetries.values():
            for sym in syms:
                assert sym not in banned
                banned.add(sym)

        # Write found symmetric move sequences to file.
        path = file_path(n, d)
        create_parent_directory(path)
        output = [(k, sorted(v)) for k, v in symmetries.items()]
        output.sort(key=lambda x: (len(x[0]), len(x[1]), x[0], x[1]))
        with open(path, "w") as file:
            for seq, syms in output:
                seq_canon = tuple([move_name(s) for s in seq])
                syms_canon = [tuple([move_name(s) for s in seq]) for seq in syms]
                file.write(f"{str(seq_canon)} -> {str(syms_canon)}\n")

        fil = sum(len(s) for s in filtered.values())
        pot = sum(len(s) for s in symmetries.values())
        print_stamped(f"d = {d}: filtered {fil}, with {pot} more filterable")
        fresh = next_fresh


def load(n: int, d: int) -> dict[MoveSeq, list[MoveSeq]]:
    if d <= 0:
        return {}

    path = file_path(n, d)
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

    return result | load(n, d - 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    args = parser.parse_args()
    generate(args.n, args.d)
