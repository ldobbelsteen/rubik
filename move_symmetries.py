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
from tools import create_parent_directory


def file_path(n: int, d: int, filtered: bool):
    if filtered:
        return f"./move_symmetries/n{n}-d{d}.txt"
    else:
        return f"./move_symmetries/n{n}-d{d}-unfiltered.txt"


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

    return True


def symmetric_bfs_iteration(
    n: int,
    states: set[Puzzle],
    paths: dict[Puzzle, MoveSeq],
    disallowed: set[MoveSeq],
    filter: bool,
):
    moves = all_moves()
    new_states: set[Puzzle] = set()
    symmetries: dict[MoveSeq, set[MoveSeq]] = {}
    filtered: dict[Puzzle, set[MoveSeq]] = {}

    for state in states:
        path = paths[state]
        for move in moves:
            new_state = state.execute_move(move)

            new_path = path + (move,)
            if filter and not allowed_by_filters(n, new_path):
                if new_state not in filtered:
                    filtered[new_state] = set()
                filtered[new_state].add(new_path)
                continue  # ignore

            skip = False
            for sym in disallowed:
                start = len(new_path) - len(sym)
                if start >= 0 and sym == new_path[start:]:
                    skip = True  # tail is not allowed, so ignore
                    break
            if skip:
                continue

            if new_state in paths:
                prev_path = paths[new_state]
                assert len(prev_path) <= len(new_path)
                if prev_path not in symmetries:
                    symmetries[prev_path] = set()
                symmetries[prev_path].add(new_path)
            else:
                paths[new_state] = new_path
                new_states.add(new_state)

    for syms in symmetries.values():
        for sym in syms:
            assert sym not in disallowed
            disallowed.add(sym)

    return new_states, symmetries, filtered


def compute(n: int, max_d: int):
    finished = Puzzle.finished(n, DEFAULT_CENTER_COLORS)

    # To keep track of the encountered states and which steps were taken to get there.
    paths_unfiltered: dict[Puzzle, MoveSeq] = {finished: tuple()}
    paths_filtered: dict[Puzzle, MoveSeq] = {finished: tuple()}

    # The new puzzle states encountered in the previous iteration.
    fresh_unfiltered: set[Puzzle] = {finished}
    fresh_filtered: set[Puzzle] = {finished}

    # The symmetrical move sequences encountered in previous iterations combined.
    disallowed_unfiltered: set[MoveSeq] = set()
    disallowed_filtered: set[MoveSeq] = set()

    # Perform BFS for both unfiltered and filtered.
    for d in range(1, max_d + 1):
        fresh_unfiltered, symmetries_unfiltered, _ = symmetric_bfs_iteration(
            n,
            fresh_unfiltered,
            paths_unfiltered,
            disallowed_unfiltered,
            False,
        )

        fresh_filtered, symmetries_filtered, filtered = symmetric_bfs_iteration(
            n,
            fresh_filtered,
            paths_filtered,
            disallowed_filtered,
            True,
        )

        # This asserts whether we don't filter too many symmetric moves such that
        # some puzzle states that are reachable when not filtering cannot be reached
        # when filtering.
        if fresh_unfiltered != fresh_filtered:
            for state in fresh_unfiltered - fresh_filtered:
                canon = [tuple([move_name(m) for m in seq]) for seq in filtered[state]]
                raise Exception(
                    f"following sequences should not all have been filtered:\n{canon}"
                )
            for state in fresh_filtered - fresh_unfiltered:
                raise Exception(
                    f"state reachable only when filtering: {paths_filtered[state]}"
                )

        # Write found filtered symmetric move sequences to file.
        path = file_path(n, d, True)
        create_parent_directory(path)
        filtered = [(k, sorted(v)) for k, v in symmetries_filtered.items()]
        filtered.sort(key=lambda x: (len(x[0]), len(x[1]), x[0], x[1]))
        with open(path, "w") as file:
            for seq, syms in filtered:
                seq_canon = tuple([move_name(s) for s in seq])
                syms_canon = [tuple([move_name(s) for s in seq]) for seq in syms]
                file.write(f"{str(seq_canon)} -> {str(syms_canon)}\n")

        # Write the unfiltered symmetric move sequences to file.
        path = file_path(n, d, False)
        create_parent_directory(path)
        unfiltered = [(k, sorted(v)) for k, v in symmetries_unfiltered.items()]
        unfiltered.sort(key=lambda x: (len(x[0]), len(x[1]), x[0], x[1]))
        with open(path, "w") as file:
            for seq, syms in unfiltered:
                seq_canon = tuple([move_name(s) for s in seq])
                syms_canon = [tuple([move_name(s) for s in seq]) for seq in syms]
                file.write(f"{str(seq_canon)} -> {str(syms_canon)}\n")


def load(n: int, d: int) -> dict[MoveSeq, list[MoveSeq]]:
    if d <= 0:
        return {}

    path = file_path(n, d, True)
    if not os.path.isfile(path):
        compute(n, d)

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
    compute(args.n, args.d)
