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
                drs(s + 1) == 2
                and drs(s + 2) == 2
                and drs(s + 3) == 2
                and axs(s) == axs(s + 1)
                and axs(s + 2) == axs(s + 3)
                and axs(s + 1) != axs(s + 2)
            ):
                if (
                    drs(s + 4) == 2
                    and axs(s + 4) == axs(s + 1)
                    and his(s + 4) == his(s + 1)
                ):
                    return False

    # Symmetric move filter #6
    if n == 3:
        for s in range(k - 4):
            if (
                drs(s) == 2
                and drs(s + 1) == 2
                and drs(s + 2) == 2
                and axs(s) == axs(s + 3)
                and axs(s + 1) == axs(s + 2)
                and axs(s) != axs(s + 1)
                and his(s) != his(s + 3)
            ):
                if drs(s + 4) == 2 and axs(s + 4) == axs(s) and his(s + 4) == his(s):
                    return False

    # # Symmetric move filter #7
    # if n == 3:
    #     for s in range(k - 4):
    #         if (
    #             (axs(s) == axs(s + 1) or axs(s) == axs(s + 3))
    #             and (axs(s) != axs(s + 2))
    #             and (axs(s + 1) == axs(s + 2) or axs(s + 2) == axs(s + 3))
    #             and drs(s) == 2
    #             and drs(s + 2) == 2
    #             and (drs(s + 1) == 2 or drs(s + 3) == 2)
    #         ):
    #             if axs(s) == axs(s + 4):
    #                 return False

    return True


def symmetric_bfs_iteration(
    n: int,
    encountered: dict[Puzzle, MoveSeq],
    prev_symmetrical: set[MoveSeq],
    prev_layer: set[Puzzle],
    filtered: bool,
) -> tuple[set[Puzzle], dict[MoveSeq, set[MoveSeq]]]:
    moves = all_moves()
    new_layer: set[Puzzle] = set()
    new_symmetrical: dict[MoveSeq, set[MoveSeq]] = {}

    for prev_puz in prev_layer:
        prev_seq = encountered[prev_puz]
        for move in moves:
            seq = prev_seq + (move,)
            if filtered and not allowed_by_filters(n, seq):
                continue  # ignore

            skip = False
            for sym in prev_symmetrical:
                start = len(seq) - len(sym)
                if start >= 0 and sym == seq[start:]:
                    skip = True  # tail is not allowed, so ignore
                    break
            if skip:
                continue

            puz = prev_puz.execute_move(move)

            if puz in encountered:
                enc_seq = encountered[puz]
                assert len(enc_seq) <= len(seq)
                if enc_seq not in new_symmetrical:
                    new_symmetrical[enc_seq] = set()
                new_symmetrical[enc_seq].add(seq)
            else:
                encountered[puz] = seq
                new_layer.add(puz)

    for syms in new_symmetrical.values():
        for sym in syms:
            assert sym not in prev_symmetrical
            prev_symmetrical.add(sym)

    return new_layer, new_symmetrical


def compute(n: int, max_d: int):
    finished = Puzzle.finished(n, DEFAULT_CENTER_COLORS)

    # To keep track of the encountered states and which steps were taken to get there.
    encountered_unfiltered: dict[Puzzle, MoveSeq] = {finished: tuple()}
    encountered_filtered: dict[Puzzle, MoveSeq] = {finished: tuple()}

    # The new puzzle states encountered in the previous iteration.
    layer_unfiltered: set[Puzzle] = {finished}
    layer_filtered: set[Puzzle] = {finished}

    # The symmetrical move sequences encountered in previous iterations combined.
    symmetrical_unfiltered: set[MoveSeq] = set()
    symmetrical_filtered: set[MoveSeq] = set()

    # Perform BFS for both unfiltered and filtered.
    for d in range(1, max_d + 1):
        layer_unfiltered, new_symmetrical_unfiltered = symmetric_bfs_iteration(
            n,
            encountered_unfiltered,
            symmetrical_unfiltered,
            layer_unfiltered,
            False,
        )

        layer_filtered, new_symmetrical_filtered = symmetric_bfs_iteration(
            n,
            encountered_filtered,
            symmetrical_filtered,
            layer_filtered,
            True,
        )

        # This asserts whether we don't filter too many symmetric moves such that
        # some puzzle states that are reachable when not filtering cannot be reached
        # when filtering.
        if layer_unfiltered != layer_filtered:
            for state in layer_unfiltered - layer_filtered:
                m = encountered_unfiltered[state]
                seqs = (
                    {m}
                    if m not in new_symmetrical_unfiltered
                    else new_symmetrical_unfiltered[m] | {m}
                )
                seqs_canon = [tuple([move_name(m) for m in seq]) for seq in seqs]
                raise Exception(
                    f"erroneously filtered one of these sequences:\n{seqs_canon}"
                )
        for state in layer_filtered - layer_unfiltered:
            m = encountered_filtered[state]
            seqs = (
                {m}
                if m not in new_symmetrical_filtered
                else new_symmetrical_filtered[m] | {m}
            )
            seqs_canon = [tuple([move_name(m) for m in seq]) for seq in seqs]
            raise Exception(
                f"state reachable when filtering not reachable when not:\n{seqs_canon}"
            )

        # Write found filtered symmetric move sequences to file.
        path = file_path(n, d, True)
        create_parent_directory(path)
        filtered = [(k, sorted(v)) for k, v in new_symmetrical_filtered.items()]
        filtered.sort(key=lambda x: (len(x[0]), len(x[1]), x[0], x[1]))
        with open(path, "w") as file:
            for seq, syms in filtered:
                seq_canon = tuple([move_name(s) for s in seq])
                syms_canon = [tuple([move_name(s) for s in seq]) for seq in syms]
                file.write(f"{str(seq_canon)} -> {str(syms_canon)}\n")

        # Write the unfiltered symmetric move sequences to file.
        path = file_path(n, d, False)
        create_parent_directory(path)
        unfiltered = [(k, sorted(v)) for k, v in new_symmetrical_unfiltered.items()]
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
