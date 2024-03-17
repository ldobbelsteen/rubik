import argparse
import ast
import os

from generate_random import all_moves
from puzzle import DEFAULT_CENTER_COLORS, MoveSeq, Puzzle, move_name, parse_move
from tools import create_parent_directory


def file_path(n: int, d: int):
    return f"./move_symmetries/n{n}-d{d}.txt"


def compute(n: int, d: int):
    finished = Puzzle.finished(n, DEFAULT_CENTER_COLORS)
    moves = all_moves()

    # To keep track of the encountered states and which steps were taken to get there.
    encountered: dict[Puzzle, MoveSeq] = {finished: tuple()}

    # The new puzzle states encountered in the previous iteration.
    prev_layer: list[tuple[Puzzle, MoveSeq]] = [(finished, tuple())]

    # The symmetrical move sequences encountered in previous iterations combined.
    prev_symmetrical: set[MoveSeq] = set()

    def allowed(seq: MoveSeq) -> bool:
        """Return whether a move sequence is allowed according to the existing move
        filters. Also disallows symmetrical move sequences from lower depths."""

        k = len(seq)

        def axs(s: int):
            return seq[s][0]

        def his(s: int):
            return seq[s][1]

        def drs(s: int):
            return seq[s][2]

        for s in range(k - 1):
            if axs(s) == axs(s + 1) and his(s) and not his(s + 1):
                return False

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
                # dr == 2 everywhere
                if (
                    drs(s) == 2
                    and drs(s + 1) == 2
                    and drs(s + 2) == 2
                    and drs(s + 3) == 2
                ):
                    # axis pattern XYYX where X != Y
                    if (
                        axs(s) == axs(s + 3)
                        and axs(s + 1) == axs(s + 2)
                        and axs(s) != axs(s + 1)
                    ):
                        # if either (!hi[0] && !hi[1]) or (hi[0] && !hi[1]) then
                        # disallowed, since they are equivalent to (hi[0] && hi[1]) and
                        # (!hi[0] && hi[1]) respectively
                        if not his(s + 3):
                            return False

                    # axis pattern XXYY where X != Y
                    if (
                        axs(s) == axs(s + 1)
                        and axs(s + 2) == axs(s + 3)
                        and axs(s + 1) != axs(s + 2)
                    ):
                        # axis has to be increasing, so disallow
                        # this works since the axis are commutative in this case
                        if axs(s + 1) > axs(s + 2):
                            return False

        for sym in prev_symmetrical:
            start = k - len(sym)
            if start >= 0:
                if sym == seq[start:]:
                    return False

        return True

    # Perform BFS.
    for current_d in range(1, d + 1):
        new_layer: list[tuple[Puzzle, MoveSeq]] = []
        new_symmetrical: dict[MoveSeq, set[MoveSeq]] = {}

        # Execute all possible moves from the states encountered in the last iteration.
        for prev_puz, prev_seq in prev_layer:
            for move in moves:
                seq = prev_seq + (move,)
                if not allowed(seq):
                    continue  # ignore

                puz = prev_puz.copy()
                puz.execute_move(move)

                # If state has been seen, add it to symmetrical set.
                if puz in encountered:
                    enc_seq = encountered[puz]
                    assert len(enc_seq) <= len(seq)
                    if enc_seq not in new_symmetrical:
                        new_symmetrical[enc_seq] = set()
                    new_symmetrical[enc_seq].add(seq)
                # Else store as encountered and add it to the next iteration.
                else:
                    encountered[puz] = seq
                    new_layer.append((puz, seq))

        # Update layer and add newly found symmetrical sequences.
        prev_layer = new_layer
        for syms in new_symmetrical.values():
            for sym in syms:
                assert sym not in prev_symmetrical
                prev_symmetrical.add(sym)

        # Write found symmetrical sequences to file.
        output = [(k, sorted(v)) for k, v in new_symmetrical.items()]
        output.sort(key=lambda x: (len(x[0]), len(x[1])))
        path = file_path(n, current_d)
        create_parent_directory(path)
        with open(path, "w") as file:
            for seq, syms in output:
                seq_canon = tuple([move_name(s) for s in seq])
                syms_canon = [tuple([move_name(s) for s in seq]) for seq in syms]
                file.write(f"{str(seq_canon)} -> {str(syms_canon)}\n")


def load(n: int, d: int) -> dict[MoveSeq, list[MoveSeq]]:
    if d <= 0:
        return {}

    path = file_path(n, d)
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
