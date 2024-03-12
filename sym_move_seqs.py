import ast
import os
import sys

from generate import list_all_moves
from misc import create_parent_directory
from puzzle import Puzzle, default_k_upperbound

MoveSequence = tuple[tuple[int, int, int], ...]


def file_path(n: int, d: int):
    return f"./sym_move_seqs/n{n}-d{d}.txt"


def compute(n: int, d: int | None = None):
    if d is None:
        d = default_k_upperbound(n)

    lower = load_duplicate(n, d - 1)
    finished = Puzzle.finished(n)
    moves = list_all_moves(n)

    def is_allowed(n: int, seq: MoveSequence) -> bool:
        k = len(seq)
        mas = [m[0] for m in seq]
        mis = [m[1] for m in seq]
        mds = [m[2] for m in seq]

        # Same index and axis banned for n moves unless diff axis in between.
        for s in range(k - 1):
            for f in range(s + 1, min(s + n + 1, k)):
                if (
                    mas[f] == mas[s]
                    and mis[f] == mis[s]
                    and all([mas[s] == mas[b] for b in range(s + 1, f)])
                ):
                    return False

        # Ascending index in same axis.
        for s in range(k - 1):
            if mas[s] == mas[s + 1] and mis[s] >= mis[s + 1]:
                return False

        # Ascending axes for consecutive center half moves.
        if n == 3:
            for s in range(k - 1):
                if (
                    mis[s] == 1
                    and mis[s + 1] == 1
                    and mds[s] == 2
                    and mds[s + 1] == 2
                    and mas[s] >= mas[s + 1]
                ):
                    return False

        # Disallow symmetric move sequences from lower depths.
        for sym in lower:
            if any(
                [sym == seq[i : i + len(sym)] for i in range(len(seq) - len(sym) + 1)]
            ):
                return False

        return True

    # To keep track of the set of move sequences equal to a move sequence.
    duplicates: dict[MoveSequence, set[MoveSequence]] = {}

    # To keep track of the encountered states and which steps taken to get there.
    encountered: dict[Puzzle, MoveSequence] = {finished: tuple()}

    # The new puzzle states encountered in the last BFS iteration.
    layer: list[tuple[Puzzle, MoveSequence]] = [(finished, tuple())]

    # Perform BFS.
    for _ in range(d):
        next_layer: list[tuple[Puzzle, MoveSequence]] = []

        # Execute all possible moves from the states encountered in the last iteration.
        for puzzle, seq in layer:
            for ma, mi, md in moves:
                next_seq = seq + ((ma, mi, md),)
                if not is_allowed(n, next_seq):
                    continue  # ignore disallowed move sequences

                next_puzzle = puzzle.copy()
                next_puzzle.execute_move(ma, mi, md)

                # If state has been seen, add it to the set of duplicates.
                if next_puzzle in encountered:
                    encountered_seq = encountered[next_puzzle]
                    assert len(encountered_seq) <= len(next_seq)
                    if encountered_seq not in duplicates:
                        duplicates[encountered_seq] = set()
                    duplicates[encountered_seq].add(next_seq)
                # Else store as encountered and add it to the next iteration.
                else:
                    encountered[next_puzzle] = next_seq
                    next_layer.append((next_puzzle, next_seq))

        layer = next_layer

    output = list((k, list(v)) for k, v in duplicates.items())
    output.sort(key=lambda x: len(x[0]))

    path = file_path(n, d)
    create_parent_directory(path)
    with open(path, "w") as file:
        for seq, sym in output:
            file.write(f"{str(seq)}\t{str(sym)}\n")


def load(n: int, d: int | None = None) -> dict[MoveSequence, list[MoveSequence]]:
    if d is None:
        d = default_k_upperbound(n)
    if d == 0:
        return {}

    path = file_path(n, d)
    if not os.path.isfile(path):
        compute(n, d)

    result: dict[MoveSequence, list[MoveSequence]] = {}
    with open(path, "r") as file:
        for line in file:
            seq_raw, sym_raw = line.rstrip("\n").split("\t")
            seq: MoveSequence = ast.literal_eval(seq_raw)
            sym: list[MoveSequence] = ast.literal_eval(sym_raw)
            result[seq] = sym

    return result | load(n, d - 1)


def load_duplicate(n: int, d: int | None = None) -> list[MoveSequence]:
    if d is None:
        d = default_k_upperbound(n)
    if d == 0:
        return []

    path = file_path(n, d)
    if not os.path.isfile(path):
        compute(n, d)

    result: list[MoveSequence] = []
    with open(path, "r") as file:
        for line in file:
            _, sym_raw = line.rstrip("\n").split("\t")
            sym: list[MoveSequence] = ast.literal_eval(sym_raw)
            result += sym

    return result + load_duplicate(n, d - 1)


# e.g. python sym_move_seqs.py {n}
if __name__ == "__main__":
    compute(int(sys.argv[1]))
