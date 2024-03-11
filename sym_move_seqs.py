import ast
import os
import sys

from generate import list_all_moves
from misc import create_parent_directory
from puzzle import Puzzle

MoveSequence = tuple[tuple[int, int, int], ...]


def is_allowed(n: int, seq: MoveSequence) -> bool:
    # Same index and axis banned for n moves unless diff axis in between.
    for s in range(len(seq) - 1):
        for f in range(s + 1, min(s + n + 1, len(seq))):
            if (
                seq[f][0] == seq[s][0]
                and seq[f][1] == seq[s][1]
                and all([seq[s][0] == seq[b][0] for b in range(s + 1, f)])
            ):
                return False

    # Ascending index.
    for s in range(len(seq) - 1):
        if seq[s][0] == seq[s + 1][0] and seq[s][1] >= seq[s + 1][1]:
            return False

    # Ascending axes for consecutive center moves.
    if n == 3:
        for s in range(len(seq) - 1):
            if (
                seq[s][1] == 1
                and seq[s + 1][1] == 1
                and seq[s][2] == 2
                and seq[s + 1][2] == 2
                and seq[s][0] >= seq[s + 1][0]
            ):
                return False

    return True


def file_path(n: int, d: int):
    return f"./sym_move_seqs/n{n}-d{d}.txt"


def compute_symmetric_move_sequences(n: int, d: int, overwrite=False):
    path = file_path(n, d)
    if not overwrite and os.path.isfile(path):
        return
    create_parent_directory(path)

    finished = Puzzle.finished(n)
    moves = list_all_moves(n)

    # To keep track of the set of move sequences equal to a move sequence.
    duplicates: dict[MoveSequence, list[MoveSequence]] = {}

    # To keep track of the encountered states and which steps taken to get there.
    encountered: dict[Puzzle, MoveSequence] = {finished: tuple()}

    # The new puzzle states encountered in the last BFS iteration.
    layer: list[tuple[Puzzle, MoveSequence]] = [(finished, tuple())]

    # Perform BFS.
    for _ in range(1, d + 1):
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
                    encountered_combination = encountered[next_puzzle]
                    assert len(encountered_combination) <= len(next_seq)
                    if encountered_combination not in duplicates:
                        duplicates[encountered_combination] = []
                    duplicates[encountered_combination].append(next_seq)
                # Else store as encountered and add it to the next iteration.
                else:
                    encountered[next_puzzle] = next_seq
                    next_layer.append((next_puzzle, next_seq))

        layer = next_layer

    output = [(list(seq), [list(s) for s in sym]) for seq, sym in duplicates.items()]
    output.sort(key=lambda x: len(x[0]))

    with open(path, "w") as file:
        for seq, sym in output:
            file.write(f"{str(seq)}\t{str(sym)}\n")


def load_symmetric_move_sequences(n: int, d: int):
    output: list[
        tuple[list[tuple[int, int, int]], list[list[tuple[int, int, int]]]]
    ] = []
    with open(file_path(n, d), "r") as file:
        for line in file:
            seq_raw, sym_raw = line.rstrip("\n").split("\t")
            output.append((ast.literal_eval(seq_raw), ast.literal_eval(sym_raw)))
    return output


# e.g. python sym_move_seqs.py {n} {depth}
if __name__ == "__main__":
    compute_symmetric_move_sequences(int(sys.argv[1]), int(sys.argv[2]), True)
