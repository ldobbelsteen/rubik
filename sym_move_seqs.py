import os
import sys

from generate import moveset
from misc import create_parent_directory, print_stamped
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

    return True


def compute_duplicate_move_sequences(n: int, depth: int, overwrite=False):
    path = f"./sym_move_seqs/n{n}-d{depth}.txt"
    if not overwrite and os.path.isfile(path):
        return
    create_parent_directory(path)

    finished = Puzzle.finished(n)
    moves = moveset(n)

    # To keep track of the set of move sequences equal to a move sequence.
    duplicates: dict[MoveSequence, set[MoveSequence]] = {}

    # To keep track of the encountered states and which steps taken to get there.
    encountered: dict[Puzzle, MoveSequence] = {finished: tuple()}

    # The new puzzle states encountered in the last BFS iteration.
    layer: list[tuple[Puzzle, MoveSequence]] = [(finished, tuple())]

    # Perform BFS.
    for d in range(1, depth + 1):
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
                        duplicates[encountered_combination] = set()
                    duplicates[encountered_combination].add(next_seq)
                # Else store as encountered and add it to the next iteration.
                else:
                    encountered[next_puzzle] = next_seq
                    next_layer.append((next_puzzle, next_seq))

        layer = next_layer

    with open(path, "w") as file:
        for original, symmetrical in duplicates.items():
            original = str(list(original))
            symmetrical = str([list(seq) for seq in symmetrical])
            file.write(f"{original}\t{symmetrical}\n")


# e.g. python sym_move_seqs.py {n} {max_depth}
if __name__ == "__main__":
    n = int(sys.argv[1])
    max_depth = int(sys.argv[2])

    for d in range(1, max_depth + 1):
        print_stamped(f"computing for depth = {d}...")
        compute_duplicate_move_sequences(n, d, True)
