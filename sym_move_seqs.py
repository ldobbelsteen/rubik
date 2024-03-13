import argparse
import ast
import os

from generate import list_all_moves
from misc import create_parent_directory
from puzzle import Puzzle

Move = tuple[int, int, int]
MoveSequence = tuple[Move, ...]


def file_path(n: int, d: int):
    return f"./sym_move_seqs/n{n}-d{d}.txt"


def compute(n: int, d: int):
    finished = Puzzle.finished(n)
    moves = list_all_moves(n)

    # To keep track of the encountered states and which steps taken to get there.
    encountered: dict[Puzzle, MoveSequence] = {finished: tuple()}

    # The new puzzle states encountered in the previous iteration.
    prev_layer: list[tuple[Puzzle, MoveSequence]] = [(finished, tuple())]

    # The symmetrical move sequences encountered in previous iterations combined.
    prev_symmetrical: set[MoveSequence] = set()

    def extension_allowed(n: int, ext_seq: MoveSequence) -> bool:
        k = len(ext_seq)

        def mas(s):
            return ext_seq[s][0]

        def mis(s):
            return ext_seq[s][1]

        def mds(s):
            return ext_seq[s][2]

        # Same index and axis banned for n moves unless diff axis in between.
        for s in range(k - 1):
            for f in range(s + 1, min(s + n + 1, k)):
                if (
                    mas(f) == mas(s)
                    and mis(f) == mis(s)
                    and all([mas(s) == mas(b) for b in range(s + 1, f)])
                ):
                    return False

        # Ascending index in same axis.
        for s in range(k - 1):
            if mas(s) == mas(s + 1) and mis(s) >= mis(s + 1):
                return False

        # Ascending axes for consecutive center half moves.
        if n == 3:
            for s in range(k - 1):
                if (
                    mis(s) == 1
                    and mis(s + 1) == 1
                    and mds(s) == 2
                    and mds(s + 1) == 2
                    and mas(s) >= mas(s + 1)
                ):
                    return False

        # # NOTE: test 1
        # if n == 3:
        #     for s in range(k - 2):
        #         if ext_seq[s] == (0, 1, 2):
        #             if mis(s + 1) == 1 and mds(s + 1) == 2:
        #                 if mas(s + 2) == 0 and mis(s + 2) == 0:
        #                     return False
        #                 if mis(s + 2) == 1 and mds(s + 2) != 2:
        #                     return False

        # # NOTE: test 2
        # if n == 3:
        #     for s in range(k - 2):
        #         if ext_seq[s] == (1, 1, 2):
        #             if (
        #                 (mas(s + 1) == 0 and mis(s + 1) == 0)
        #                 or (mas(s + 1) == 1 and mis(s + 1) == 2)
        #                 or (mas(s + 1) == 2 and mis(s + 1) == 1 and mds(s + 1) != 2)
        #             ):
        #                 if ext_seq[s + 2] == (0, 1, 2):
        #                     return False

        # # NOTE: test 3
        # if n == 3:
        #     for s in range(k - 2):
        #         if ext_seq[s] == (1, 1, 2):
        #             if ext_seq[s + 1] == (2, 1, 2):
        #                 if mas(s + 1) == 1 and mis(s + 2) == 0:
        #                     return False
        #                 if mas(s + 2) != 2 and mis(s + 2) == 1 and mds(s + 2) != 2:
        #                     return False

        # # NOTE: test 4
        # if n == 3:
        #     for s in range(k - 2):
        #         if mas(s) == 2 and mis(s) == 1 and mds(s) != 2:
        #             if mas(s + 1) != 2 and mis(s + 1) == 1 and mds(s + 1) == 2:
        #                 if ext_seq[s + 2] == (2, 1, 2):
        #                     return False

        # # NOTE: test 5
        # if n == 3:
        #     for s in range(k - 2):
        #         if mas(s) == 1 and mis(s) == 1 and mds(s) != 2:
        #             if ext_seq[s + 1] == (0, 1, 2):
        #                 if ext_seq[s + 2] == (1, 1, 2):
        #                     return False

        # Disallow symmetric move sequences from lower depths.
        for sym in prev_symmetrical:
            start = k - len(sym)
            if start >= 0:
                if sym == ext_seq[start:]:
                    return False

        return True

    # Perform BFS.
    for current_d in range(1, d + 1):
        new_layer: list[tuple[Puzzle, MoveSequence]] = []
        new_symmetrical: dict[MoveSequence, set[MoveSequence]] = {}

        # Execute all possible moves from the states encountered in the last iteration.
        for prev_puz, prev_seq in prev_layer:
            for ma, mi, md in moves:
                seq = prev_seq + ((ma, mi, md),)
                if not extension_allowed(n, seq):
                    continue  # ignore if not allowed

                puz = prev_puz.copy()
                puz.execute_move(ma, mi, md)

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
                file.write(f"{str(seq)}\t{str(syms)}\n")


def load(n: int, d: int) -> dict[MoveSequence, list[MoveSequence]]:
    if d <= 0:
        return {}

    path = file_path(n, d)
    if not os.path.isfile(path):
        compute(n, d)

    result: dict[MoveSequence, list[MoveSequence]] = {}
    with open(path, "r") as file:
        for line in file:
            seq_raw, syms_raw = line.rstrip("\n").split("\t")
            result[ast.literal_eval(seq_raw)] = ast.literal_eval(syms_raw)

    return result | load(n, d - 1)


# e.g. python sym_move_seqs.py {n} {d}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("d", type=int)
    args = parser.parse_args()
    compute(args.n, args.d)
