import argparse
import os

from puzzle import DEFAULT_CENTER_COLORS, Puzzle
from state import Move, MoveSeq
from tools import print_stamped

MOVE_SYMMETRY_RESULTS = "./move_symmetry_results"


def symmetries_file_path(n: int, d: int):
    """Return the file path for found symmetrical move sequences."""
    dir = os.path.join(MOVE_SYMMETRY_RESULTS, "symmetrical")
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, f"n{n}-d{d}.txt")


def filtered_file_path(n: int, d: int):
    """Return the file path for found filtered move sequences."""
    dir = os.path.join(MOVE_SYMMETRY_RESULTS, "filtered")
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, f"n{n}-d{d}.txt")


def unique_file_path(n: int, d: int):
    """Return the file path for found unique move sequences."""
    dir = os.path.join(MOVE_SYMMETRY_RESULTS, "unique")
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, f"n{n}-d{d}.txt")


def allowed_by_filters(n: int, seq: MoveSeq) -> bool:
    """Return whether a move sequence is allowed by applying filters."""
    k = len(seq)

    if n == 2:
        # Move filter #1 and #2
        for s in range(k - 1):
            if seq.ax(s) == seq.ax(s + 1):
                if not seq.hi(s + 1):
                    return False
                if seq.hi(s):
                    return False

        return True

    if n == 3:
        # Move filter #1 and #2
        for s in range(k - 1):
            if seq.ax(s) == seq.ax(s + 1):
                if not seq.hi(s):
                    return False
                if seq.hi(s + 1):
                    return False

        # Move filter #3 and #4
        for s in range(k - 3):
            if (
                seq.dr(s) == 2
                and seq.dr(s + 1) == 2
                and seq.dr(s + 2) == 2
                and seq.dr(s + 3) == 2
            ):
                if (
                    seq.ax(s) == seq.ax(s + 3)
                    and seq.ax(s + 1) == seq.ax(s + 2)
                    and not seq.hi(s)
                ):
                    return False
                if (
                    seq.ax(s) == seq.ax(s + 1)
                    and seq.ax(s + 1) > seq.ax(s + 2)
                    and seq.ax(s + 2) == seq.ax(s + 3)
                ):
                    return False

        # # Manual move filter #5
        # for s in range(k - 4):
        #     if (
        #         seq.ax(s) != seq.ax(s + 2)
        #         and seq.dr(s + 2) == 2
        #         and seq.ax(s) == seq.ax(s + 4)
        #         and (
        #             (
        #                 seq.ax(s) == seq.ax(s + 1)
        #                 and seq.ax(s + 2) == seq.ax(s + 3)
        #                 and seq.dr(s + 3) == 2
        #                 and (seq.dr(s) == 2 or (seq.dr(s + 1) == 2))
        #                 and (seq.dr(s) == 2 + seq.dr(s + 1) == 2 + seq.dr(s + 4) == 2)
        #                 >= 2
        #             )
        #             or (
        #                 seq.ax(s) == seq.ax(s + 3)
        #                 and seq.ax(s + 1) == seq.ax(s + 2)
        #                 and seq.dr(s + 1) == 2
        #                 and (seq.dr(s) == 2 or (seq.dr(s + 3) == 2))
        #                 and (seq.dr(s) == 2 + seq.dr(s + 3) == 2 + seq.dr(s + 4) == 2)
        #                 >= 2
        #             )
        #         )
        #     ):
        #         return False

        # # Manual move filter #6
        # for s in range(k - 4):
        #     if (
        #         seq.ax(s) == seq.ax(s + 2)
        #         and seq.ax(s + 1) == seq.ax(s + 3)
        #         and seq.ax(s + 3) == seq.ax(s + 4)
        #         and seq.dr(s) == seq.dr(s + 2)
        #         and seq.hi(s) == seq.hi(s + 2)
        #         and seq.dr(s) == 2
        #         and seq.dr(s + 3) != 2
        #         and seq.dr(s + 3) == seq.dr(s + 4)
        #     ):
        #         return False

        # # Manual move filter #7
        # for s in range(k - 4):
        #     if (
        #         seq.ax(s) == seq.ax(s + 1)
        #         and seq.ax(s + 1) == seq.ax(s + 4)
        #         and (seq.dr(s) == 2 or seq.dr(s + 1) == 2 or seq.dr(s + 4) == 2)
        #         and seq.ax(s + 2) == seq.ax(s + 3)
        #         and seq.hi(s + 2) != seq.hi(s + 3)
        #         and seq.dr(s + 2) == seq.dr(s + 3)
        #         and seq.dr(s + 2) == 2
        #     ):
        #        if ((seq.dr(s + 1) == 2 or seq.dr(s + 4) == 2) and seq.dr(s) == 1) or (
        #             seq.dr(s) == 2 and seq.dr(s + 1) == 1
        #         ):
        #             return False

        # # Manual move filter #8
        # for s in range(k - 4):
        #     if (
        #         seq.ax(s + 1) == seq.ax(s + 3)
        #         and seq.dr(s + 1) == 2
        #         and seq.dr(s + 1) == seq.dr(s + 3)
        #         and seq.hi(s + 1) == seq.hi(s + 3)
        #     ):
        #         if seq.ax(s) == seq.ax(s + 2) and seq.ax(s + 2) == seq.ax(s + 4):
        #             if seq.hi(s) == seq.hi(s + 4) and (
        #                 (
        #                     seq.dr(s) != seq.dr(s + 4)
        #                     and seq.dr(s) != 2
        #                     and seq.dr(s + 4) != 2
        #                 )
        #                 or (seq.dr(s) == 2 and seq.dr(s + 4) == 2)
        #             ):
        #                 if seq.hi(s) != 0:
        #                     return False

        return True

    raise Exception("invalid n or misconfigured filters")


def generate_move_symmetries(n: int, d: int):
    """Generate files containing found symmetrical, filtered and unique move sequences
    given n and a depth.
    """
    moves = Move.list_all()
    finished = Puzzle.finished(n, "???", DEFAULT_CENTER_COLORS)
    paths: dict[Puzzle, MoveSeq] = {finished: MoveSeq(())}
    fresh: set[Puzzle] = {finished}

    for cd in range(1, d + 1):
        next_fresh: set[Puzzle] = set()
        symmetries: dict[MoveSeq, set[MoveSeq]] = {}
        filtered: dict[Puzzle, set[MoveSeq]] = {}
        unique: set[MoveSeq] = set()

        for state in fresh:
            path = paths[state]
            for move in moves:
                new_path = path.extended(move)
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
                seqs_str = [str(seq) for seq in seqs]
                raise Exception(
                    f"following sequences should not all be filtered:\n{seqs_str}"
                )

        # Write found symmetrical move sequences to file.
        symmetrical_output = [(k, sorted(v)) for k, v in symmetries.items()]
        symmetrical_output.sort(key=lambda x: (len(x[0]), len(x[1]), x[0], x[1]))
        with open(symmetries_file_path(n, cd), "w") as file:
            for seq, syms in symmetrical_output:
                seq_str = str(seq)
                syms_str = ", ".join(map(str, syms))
                file.write(f"{seq_str} -> {syms_str}\n")

        # Write found filtered move sequences to file.
        filtered_output: list[MoveSeq] = []
        for vs in filtered.values():
            filtered_output.extend(vs)
        filtered_output.sort()
        with open(filtered_file_path(n, cd), "w") as file:
            for seq in filtered_output:
                seq_str = str(seq)
                file.write(f"{seq_str}\n")

        # Write found unique move sequences to file.
        unique_output = list(unique)
        unique_output.sort()
        with open(unique_file_path(n, cd), "w") as file:
            for seq in unique_output:
                seq_str = str(seq)
                file.write(f"{seq_str}\n")

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
        generate_move_symmetries(n, d)

    result: dict[MoveSeq, list[MoveSeq]] = {}
    with open(path) as file:
        for line in file:
            seq_raw, syms_raw = line.rstrip("\n").split(" -> ")
            seq = MoveSeq.from_str(seq_raw)
            syms = [MoveSeq.from_str(sym) for sym in syms_raw.split(", ")]
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
        generate_move_symmetries(n, d)

    result: list[MoveSeq] = []
    with open(path) as file:
        for line in file:
            seq_raw = line.rstrip("\n")
            seq = MoveSeq.from_str(seq_raw)
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
        generate_move_symmetries(n, d)

    result: list[MoveSeq] = []
    with open(path) as file:
        for line in file:
            seq_raw = line.rstrip("\n")
            seq = MoveSeq.from_str(seq_raw)
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
    generate_move_symmetries(args.n, args.d)
