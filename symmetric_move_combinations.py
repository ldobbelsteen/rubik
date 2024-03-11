import sys

from generate import moveset
from misc import print_stamped
from puzzle import Puzzle

MoveCombination = tuple[tuple[int, int, int], ...]


def is_allowed(n: int, mc: MoveCombination) -> bool:
    # Same index and axis banned for n moves unless diff axis in between.
    for s in range(len(mc) - 1):
        for f in range(s + 1, min(s + n + 1, len(mc))):
            if (
                mc[f][0] == mc[s][0]
                and mc[f][1] == mc[s][1]
                and all([mc[s][0] == mc[b][0] for b in range(s + 1, f)])
            ):
                return False

    # Ascending index heuristic.
    for s in range(len(mc) - 1):
        if mc[s][0] == mc[s + 1][0] and mc[s][1] >= mc[s + 1][1]:
            return False

    return True


# e.g. python symmetric_move_combinations.py {n} {max_depth}
if __name__ == "__main__":
    n = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    finished = Puzzle.finished(n)
    moves = moveset(n)

    duplicates: dict[MoveCombination, set[MoveCombination]] = {}
    encountered: dict[Puzzle, MoveCombination] = {finished: tuple()}
    layer: list[tuple[Puzzle, MoveCombination]] = [(finished, tuple())]

    for depth in range(1, max_depth + 1):
        print_stamped(f"starting depth = {depth}...")
        next_layer: list[tuple[Puzzle, MoveCombination]] = []
        for puzzle, combination in layer:
            for ma, mi, md in moves:
                next_combination = combination + ((ma, mi, md),)
                if not is_allowed(n, next_combination):
                    continue  # ignore disallowed move combinations
                next_puzzle = puzzle.copy()
                next_puzzle.execute_move(ma, mi, md)
                if next_puzzle in encountered:
                    encountered_combination = encountered[next_puzzle]
                    assert len(encountered_combination) <= len(next_combination)
                    if encountered_combination not in duplicates:
                        duplicates[encountered_combination] = set()
                    duplicates[encountered_combination].add(next_combination)
                else:
                    encountered[next_puzzle] = next_combination
                    next_layer.append((next_puzzle, next_combination))
        layer = next_layer

    print(duplicates)
    duplicate_count = sum([len(dups) for dups in duplicates.values()])
    print(f"duplicate count: {duplicate_count}")
