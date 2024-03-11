import sys

from generate import moveset
from misc import print_stamped
from puzzle import Puzzle

MoveCombination = tuple[tuple[int, int, int], ...]


def is_allowed(mc: MoveCombination) -> bool:
    return True  # TODO: implement


# e.g. python symmetric_move_combinations.py {n} {max_depth}
if __name__ == "__main__":
    n = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    finished = Puzzle.finished(n)
    moves = moveset(n)

    identical: dict[MoveCombination, set[MoveCombination]] = {}
    encountered: dict[Puzzle, MoveCombination] = {finished: tuple()}
    layer: list[tuple[Puzzle, MoveCombination]] = [(finished, tuple())]

    for depth in range(max_depth + 1):
        print_stamped(f"starting depth = {depth}...")
        next_layer: list[tuple[Puzzle, MoveCombination]] = []
        for puzzle, combination in layer:
            for ma, mi, md in moves:
                next_combination = combination + ((ma, mi, md),)
                if not is_allowed(next_combination):
                    continue  # ignore disallowed move combinations
                next_puzzle = puzzle.copy()
                next_puzzle.execute_move(ma, mi, md)
                if next_puzzle in encountered:
                    encountered_combination = encountered[next_puzzle]
                    if encountered_combination not in identical:
                        identical[encountered_combination] = set()
                    identical[encountered_combination].add(next_combination)
                else:
                    encountered[next_puzzle] = next_combination
                    next_layer.append((next_puzzle, next_combination))
        layer = next_layer

        print(identical)
