import argparse
import ast
import os

from puzzle import DEFAULT_CENTER_COLORS, Puzzle
from state import CornerState, EdgeState, Move
from tools import str_to_file

CUBIE_MIN_PATTERNS_RESULTS = "./cubie_min_patterns_results"


def corner_file_path(n: int) -> str:
    """Return the file path for the corner patterns for n."""
    os.makedirs(CUBIE_MIN_PATTERNS_RESULTS, exist_ok=True)
    return os.path.join(CUBIE_MIN_PATTERNS_RESULTS, f"n{n}-corners.txt")


def edge_file_path(n: int) -> str:
    """Return the file path for the edge patterns for n."""
    os.makedirs(CUBIE_MIN_PATTERNS_RESULTS, exist_ok=True)
    return os.path.join(CUBIE_MIN_PATTERNS_RESULTS, f"n{n}-edges.txt")


def find_cubie_min_patterns(n: int):
    """Find the strongest corner and edge patterns for n and output them to file."""
    moves = Move.list_all()
    finished = Puzzle.finished(n, "???", DEFAULT_CENTER_COLORS)

    depth = 0
    enc: set[Puzzle] = set([finished])
    enc_corners: list[dict[CornerState, int]] = [{c: 0} for c in finished.corners]
    enc_edges: list[dict[EdgeState, int]] = [{e: 0} for e in finished.edges]
    fresh: set[Puzzle] = set([finished])

    while True:
        depth += 1
        changed = False
        next_fresh: set[Puzzle] = set()
        for state in fresh:
            for move in moves:
                new_state = state.execute_move(move)
                if new_state not in enc:
                    enc.add(new_state)
                    next_fresh.add(new_state)

                    for i, corner in enumerate(new_state.corners):
                        if corner not in enc_corners[i]:
                            enc_corners[i][corner] = depth
                            changed = True
                    for i, edge in enumerate(new_state.edges):
                        if edge not in enc_edges[i]:
                            enc_edges[i][edge] = depth
                            changed = True
        fresh = next_fresh
        if not changed:
            break

    max_corner_depth = max(max(enc_corner.values()) for enc_corner in enc_corners)
    strongest_corner_patterns = [
        {str(c): dep for c, dep in enc_corner.items() if dep >= max_corner_depth}
        for enc_corner in enc_corners
    ]
    str_to_file(str(strongest_corner_patterns), corner_file_path(n))

    if n == 3:
        max_edge_depth = max(max(enc_edge.values()) for enc_edge in enc_edges)
    else:
        max_edge_depth = 0  # there are no edges anyway
    strongest_edge_patterns = [
        {str(e): dep for e, dep in enc_edge.items() if dep >= max_edge_depth}
        for enc_edge in enc_edges
    ]
    str_to_file(str(strongest_edge_patterns), edge_file_path(n))


def load_corner_min_patterns(n: int):
    """Return the corner patterns for n."""
    path = corner_file_path(n)
    if not os.path.isfile(path):
        find_cubie_min_patterns(n)

    with open(path) as file:
        strongest_corner_patterns_raw: list[dict[str, int]] = ast.literal_eval(
            file.read()
        )
        strongest_corner_patterns = [
            {CornerState.from_str(c): dep for c, dep in enc_corner.items()}
            for enc_corner in strongest_corner_patterns_raw
        ]
        return strongest_corner_patterns


def load_edge_min_patterns(n: int):
    """Return the edge patterns for n."""
    path = edge_file_path(n)
    if not os.path.isfile(path):
        find_cubie_min_patterns(n)

    with open(path) as file:
        strongest_edge_patterns_raw: list[dict[str, int]] = ast.literal_eval(
            file.read()
        )
        strongest_edge_patterns = [
            {EdgeState.from_str(e): dep for e, dep in enc_edge.items()}
            for enc_edge in strongest_edge_patterns_raw
        ]
        return strongest_edge_patterns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    args = parser.parse_args()
    find_cubie_min_patterns(args.n)
