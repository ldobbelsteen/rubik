import argparse
import os

from puzzle import Puzzle
from solve import SolveInstance
from solve_config import SolveConfig

DIMACS_RESULTS = "./dimacs_results"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("k", type=int)
    args = parser.parse_args()
    SolveInstance(Puzzle.from_file(args.name), args.k, SolveConfig()).to_dimacs(
        os.path.join(DIMACS_RESULTS, f"{args.name}_{args.k}.dimacs")
    )
