import argparse

from puzzle import Puzzle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("randomizations", type=int)
    args = parser.parse_args()
    Puzzle.random(args.n, args.randomizations).to_file()
