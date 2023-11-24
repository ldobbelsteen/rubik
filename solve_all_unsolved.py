import sys
import os
import solve_puzzle

# python solve_all_unsolved.py {directory} {max_moves} {minimize_cores}
if __name__ == "__main__":
    directory = sys.argv[1]
    max_moves = int(sys.argv[2])
    minimize_cores = int(sys.argv[3])

    n = int(directory[-1])
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for file in files:
        if file.endswith(".txt") and file + ".solution" not in files:
            print(f"\nsolving {file} with max moves {max_moves} and minimize cores {minimize_cores}...")
            solve_puzzle.main(file, max_moves, minimize_cores)
