from z3 import Solver
import argparse
import os


_PUZZLES_FOLDER = 'puzzles/'
_FLORIAN_FOLDER = 'florian/'


def puzzle_us_to_florian(puzzle: str) -> str:
    face_1 = puzzle[0:4]
    face_2 = puzzle[4:8]
    face_3 = puzzle[8:12]
    face_4 = puzzle[12:16]
    face_5 = puzzle[16:20]
    face_6 = puzzle[20:24]

    face_1 = face_1[1] + face_1[3] + face_1[0] + face_1[2]
    face_2 = face_2[1] + face_2[3] + face_2[0] + face_2[2]
    face_3 = face_3[2] + face_3[0] + face_3[3] + face_3[1]
    face_4 = face_4[1] + face_4[3] + face_4[0] + face_4[2]
    face_5 = face_5[1] + face_5[3] + face_5[0] + face_5[2]
    face_6 = face_6[1] + face_6[3] + face_6[0] + face_6[2]

    return face_2 + face_1 + face_5 + face_3 + face_6 + face_4


def puzzle_file_to_florian(puzzle_file: str) -> None:
    puzzle = open(f'{_PUZZLES_FOLDER}{puzzle_file}', 'r').readline()
    open(f'{_FLORIAN_FOLDER}{_PUZZLES_FOLDER}{puzzle_file}', 'w').write(puzzle_us_to_florian(puzzle))


def florian_to_dimacs(puzzle_file: str, number_of_moves: int) -> None:
    puzzle_file = puzzle_file + '.txt'
    puzzle_file_to_florian(puzzle_file)

    os.system(
        f'python {_FLORIAN_FOLDER}/rubiks_sat.py {_FLORIAN_FOLDER}{_PUZZLES_FOLDER}{puzzle_file} {number_of_moves}'
    )


def solve_florians_dimacs():
    s = Solver()
    s.from_file(f'{_FLORIAN_FOLDER}dimacs/rubiks.dimacs')

    print(s.check())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("nr_of_moves", type=int)
    args = parser.parse_args()

    florian_to_dimacs(args.path, args.nr_of_moves)
    solve_florians_dimacs()
