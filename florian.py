from z3 import Solver
import os


puzzle_file = 'puzzles/n2-k5-0.txt'
puzzle = open(puzzle_file, 'r').readline()
print(puzzle)

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

florian = face_2 + face_1 + face_5 + face_3 + face_6 + face_4
print(florian)
open(f'florian/{puzzle_file}', 'w').write(florian)

number_of_moves = 5
os.system(f'python florian/rubiks_sat.py florian/{puzzle_file} {number_of_moves}')

s = Solver()
s.from_file('florian/dimacs/rubiks.dimacs')

# print(s)
print(s.check())
print(s.model())
