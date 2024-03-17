import os
import random
import unittest

from generate_random import PUZZLE_DIR, all_moves
from puzzle import (
    DEFAULT_CENTER_COLORS,
    FinishedState,
    Puzzle,
    cubicle_colors,
    cubicle_type,
    decode_corner,
    decode_edge,
    encode_corner,
    encode_edge,
    facelet_cubicle,
    inverse_move,
)


class Testing(unittest.TestCase):
    def test_cubicle_type(self):
        self.assertEqual(cubicle_type(2, (1, 1, 1)), 0)
        self.assertEqual(cubicle_type(3, (1, 1, 1)), -1)
        self.assertEqual(cubicle_type(3, (2, 2, 2)), 0)
        self.assertEqual(cubicle_type(3, (1, 2, 1)), 1)
        self.assertEqual(cubicle_type(3, (1, 2, 2)), 2)

    def test_cubicle_colors(self):
        self.assertEqual(cubicle_colors(2, (0, 0, 0), DEFAULT_CENTER_COLORS), [5, 0, 3])
        self.assertEqual(cubicle_colors(3, (0, 0, 0), DEFAULT_CENTER_COLORS), [5, 0, 3])
        self.assertEqual(cubicle_colors(3, (2, 2, 2), DEFAULT_CENTER_COLORS), [4, 2, 1])
        self.assertEqual(cubicle_colors(3, (1, 2, 1), DEFAULT_CENTER_COLORS), [4])
        self.assertEqual(cubicle_colors(3, (2, 2, 1), DEFAULT_CENTER_COLORS), [4, 1])

    def test_facelet_cubicle(self):
        self.assertEqual(facelet_cubicle(2, (0, 1, 1)), (1, 1, 0))
        self.assertEqual(facelet_cubicle(3, (2, 2, 1)), (1, 2, 2))
        self.assertEqual(facelet_cubicle(3, (4, 2, 2)), (2, 2, 2))
        self.assertEqual(facelet_cubicle(3, (5, 2, 1)), (1, 0, 0))

    def test_encoding_decoding(self):
        for n in [2, 3]:
            for x in range(n):
                for y in range(n):
                    for z in range(n):
                        cubicle = (x, y, z)
                        match cubicle_type(n, cubicle):
                            case 0:
                                self.assertEqual(
                                    cubicle,
                                    decode_corner(n, encode_corner(n, cubicle)),
                                )
                            case 2:
                                self.assertEqual(
                                    cubicle,
                                    decode_edge(n, encode_edge(n, cubicle)),
                                )

    def test_finished_cubicles_sizes(self):
        cubicles = FinishedState(2)
        self.assertEqual(len(cubicles.corners), 8)
        self.assertEqual(len(cubicles.edges), 0)

        cubicles = FinishedState(3)
        self.assertEqual(len(cubicles.corners), 8)
        self.assertEqual(len(cubicles.edges), 12)

    def test_puzzle_parsing(self):
        for filename in os.listdir(PUZZLE_DIR):
            if filename.endswith(".txt"):
                path = os.path.join(PUZZLE_DIR, filename)
                with open(path, "r") as file:
                    puzzle = file.read()
                    self.assertEqual(Puzzle.from_str(puzzle).to_str(), puzzle)

    def test_puzzle_move_execution(self):
        for n in [2, 3]:
            # Take a random permutation of all possible moves.
            moves = all_moves()
            random.shuffle(moves)

            state = Puzzle.finished(n, DEFAULT_CENTER_COLORS)
            states = []

            # Execute the moves and store the states before each move.
            for move in moves:
                states.append(state.copy())
                state.execute_move(move)

            # Execute the inverted moves and check whether we get the same states.
            moves.reverse()
            states.reverse()
            for i, move in enumerate(moves):
                state.execute_move(inverse_move(move))
                self.assertEqual(state, states[i])


if __name__ == "__main__":
    unittest.main()
