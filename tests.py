"""Various (but not exhaustive) tests for all modules."""

import os
import random
import unittest

from generate_random import PUZZLE_DIR, all_moves
from puzzle import (
    DEFAULT_CENTER_COLORS,
    Puzzle,
    cubicle_colors,
    cubicle_type,
    decode_corner,
    decode_edge,
    encode_corner,
    encode_edge,
    facelet_cubicle,
    inverse_move,
    move_name,
    parse_move,
)


class Testing(unittest.TestCase):
    """Various unit tests."""

    def test_cubicle_type(self):
        """Test whether the cubicle type function works as expected."""
        self.assertEqual(cubicle_type(2, (1, 1, 1)), 0)
        self.assertEqual(cubicle_type(3, (1, 1, 1)), -1)
        self.assertEqual(cubicle_type(3, (2, 2, 2)), 0)
        self.assertEqual(cubicle_type(3, (1, 2, 1)), 1)
        self.assertEqual(cubicle_type(3, (1, 2, 2)), 2)

    def test_cubicle_colors(self):
        """Test whether the cubicle colors function works as expected."""
        self.assertEqual(cubicle_colors(2, (0, 0, 0), DEFAULT_CENTER_COLORS), [5, 0, 3])
        self.assertEqual(cubicle_colors(3, (0, 0, 0), DEFAULT_CENTER_COLORS), [5, 0, 3])
        self.assertEqual(cubicle_colors(3, (2, 2, 2), DEFAULT_CENTER_COLORS), [4, 2, 1])
        self.assertEqual(cubicle_colors(3, (1, 2, 1), DEFAULT_CENTER_COLORS), [4])
        self.assertEqual(cubicle_colors(3, (2, 2, 1), DEFAULT_CENTER_COLORS), [4, 1])

    def test_facelet_cubicle(self):
        """Test whether the facelet cubicle function works as expected."""
        self.assertEqual(facelet_cubicle(2, (0, 1, 1)), (1, 1, 0))
        self.assertEqual(facelet_cubicle(3, (2, 2, 1)), (1, 2, 2))
        self.assertEqual(facelet_cubicle(3, (4, 2, 2)), (2, 2, 2))
        self.assertEqual(facelet_cubicle(3, (5, 2, 1)), (1, 0, 0))

    def test_encoding_decoding_cubies(self):
        """Test whether encoding and decoding cubie coordinates is bijective."""
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

    def test_all_moves(self):
        """Test whether the list of all moves is complete."""
        self.assertEqual(len(all_moves()), 18)

    def test_encoding_decoding_moves(self):
        """Test whether encoding and decoding moves is bijective."""
        for move in all_moves():
            self.assertEqual(move, parse_move(move_name(move)))

    def test_puzzle_parsing(self):
        """Test whether parsing and serializing puzzles is bijective."""
        for filename in os.listdir(PUZZLE_DIR):
            if filename.endswith(".txt"):
                path = os.path.join(PUZZLE_DIR, filename)
                with open(path) as file:
                    puzzle = file.read()
                    self.assertEqual(str(Puzzle.from_str(puzzle)), puzzle)

    def test_move_consistency(self):
        """Test whether executing moves and reverting them is bijective."""
        for n in [2, 3]:
            # Take a random permutation of all possible moves.
            moves = all_moves()
            random.shuffle(moves)

            state = Puzzle.finished(n, DEFAULT_CENTER_COLORS)
            states = []

            # Execute the moves and store the states before each move.
            for move in moves:
                states.append(state)
                state = state.execute_move(move)

            # Execute the inverted moves and check whether we get the same states.
            moves.reverse()
            states.reverse()
            for i, move in enumerate(moves):
                state = state.execute_move(inverse_move(move))
                self.assertEqual(state, states[i])


if __name__ == "__main__":
    unittest.main()
