import os
import random
import unittest

from generate import list_all_moves
from puzzle import (
    SUPPORTED_NS,
    Cubicles,
    Puzzle,
    center_to_coord,
    coord_to_center,
    coord_to_corner,
    coord_to_edge,
    corner_to_coord,
    cubicle_colors,
    cubicle_type,
    edge_to_coord,
    facelet_cubicle,
)


class Testing(unittest.TestCase):
    def test_cubicle_type(self):
        self.assertEqual(cubicle_type(2, 1, 1, 1), 0)
        self.assertEqual(cubicle_type(3, 1, 1, 1), -1)
        self.assertEqual(cubicle_type(3, 2, 2, 2), 0)
        self.assertEqual(cubicle_type(3, 1, 2, 1), 1)
        self.assertEqual(cubicle_type(3, 1, 2, 2), 2)

    def test_cubicle_colors(self):
        for n in SUPPORTED_NS:
            self.assertEqual(cubicle_colors(n, 0, 0, 0), [5, 0, 3])
            self.assertEqual(cubicle_colors(3, 2, 2, 2), [4, 2, 1])
        self.assertEqual(cubicle_colors(3, 1, 2, 1), [4])
        self.assertEqual(cubicle_colors(3, 2, 2, 1), [4, 1])

    def test_facelet_cubicle(self):
        self.assertEqual(facelet_cubicle(2, 0, 1, 1), (1, 1, 0))
        self.assertEqual(facelet_cubicle(3, 2, 2, 1), (1, 2, 2))
        self.assertEqual(facelet_cubicle(3, 4, 2, 2), (2, 2, 2))
        self.assertEqual(facelet_cubicle(3, 5, 2, 1), (1, 0, 0))

    def test_encoding_decoding(self):
        for n in SUPPORTED_NS:
            for x in range(n):
                for y in range(n):
                    for z in range(n):
                        match cubicle_type(n, x, y, z):
                            case 0:
                                self.assertEqual(
                                    (x, y, z),
                                    corner_to_coord(n, coord_to_corner(n, x, y, z)),
                                )
                            case 1:
                                self.assertEqual(
                                    (x, y, z),
                                    center_to_coord(n, coord_to_center(n, x, y, z)),
                                )
                            case 2:
                                self.assertEqual(
                                    (x, y, z),
                                    edge_to_coord(coord_to_edge(x, y, z)),
                                )

    def test_finished_cubicles(self):
        cubicles = Cubicles(2)
        self.assertEqual(len(cubicles.corners), 8)
        self.assertEqual(len(cubicles.centers), 0)
        self.assertEqual(len(cubicles.edges), 0)

        cubicles = Cubicles(3)
        self.assertEqual(len(cubicles.corners), 8)
        self.assertEqual(len(cubicles.centers), 6)
        self.assertEqual(len(cubicles.edges), 12)

    def test_puzzle_parsing(self):
        dir = "./puzzles"
        for filename in os.listdir(dir):
            if filename.endswith(".txt"):
                path = os.path.join(dir, filename)
                with open(path, "r") as file:
                    puzzle = file.read()
                    self.assertEqual(Puzzle.from_str(puzzle).to_str(), puzzle)

    def test_puzzle_move_consistency(self):
        for n in [2, 3]:
            # Take a random permutation of all possible moves.
            moves = list_all_moves(n)
            random.shuffle(moves)

            state = Puzzle.finished(n)
            states = []

            # Execute the moves and store the states before each move.
            for ma, mi, md in moves:
                states.append(state.copy())
                state.execute_move(ma, mi, md)

            # Execute the inverted moves and check whether we get the same states.
            moves.reverse()
            states.reverse()
            for i, (ma, mi, md) in enumerate(moves):
                if md == 0:
                    md_inv = 1
                elif md == 1:
                    md_inv = 0
                else:
                    md_inv = 2
                state.execute_move(ma, mi, md_inv)
                self.assertEqual(state, states[i])


if __name__ == "__main__":
    unittest.main()
