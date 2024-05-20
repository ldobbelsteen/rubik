# ruff: noqa: D101, D102

import unittest

import move_filter_finder
from puzzle import (
    DEFAULT_CENTER_COLORS,
    Puzzle,
    all_puzzles_names,
    # cubie_colors,
    facelet_cubie,
)
from state import CornerState, EdgeState, Move, MoveSeq, cubie_type


class MoveFilterFinderModule(unittest.TestCase):
    def test_variable_encoding_decoding(self):
        for step in range(10):
            var = move_filter_finder.Variable("test", step)
            self.assertEqual(
                move_filter_finder.Variable.from_str(str(var)),
                var,
            )

    def test_condition_encoding_decoding(self):
        for step in range(10):
            for op in move_filter_finder.Operator:
                left = move_filter_finder.Variable("test_left", step)
                right = move_filter_finder.Variable("test_right", step)
                cond = move_filter_finder.Condition(left, op, right)
                self.assertEqual(
                    move_filter_finder.Condition.from_str(str(cond)),
                    cond,
                )


class PuzzleModule(unittest.TestCase):
    def test_cubie_colors(self):
        pass
        # TODO: Fix test cases as DEFAULT_CENTER_COLORS is not used anymore.
        # self.assertEqual(cubie_colors(2, 0, 0, 0, DEFAULT_CENTER_COLORS), [5, 0, 3])
        # self.assertEqual(cubie_colors(3, 0, 0, 0, DEFAULT_CENTER_COLORS), [5, 0, 3])
        # self.assertEqual(cubie_colors(3, 2, 2, 2, DEFAULT_CENTER_COLORS), [4, 2, 1])
        # self.assertEqual(cubie_colors(3, 1, 2, 1, DEFAULT_CENTER_COLORS), [4])
        # self.assertEqual(cubie_colors(3, 2, 2, 1, DEFAULT_CENTER_COLORS), [4, 1])

    def test_facelet_cubie(self):
        self.assertEqual(facelet_cubie(2, 0, 1, 1), (1, 1, 0))
        self.assertEqual(facelet_cubie(3, 2, 2, 1), (1, 2, 2))
        self.assertEqual(facelet_cubie(3, 4, 2, 2), (2, 2, 2))
        self.assertEqual(facelet_cubie(3, 5, 2, 1), (1, 0, 0))

    def test_puzzle_encoding_decoding(self):
        for name in all_puzzles_names():
            puzzle = Puzzle.from_file(name)
            self.assertTrue(puzzle.is_valid())
            self.assertEqual(puzzle, Puzzle.from_str(str(puzzle), name))

    def test_puzzle_execute_move_consistency(self):
        for n in (2, 3):
            # Take a random move sequence.
            seq = MoveSeq.random(100)

            # Take a random puzzle state.
            state = Puzzle.random(n, 100)
            states = []

            # Execute the moves and store the states before each move.
            for move in seq:
                states.append(state)
                state = state.execute_move(move)
                self.assertTrue(state.is_valid())

            # Execute the inverted moves and check whether we get the same states.
            states.reverse()
            for i, move in enumerate(seq.inverted()):
                state = state.execute_move(move)
                self.assertEqual(state, states[i])
                self.assertTrue(state.is_valid())

    def test_puzzle_execute_move_seq(self):
        for n in (2, 3):
            seq = MoveSeq.random(100)
            state = Puzzle.random(n, 100)
            self.assertEqual(
                state,
                state.execute_move_seq(seq).execute_move_seq(
                    seq.inverted(),
                ),
            )

    def test_random_puzzle_is_valid(self):
        for n in (2, 3):
            puzzle = Puzzle.random(n, 100)
            self.assertTrue(puzzle.is_valid())

    def test_puzzle_is_finished(self):
        for n in (2, 3):
            seq = MoveSeq.random(100)
            finished = Puzzle.finished(n, "???", DEFAULT_CENTER_COLORS)
            self.assertTrue(finished.is_finished())
            self.assertTrue(
                finished.execute_move_seq(seq)
                .execute_move_seq(seq.inverted())
                .is_finished()
            )

    def test_puzzle_is_solution(self):
        for n in (2, 3):
            seq = MoveSeq.random(100)
            finished = Puzzle.finished(n, "???", DEFAULT_CENTER_COLORS)
            self.assertTrue(finished.execute_move_seq(seq).is_solution(seq.inverted()))


class StateModule(unittest.TestCase):
    def test_encoding_decoding_moves(self):
        for move in Move.list_all():
            self.assertEqual(move, move.from_str(str(move)))

    def test_encoding_decoding_moveseqs(self):
        for k in range(20):
            seq = MoveSeq.random(k)
            self.assertEqual(seq, MoveSeq.from_str(str(seq)))

    def test_encoding_decoding_corner_state(self):
        for n in (2, 3):
            for corner_state in CornerState.all_finished(n):
                self.assertEqual(
                    corner_state, CornerState.from_str(n, str(corner_state))
                )

    def test_encoding_decoding_edge_state(self):
        for n in (2, 3):
            for edge_state in EdgeState.all_finished(n):
                self.assertEqual(edge_state, EdgeState.from_str(n, str(edge_state)))

    def test_inverse_move(self):
        for n in (2, 3):
            for move in Move.list_all():
                puzzle = Puzzle.random(n, 20)
                self.assertEqual(
                    puzzle,
                    puzzle.execute_move(move).execute_move(move.inverse()),
                )

    def test_all_moves_len(self):
        self.assertEqual(len(Move.list_all()), 18)

    def test_cubie_type(self):
        self.assertEqual(cubie_type(2, 1, 1, 1), 0)
        self.assertEqual(cubie_type(3, 1, 1, 1), -1)
        self.assertEqual(cubie_type(3, 2, 2, 2), 0)
        self.assertEqual(cubie_type(3, 1, 2, 1), 1)
        self.assertEqual(cubie_type(3, 1, 2, 2), 2)

    def test_encoding_decoding_cubies(self):
        for n in (2, 3):
            for x in range(n):
                for y in range(n):
                    for z in range(n):
                        match cubie_type(n, x, y, z):
                            case 0:
                                self.assertEqual(
                                    (x, y, z),
                                    CornerState.from_coords(n, x, y, z).coords(),
                                )
                            case 2:
                                self.assertEqual(
                                    (x, y, z),
                                    EdgeState.from_coords(n, x, y, z).coords(),
                                )

    def test_finished_states_len(self):
        self.assertEqual(len(CornerState.all_finished(2)), 8)
        self.assertEqual(len(EdgeState.all_finished(2)), 0)
        self.assertEqual(len(CornerState.all_finished(3)), 8)
        self.assertEqual(len(EdgeState.all_finished(3)), 12)


if __name__ == "__main__":
    unittest.main()
