"""'Stacked' versions of the flat mapper functions. They allow applying multiple moves
at once. They use recursion to apply the moves stepwise.
"""

import move_mappers_flat
from state import CornerStateZ3, EdgeStateZ3, MoveZ3, TernaryZ3


def corner_x_hi(c: CornerStateZ3, moves: list[MoveZ3]):
    """Return the next value of x_hi. Apply the moves in the given list."""
    assert len(moves) > 0
    if len(moves) == 1:
        return move_mappers_flat.corner_x_hi(c, moves[0])
    else:
        return move_mappers_flat.corner_x_hi(
            CornerStateZ3(
                corner_x_hi(c, moves[:-1]),
                corner_y_hi(c, moves[:-1]),
                corner_z_hi(c, moves[:-1]),
                corner_r(c, moves[:-1]),
                corner_cw(c, moves[:-1]),
            ),
            moves[-1],
        )


def corner_y_hi(c: CornerStateZ3, moves: list[MoveZ3]):
    """Return the next value of y_hi. Apply the moves in the given list."""
    assert len(moves) > 0
    if len(moves) == 1:
        return move_mappers_flat.corner_y_hi(c, moves[0])
    else:
        return move_mappers_flat.corner_y_hi(
            CornerStateZ3(
                corner_x_hi(c, moves[:-1]),
                corner_y_hi(c, moves[:-1]),
                corner_z_hi(c, moves[:-1]),
                corner_r(c, moves[:-1]),
                corner_cw(c, moves[:-1]),
            ),
            moves[-1],
        )


def corner_z_hi(c: CornerStateZ3, moves: list[MoveZ3]):
    """Return the next value of z_hi. Apply the moves in the given list."""
    assert len(moves) > 0
    if len(moves) == 1:
        return move_mappers_flat.corner_z_hi(c, moves[0])
    else:
        return move_mappers_flat.corner_z_hi(
            CornerStateZ3(
                corner_x_hi(c, moves[:-1]),
                corner_y_hi(c, moves[:-1]),
                corner_z_hi(c, moves[:-1]),
                corner_r(c, moves[:-1]),
                corner_cw(c, moves[:-1]),
            ),
            moves[-1],
        )


def corner_r(c: CornerStateZ3, moves: list[MoveZ3]):
    """Return the next value of r. Apply the moves in the given list."""
    assert len(moves) > 0
    if len(moves) == 1:
        return TernaryZ3(
            move_mappers_flat.corner_r_b1(c, moves[0]),
            move_mappers_flat.corner_r_b2(c, moves[0]),
        )
    else:
        next = CornerStateZ3(
            corner_x_hi(c, moves[:-1]),
            corner_y_hi(c, moves[:-1]),
            corner_z_hi(c, moves[:-1]),
            corner_r(c, moves[:-1]),
            corner_cw(c, moves[:-1]),
        )
        return TernaryZ3(
            move_mappers_flat.corner_r_b1(next, moves[-1]),
            move_mappers_flat.corner_r_b2(next, moves[-1]),
        )


def corner_cw(c: CornerStateZ3, moves: list[MoveZ3]):
    """Return the next value of cw. Apply the moves in the given list."""
    assert len(moves) > 0
    if len(moves) == 1:
        return move_mappers_flat.corner_cw(c, moves[0])
    else:
        return move_mappers_flat.corner_cw(
            CornerStateZ3(
                corner_x_hi(c, moves[:-1]),
                corner_y_hi(c, moves[:-1]),
                corner_z_hi(c, moves[:-1]),
                corner_r(c, moves[:-1]),
                corner_cw(c, moves[:-1]),
            ),
            moves[-1],
        )


def edge_a(e: EdgeStateZ3, moves: list[MoveZ3]):
    """Return the next value of a. Apply the moves in the given list."""
    assert len(moves) > 0
    if len(moves) == 1:
        return TernaryZ3(
            move_mappers_flat.edge_a_b1(e, moves[0]),
            move_mappers_flat.edge_a_b2(e, moves[0]),
        )
    else:
        next = EdgeStateZ3(
            edge_a(e, moves[:-1]),
            edge_x_hi(e, moves[:-1]),
            edge_y_hi(e, moves[:-1]),
            edge_r(e, moves[:-1]),
        )
        return TernaryZ3(
            move_mappers_flat.edge_a_b1(next, moves[-1]),
            move_mappers_flat.edge_a_b2(next, moves[-1]),
        )


def edge_x_hi(e: EdgeStateZ3, moves: list[MoveZ3]):
    """Return the next value of x_hi. Apply the moves in the given list."""
    assert len(moves) > 0
    if len(moves) == 1:
        return move_mappers_flat.edge_x_hi(e, moves[0])
    else:
        return move_mappers_flat.edge_x_hi(
            EdgeStateZ3(
                edge_a(e, moves[:-1]),
                edge_x_hi(e, moves[:-1]),
                edge_y_hi(e, moves[:-1]),
                edge_r(e, moves[:-1]),
            ),
            moves[-1],
        )


def edge_y_hi(e: EdgeStateZ3, moves: list[MoveZ3]):
    """Return the next value of y_hi. Apply the moves in the given list."""
    assert len(moves) > 0
    if len(moves) == 1:
        return move_mappers_flat.edge_y_hi(e, moves[0])
    else:
        return move_mappers_flat.edge_y_hi(
            EdgeStateZ3(
                edge_a(e, moves[:-1]),
                edge_x_hi(e, moves[:-1]),
                edge_y_hi(e, moves[:-1]),
                edge_r(e, moves[:-1]),
            ),
            moves[-1],
        )


def edge_r(e: EdgeStateZ3, moves: list[MoveZ3]):
    """Return the next value of r. Apply the moves in the given list."""
    assert len(moves) > 0
    if len(moves) == 1:
        next_a = TernaryZ3(
            move_mappers_flat.edge_a_b1(e, moves[0]),
            move_mappers_flat.edge_a_b2(e, moves[0]),
        )
        return move_mappers_flat.edge_r(e, next_a)
    else:
        return move_mappers_flat.edge_r(
            EdgeStateZ3(
                edge_a(e, moves[:-1]),
                edge_x_hi(e, moves[:-1]),
                edge_y_hi(e, moves[:-1]),
                edge_r(e, moves[:-1]),
            ),
            edge_a(e, moves),
        )
