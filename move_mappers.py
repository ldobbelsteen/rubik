"""All of the move mapping functions for simulating the behaviour of cubies given their
state and the move taken. These functions are the Z3 implementations of the 'next'
functions in the state module. They are used in the solver to generate the constraints.
"""

import z3

from state import (
    CornerStateZ3,
    EdgeStateZ3,
    MoveZ3,
)


def corner_x_hi(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3):
    """Z3 implementation of the next value of x_hi. Returns a constraint of the next
    value being equal to the value.
    """
    return z3.If(
        z3.And(m.ax == 1, m.hi == c.y_hi),
        z3.If(
            m.dr == 0,
            nxt.x_hi != c.z_hi,
            z3.If(m.dr == 1, nxt.x_hi == c.z_hi, nxt.x_hi != c.x_hi),
        ),
        z3.If(
            z3.And(m.ax == 2, m.hi == c.z_hi),
            z3.If(
                m.dr == 0,
                nxt.x_hi == c.y_hi,
                z3.If(m.dr == 1, nxt.x_hi != c.y_hi, nxt.x_hi != c.x_hi),
            ),
            nxt.x_hi == c.x_hi,
        ),
    )


def corner_y_hi(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3):
    """Z3 implementation of the next value of y_hi. Returns a constraint of the next
    value being equal to the value.
    """
    return z3.If(
        z3.And(m.ax == 0, m.hi == c.x_hi),
        z3.If(
            m.dr == 0,
            nxt.y_hi == c.z_hi,
            z3.If(m.dr == 1, nxt.y_hi != c.z_hi, nxt.y_hi != c.y_hi),
        ),
        z3.If(
            z3.And(m.ax == 2, m.hi == c.z_hi),
            z3.If(
                m.dr == 0,
                nxt.y_hi != c.x_hi,
                z3.If(m.dr == 1, nxt.y_hi == c.x_hi, nxt.y_hi != c.y_hi),
            ),
            nxt.y_hi == c.y_hi,
        ),
    )


def corner_z_hi(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3):
    """Z3 implementation of the next value of z_hi. Returns a constraint of the next
    value being equal to the value.
    """
    return z3.If(
        z3.And(m.ax == 0, m.hi == c.x_hi),
        z3.If(
            m.dr == 0,
            nxt.z_hi != c.y_hi,
            z3.If(m.dr == 1, nxt.z_hi == c.y_hi, nxt.z_hi != c.z_hi),
        ),
        z3.If(
            z3.And(m.ax == 1, m.hi == c.y_hi),
            z3.If(
                m.dr == 0,
                nxt.z_hi == c.x_hi,
                z3.If(m.dr == 1, nxt.z_hi != c.x_hi, nxt.z_hi != c.z_hi),
            ),
            nxt.z_hi == c.z_hi,
        ),
    )


def corner_r(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3):
    """Z3 implementation of the next value of r. Returns a constraint of the next value
    being equal to the value.
    """
    add_one = z3.If(
        c.r == 0, nxt.r == 1, z3.If(c.r == 1, nxt.r == 2, nxt.r == 0)
    )  # condition for next == (r + 1) % 3

    minus_one = z3.If(
        c.r == 0, nxt.r == 2, z3.If(c.r == 1, nxt.r == 0, nxt.r == 1)
    )  # condition for next == (r - 1) % 3
    return z3.If(
        m.dr != 2,
        z3.If(
            z3.And(m.ax == 0, m.hi == c.x_hi),
            z3.If(c.cw, minus_one, add_one),
            z3.If(
                z3.And(m.ax == 2, m.hi == c.z_hi),
                z3.If(c.cw, add_one, minus_one),
                nxt.r == c.r,
            ),
        ),
        nxt.r == c.r,
    )


def corner_cw(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3):
    """Z3 implementation of the next value of cw. Returns a constraint of the next value
    being equal to the value.
    """
    return z3.If(
        z3.And(
            m.dr != 2,
            z3.Or(
                z3.And(m.ax == 0, m.hi == c.x_hi),
                z3.And(m.ax == 1, m.hi == c.y_hi),
                z3.And(m.ax == 2, m.hi == c.z_hi),
            ),
        ),
        nxt.cw != c.cw,
        nxt.cw == c.cw,
    )


def edge_a(e: EdgeStateZ3, m: MoveZ3, nxt: EdgeStateZ3):
    """Z3 implementation of the next value of a. Returns a constraint of the next value
    being equal to the value.
    """
    return z3.If(
        m.dr != 2,
        z3.If(
            e.a == 0,
            z3.If(
                z3.And(m.ax == 1, m.hi == e.y_hi),
                nxt.a == 2,
                z3.If(z3.And(m.ax == 2, m.hi == e.x_hi), nxt.a == 1, nxt.a == e.a),
            ),
            z3.If(
                e.a == 1,
                z3.If(
                    z3.And(m.ax == 0, m.hi == e.x_hi),
                    nxt.a == 2,
                    z3.If(z3.And(m.ax == 2, m.hi == e.y_hi), nxt.a == 0, nxt.a == e.a),
                ),
                z3.If(
                    z3.And(m.ax == 0, m.hi == e.x_hi),
                    nxt.a == 1,
                    z3.If(z3.And(m.ax == 1, m.hi == e.y_hi), nxt.a == 0, nxt.a == e.a),
                ),
            ),
        ),
        nxt.a == e.a,
    )


def edge_x_hi(e: EdgeStateZ3, m: MoveZ3, nxt: EdgeStateZ3):
    """Z3 implementation of the next value of x_hi. Returns a constraint of the next
    value being equal to the value.
    """
    return z3.If(
        e.a == 0,
        z3.If(
            z3.And(m.ax == 1, m.hi == e.y_hi, m.dr != 1),
            nxt.x_hi != e.x_hi,
            z3.If(
                z3.And(m.ax == 2, m.hi == e.x_hi),
                z3.If(
                    m.dr == 0,
                    nxt.x_hi == e.y_hi,
                    z3.If(m.dr == 1, nxt.x_hi != e.y_hi, nxt.x_hi == e.x_hi),
                ),
                nxt.x_hi == e.x_hi,
            ),
        ),
        z3.If(
            z3.And(e.a == 1, m.ax == 2, m.hi == e.y_hi),
            z3.If(m.dr == 2, nxt.x_hi != e.x_hi, nxt.x_hi == e.y_hi),
            z3.If(
                z3.And(e.a == 2, m.ax == 1, m.hi == e.y_hi, m.dr != 0),
                nxt.x_hi != e.x_hi,
                nxt.x_hi == e.x_hi,
            ),
        ),
    )


def edge_y_hi(e: EdgeStateZ3, m: MoveZ3, nxt: EdgeStateZ3):
    """Z3 implementation of the next value of y_hi. Returns a constraint of the next
    value being equal to the value.
    """
    return z3.If(
        z3.And(e.a == 0, m.ax == 2, m.hi == e.x_hi),
        z3.If(m.dr == 2, nxt.y_hi != e.y_hi, nxt.y_hi == e.x_hi),
        z3.If(
            e.a == 1,
            z3.If(
                z3.And(m.ax == 0, m.hi == e.x_hi, m.dr != 0),
                nxt.y_hi != e.y_hi,
                z3.If(
                    z3.And(m.ax == 2, m.hi == e.y_hi),
                    z3.If(
                        m.dr == 0,
                        nxt.y_hi != e.x_hi,
                        z3.If(m.dr == 1, nxt.y_hi == e.x_hi, nxt.y_hi == e.y_hi),
                    ),
                    nxt.y_hi == e.y_hi,
                ),
            ),
            z3.If(
                z3.And(e.a == 2, m.ax == 0, m.hi == e.x_hi, m.dr != 1),
                nxt.y_hi != e.y_hi,
                nxt.y_hi == e.y_hi,
            ),
        ),
    )


def edge_r(e: EdgeStateZ3, nxt: EdgeStateZ3):
    """Z3 implementation of the next value of r. Returns a constraint of the next value
    being equal to the value.
    """
    return z3.If(
        z3.Or(z3.And(e.a == 0, nxt.a == 1), z3.And(e.a == 1, nxt.a == 0)),
        nxt.r != e.r,
        nxt.r == e.r,
    )
