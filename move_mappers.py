"""All of the move mapping functions for simulating the behaviour of cubies given their
state and the move taken. These functions are the Z3 implementations of the 'next'
functions in the state module. They are used in the solver to generate the constraints.
"""

import z3

from state import (
    CornerStateZ3,
    EdgeStateZ3,
    MoveZ3,
    TernaryZ3,
)


def corner_x_hi(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner x_hi. Returns the constraint
    of the next value being equal to the previous value with the move applied.
    """
    result = z3.If(
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
    assert isinstance(result, z3.BoolRef)
    return result


def corner_x_hi_flat(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner x_hi. This flat version directly
    returns previous value with the move applied.
    """
    result = z3.If(
        z3.And(m.ax == 1, m.hi == c.y_hi),
        z3.If(
            m.dr == 0,
            z3.Not(c.z_hi),
            z3.If(m.dr == 1, c.z_hi, z3.Not(c.x_hi)),
        ),
        z3.If(
            z3.And(m.ax == 2, m.hi == c.z_hi),
            z3.If(
                m.dr == 0,
                c.y_hi,
                z3.If(m.dr == 1, z3.Not(c.y_hi), z3.Not(c.x_hi)),
            ),
            c.x_hi,
        ),
    )
    assert isinstance(result, z3.BoolRef)
    return result


def corner_y_hi(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner y_hi. Returns the constraint
    of the next value being equal to the previous value with the move applied.
    """
    result = z3.If(
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
    assert isinstance(result, z3.BoolRef)
    return result


def corner_y_hi_flat(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner y_hi. This flat version directly
    returns previous value with the move applied.
    """
    result = z3.If(
        z3.And(m.ax == 0, m.hi == c.x_hi),
        z3.If(
            m.dr == 0,
            c.z_hi,
            z3.If(m.dr == 1, z3.Not(c.z_hi), z3.Not(c.y_hi)),
        ),
        z3.If(
            z3.And(m.ax == 2, m.hi == c.z_hi),
            z3.If(
                m.dr == 0,
                z3.Not(c.x_hi),
                z3.If(m.dr == 1, c.x_hi, z3.Not(c.y_hi)),
            ),
            c.y_hi,
        ),
    )
    assert isinstance(result, z3.BoolRef)
    return result


def corner_z_hi(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner z_hi. Returns the constraint
    of the next value being equal to the previous value with the move applied.
    """
    result = z3.If(
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
    assert isinstance(result, z3.BoolRef)
    return result


def corner_z_hi_flat(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner z_hi. This flat version directly
    returns previous value with the move applied.
    """
    result = z3.If(
        z3.And(m.ax == 0, m.hi == c.x_hi),
        z3.If(
            m.dr == 0,
            z3.Not(c.y_hi),
            z3.If(m.dr == 1, c.y_hi, z3.Not(c.z_hi)),
        ),
        z3.If(
            z3.And(m.ax == 1, m.hi == c.y_hi),
            z3.If(
                m.dr == 0,
                c.x_hi,
                z3.If(m.dr == 1, z3.Not(c.x_hi), z3.Not(c.z_hi)),
            ),
            c.z_hi,
        ),
    )
    assert isinstance(result, z3.BoolRef)
    return result


def corner_r(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner r. Returns the constraint
    of the next value being equal to the previous value with the move applied.
    """
    add_one = z3.If(
        c.r == 0, nxt.r == 1, z3.If(c.r == 1, nxt.r == 2, nxt.r == 0)
    )  # condition for next == (r + 1) % 3

    minus_one = z3.If(
        c.r == 0, nxt.r == 2, z3.If(c.r == 1, nxt.r == 0, nxt.r == 1)
    )  # condition for next == (r - 1) % 3
    result = z3.If(
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
    assert isinstance(result, z3.BoolRef)
    return result


def corner_r_b1_flat(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner r.b1. This flat version directly
    returns previous value with the move applied.
    """
    minus_one = z3.And(z3.Not(c.r.b1), z3.Not(c.r.b2))  # condition for (r - 1) % 3
    add_one = c.r.b2  # condition for (r + 1) % 3

    result = z3.If(
        m.dr != 2,
        z3.If(
            z3.And(m.ax == 0, m.hi == c.x_hi),
            z3.If(c.cw, minus_one, add_one),
            z3.If(
                z3.And(m.ax == 2, m.hi == c.z_hi),
                z3.If(c.cw, add_one, minus_one),
                c.r.b1,
            ),
        ),
        c.r.b1,
    )
    assert isinstance(result, z3.BoolRef)
    return result


def corner_r_b2_flat(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner r.b2. This flat version directly
    returns previous value with the move applied.
    """
    minus_one = c.r.b1  # condition for (r - 1) % 3
    add_one = z3.And(z3.Not(c.r.b1), z3.Not(c.r.b2))  # condition for (r + 1) % 3

    result = z3.If(
        m.dr != 2,
        z3.If(
            z3.And(m.ax == 0, m.hi == c.x_hi),
            z3.If(c.cw, minus_one, add_one),
            z3.If(
                z3.And(m.ax == 2, m.hi == c.z_hi),
                z3.If(c.cw, add_one, minus_one),
                c.r.b2,
            ),
        ),
        c.r.b2,
    )
    assert isinstance(result, z3.BoolRef)
    return result


def corner_cw(c: CornerStateZ3, m: MoveZ3, nxt: CornerStateZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner cw. Returns the constraint
    of the next value being equal to the previous value with the move applied.
    """
    result = z3.If(
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
    assert isinstance(result, z3.BoolRef)
    return result


def corner_cw_flat(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of corner cw. This flat version directly
    returns previous value with the move applied.
    """
    result = z3.If(
        z3.And(
            m.dr != 2,
            z3.Or(
                z3.And(m.ax == 0, m.hi == c.x_hi),
                z3.And(m.ax == 1, m.hi == c.y_hi),
                z3.And(m.ax == 2, m.hi == c.z_hi),
            ),
        ),
        z3.Not(c.cw),
        c.cw,
    )
    assert isinstance(result, z3.BoolRef)
    return result


def edge_a(e: EdgeStateZ3, m: MoveZ3, nxt: EdgeStateZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of edge a. Returns the constraint
    of the next value being equal to the previous value with the move applied.
    """
    result = z3.If(
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
    assert isinstance(result, z3.BoolRef)
    return result


def edge_a_b1_flat(e: EdgeStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of edge a.b1. This flat version directly
    returns previous value with the move applied.
    """
    result = z3.If(
        m.dr != 2,
        z3.If(
            e.a == 0,
            z3.If(
                z3.And(m.ax == 1, m.hi == e.y_hi),
                True,
                z3.If(z3.And(m.ax == 2, m.hi == e.x_hi), False, e.a.b1),
            ),
            z3.If(
                e.a == 1,
                z3.If(
                    z3.And(m.ax == 0, m.hi == e.x_hi),
                    True,
                    z3.If(z3.And(m.ax == 2, m.hi == e.y_hi), False, e.a.b1),
                ),
                z3.If(
                    z3.And(m.ax == 0, m.hi == e.x_hi),
                    False,
                    z3.If(z3.And(m.ax == 1, m.hi == e.y_hi), False, e.a.b1),
                ),
            ),
        ),
        e.a.b1,
    )
    assert isinstance(result, z3.BoolRef)
    return result


def edge_a_b2_flat(e: EdgeStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of edge a.b2. This flat version directly
    returns previous value with the move applied.
    """
    result = z3.If(
        m.dr != 2,
        z3.If(
            e.a == 0,
            z3.If(
                z3.And(m.ax == 1, m.hi == e.y_hi),
                False,
                z3.If(z3.And(m.ax == 2, m.hi == e.x_hi), True, e.a.b2),
            ),
            z3.If(
                e.a == 1,
                z3.If(
                    z3.And(m.ax == 0, m.hi == e.x_hi),
                    False,
                    z3.If(z3.And(m.ax == 2, m.hi == e.y_hi), False, e.a.b2),
                ),
                z3.If(
                    z3.And(m.ax == 0, m.hi == e.x_hi),
                    True,
                    z3.If(z3.And(m.ax == 1, m.hi == e.y_hi), False, e.a.b2),
                ),
            ),
        ),
        e.a.b2,
    )
    assert isinstance(result, z3.BoolRef)
    return result


def edge_x_hi(e: EdgeStateZ3, m: MoveZ3, nxt: EdgeStateZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of edge x_hi. Returns the constraint
    of the next value being equal to the previous value with the move applied.
    """
    result = z3.If(
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
    assert isinstance(result, z3.BoolRef)
    return result


def edge_x_hi_flat(e: EdgeStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of edge x_hi. This flat version directly
    returns previous value with the move applied.
    """
    result = z3.If(
        e.a == 0,
        z3.If(
            z3.And(m.ax == 1, m.hi == e.y_hi, m.dr != 1),
            z3.Not(e.x_hi),
            z3.If(
                z3.And(m.ax == 2, m.hi == e.x_hi),
                z3.If(
                    m.dr == 0,
                    e.y_hi,
                    z3.If(m.dr == 1, z3.Not(e.y_hi), e.x_hi),
                ),
                e.x_hi,
            ),
        ),
        z3.If(
            z3.And(e.a == 1, m.ax == 2, m.hi == e.y_hi),
            z3.If(m.dr == 2, z3.Not(e.x_hi), e.y_hi),
            z3.If(
                z3.And(e.a == 2, m.ax == 1, m.hi == e.y_hi, m.dr != 0),
                z3.Not(e.x_hi),
                e.x_hi,
            ),
        ),
    )
    assert isinstance(result, z3.BoolRef)
    return result


def edge_y_hi(e: EdgeStateZ3, m: MoveZ3, nxt: EdgeStateZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of edge y_hi. Returns the constraint
    of the next value being equal to the previous value with the move applied.
    """
    result = z3.If(
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
    assert isinstance(result, z3.BoolRef)
    return result


def edge_y_hi_flat(e: EdgeStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of edge y_hi. This flat version directly
    returns previous value with the move applied.
    """
    result = z3.If(
        z3.And(e.a == 0, m.ax == 2, m.hi == e.x_hi),
        z3.If(m.dr == 2, z3.Not(e.y_hi), e.x_hi),
        z3.If(
            e.a == 1,
            z3.If(
                z3.And(m.ax == 0, m.hi == e.x_hi, m.dr != 0),
                z3.Not(e.y_hi),
                z3.If(
                    z3.And(m.ax == 2, m.hi == e.y_hi),
                    z3.If(
                        m.dr == 0,
                        z3.Not(e.x_hi),
                        z3.If(m.dr == 1, e.x_hi, e.y_hi),
                    ),
                    e.y_hi,
                ),
            ),
            z3.If(
                z3.And(e.a == 2, m.ax == 0, m.hi == e.x_hi, m.dr != 1),
                z3.Not(e.y_hi),
                e.y_hi,
            ),
        ),
    )
    assert isinstance(result, z3.BoolRef)
    return result


def edge_r(e: EdgeStateZ3, nxt: EdgeStateZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of edge r. Returns the constraint
    of the next value being equal to the previous value with the move applied.
    """
    result = z3.If(
        z3.Or(z3.And(e.a == 0, nxt.a == 1), z3.And(e.a == 1, nxt.a == 0)),
        nxt.r != e.r,
        nxt.r == e.r,
    )
    assert isinstance(result, z3.BoolRef)
    return result


def edge_r_flat(e: EdgeStateZ3, next_a: TernaryZ3) -> z3.BoolRef:
    """Z3 implementation of the next value of edge r. This flat version directly
    returns previous value with the move applied.
    """
    result = z3.If(
        z3.Or(z3.And(e.a == 0, next_a == 1), z3.And(e.a == 1, next_a == 0)),
        z3.Not(e.r),
        e.r,
    )
    assert isinstance(result, z3.BoolRef)
    return result
