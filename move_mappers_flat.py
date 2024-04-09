"""'Flattened' version of the mapper functions. Instead of returning an equality which
must hold, only the values the next value should equal to are returned.
"""

import z3

from state import CornerStateZ3, EdgeStateZ3, MoveZ3, TernaryZ3


def corner_x_hi(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of x_hi."""
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


def corner_y_hi(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of y_hi."""
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


def corner_z_hi(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of z_hi."""
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


def corner_r_b1(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of r.b1."""
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


def corner_r_b2(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of r.b2."""
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


def corner_cw(c: CornerStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of cw."""
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


def edge_a_b1(e: EdgeStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of a.b1."""
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


def edge_a_b2(e: EdgeStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of a.b2."""
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


def edge_x_hi(e: EdgeStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of x_hi."""
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


def edge_y_hi(e: EdgeStateZ3, m: MoveZ3) -> z3.BoolRef:
    """Return the next value of y_hi."""
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


def edge_r(e: EdgeStateZ3, next_a: TernaryZ3) -> z3.BoolRef:
    """Return the next value of r."""
    result = z3.If(
        z3.Or(z3.And(e.a == 0, next_a == 1), z3.And(e.a == 1, next_a == 0)),
        z3.Not(e.r),
        e.r,
    )
    assert isinstance(result, z3.BoolRef)
    return result
