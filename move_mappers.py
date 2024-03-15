"""Module containing the move mapping functions for the behaviour of cubies given their
state and the move taken. All functions have a plain Python implementation, and a Z3
implementation. Both implementations should be equivalent."""

import z3


def generic_cubie_coord(
    n: int, x: int, y: int, z: int, ax: int, hi: bool, dr: int
) -> tuple[int, int, int]:
    """Generic cubie coordinate mapping constructed from 90 or 180 degree rotation
    matrices. This serves as a reference implementation for the other functions."""
    if ax == 0 and ((hi and x == n - 1) or (not hi and x == 0)):
        if dr == 0:
            return (x, z, n - 1 - y)
        elif dr == 1:
            return (x, n - 1 - z, y)
        elif dr == 2:
            return (x, n - 1 - y, n - 1 - z)
    elif ax == 1 and ((hi and y == n - 1) or (not hi and y == 0)):
        if dr == 0:
            return (n - 1 - z, y, x)
        elif dr == 1:
            return (z, y, n - 1 - x)
        elif dr == 2:
            return (n - 1 - x, y, n - 1 - z)
    elif ax == 2 and ((hi and z == n - 1) or (not hi and z == 0)):
        if dr == 0:
            return (y, n - 1 - x, z)
        elif dr == 1:
            return (n - 1 - y, x, z)
        elif dr == 2:
            return (n - 1 - x, n - 1 - y, z)
    return (x, y, z)


def corner_x_hi(x_hi: bool, y_hi: bool, z_hi: bool, ax: int, hi: bool, dr: int) -> bool:
    if ax == 1 and hi == y_hi:
        if dr == 0:
            return not z_hi
        elif dr == 1:
            return z_hi
        elif dr == 2:
            return not x_hi
    elif ax == 2 and hi == z_hi:
        if dr == 0:
            return y_hi
        elif dr == 1:
            return not y_hi
        elif dr == 2:
            return not x_hi
    return x_hi


def z3_corner_x_hi(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
    next: z3.BoolRef,
):
    return (
        z3.If(
            z3.And(ax == 1, hi == y_hi),
            z3.If(dr == 0, next != z_hi, z3.If(dr == 1, next == z_hi, next != x_hi)),
            z3.If(
                z3.And(ax == 2, hi == z_hi),
                z3.If(
                    dr == 0, next == y_hi, z3.If(dr == 1, next != y_hi, next != x_hi)
                ),
                next == x_hi,
            ),
        ),
    )


def corner_y_hi(x_hi: bool, y_hi: bool, z_hi: bool, ax: int, hi: bool, dr: int) -> bool:
    if ax == 0 and hi == x_hi:
        if dr == 0:
            return z_hi
        elif dr == 1:
            return not z_hi
        elif dr == 2:
            return not y_hi
    elif ax == 2 and hi == z_hi:
        if dr == 0:
            return not x_hi
        elif dr == 1:
            return x_hi
        elif dr == 2:
            return not y_hi
    return y_hi


def z3_corner_y_hi(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
    next: z3.BoolRef,
):
    return (
        z3.If(
            z3.And(ax == 0, hi == x_hi),
            z3.If(dr == 0, next == z_hi, z3.If(dr == 1, next != z_hi, next != y_hi)),
            z3.If(
                z3.And(ax == 2, hi == z_hi),
                z3.If(
                    dr == 0, next != x_hi, z3.If(dr == 1, next == x_hi, next != y_hi)
                ),
                next == y_hi,
            ),
        ),
    )


def corner_z_hi(x_hi: bool, y_hi: bool, z_hi: bool, ax: int, hi: bool, dr: int) -> bool:
    if ax == 0 and hi == x_hi:
        if dr == 0:
            return not y_hi
        elif dr == 1:
            return y_hi
        elif dr == 2:
            return not z_hi
    elif ax == 1 and hi == y_hi:
        if dr == 0:
            return x_hi
        elif dr == 1:
            return not x_hi
        elif dr == 2:
            return not z_hi
    return z_hi


def z3_corner_z_hi(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
    next: z3.BoolRef,
):
    return (
        z3.If(
            z3.And(ax == 0, hi == x_hi),
            z3.If(dr == 0, next != y_hi, z3.If(dr == 1, next == y_hi, next != z_hi)),
            z3.If(
                z3.And(ax == 1, hi == y_hi),
                z3.If(
                    dr == 0, next == x_hi, z3.If(dr == 1, next != x_hi, next != z_hi)
                ),
                next == z_hi,
            ),
        ),
    )


def corner_r(
    x_hi: bool, z_hi: bool, r: int, cw: bool, ax: int, hi: bool, dr: int
) -> int:
    if dr != 2:
        if ax == 0 and hi == x_hi:
            if cw:
                return (r - 1) % 3
            else:
                return (r + 1) % 3
        elif ax == 2 and hi == z_hi:
            if cw:
                return (r + 1) % 3
            else:
                return (r - 1) % 3
    return r


def z3_corner_r(
    x_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    r: z3.ArithRef,
    cw: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
    next: z3.ArithRef,
):
    # Condition for next == (r + 1) % 3
    add_one = z3.If(r == 0, next == 1, z3.If(r == 1, next == 2, next == 0))

    # Condition for next == (r - 1) % 3
    minus_one = z3.If(r == 0, next == 2, z3.If(r == 1, next == 0, next == 1))

    return z3.If(
        dr != 2,
        z3.If(
            z3.And(ax == 0, hi == x_hi),
            z3.If(cw, minus_one, add_one),
            z3.If(
                z3.And(ax == 2, hi == z_hi),
                z3.If(cw, add_one, minus_one),
                next == r,
            ),
        ),
        next == r,
    )


def corner_cw(
    x_hi: bool, y_hi: bool, z_hi: bool, cw: bool, ax: int, hi: bool, dr: int
) -> bool:
    if dr != 2 and (
        (ax == 0 and hi == x_hi) or (ax == 1 and hi == y_hi) or (ax == 2 and hi == z_hi)
    ):
        return not cw
    return cw


def z3_corner_cw(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    cw: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
    next: z3.BoolRef,
):
    return z3.If(
        z3.And(
            dr != 2,
            z3.Or(
                z3.And(ax == 0, hi == x_hi),
                z3.And(ax == 1, hi == y_hi),
                z3.And(ax == 2, hi == z_hi),
            ),
        ),
        next != cw,
        next == cw,
    )


def edge_x(n: int, x: int, y: int, z: int, ax: int, hi: bool, dr: int) -> int:
    if ax == 1 and ((hi and y == n - 1) or (not hi and y == 0)):
        if dr == 0:
            return n - 1 - z
        elif dr == 1:
            return z
        elif dr == 2:
            return n - 1 - x
    elif ax == 2 and ((hi and z == n - 1) or (not hi and z == 0)):
        if dr == 0:
            return y
        elif dr == 1:
            return n - 1 - y
        elif dr == 2:
            return n - 1 - x
    return x


def z3_edge_x(
    n: int,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
    next: z3.ArithRef,
):
    return z3.If(
        z3.And(ax == 1, z3.If(hi, y == n - 1, y == 0)),
        z3.If(
            dr == 0, next == (n - 1) - z, z3.If(dr == 1, next == z, next == (n - 1) - x)
        ),
        z3.If(
            z3.And(ax == 2, z3.If(hi, z == n - 1, z == 0)),
            z3.If(
                dr == 0,
                next == y,
                z3.If(dr == 1, next == (n - 1) - y, next == (n - 1) - x),
            ),
            next == x,
        ),
    )


def edge_y(n: int, x: int, y: int, z: int, ax: int, hi: bool, dr: int) -> int:
    if ax == 0 and ((hi and x == n - 1) or (not hi and x == 0)):
        if dr == 0:
            return z
        elif dr == 1:
            return n - 1 - z
        elif dr == 2:
            return n - 1 - y
    elif ax == 2 and ((hi and z == n - 1) or (not hi and z == 0)):
        if dr == 0:
            return n - 1 - x
        elif dr == 1:
            return x
        elif dr == 2:
            return n - 1 - y
    return y


def z3_edge_y(
    n: int,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
    next: z3.ArithRef,
):
    return z3.If(
        z3.And(ax == 0, z3.If(hi, x == n - 1, x == 0)),
        z3.If(
            dr == 0, next == z, z3.If(dr == 1, next == (n - 1) - z, next == (n - 1) - y)
        ),
        z3.If(
            z3.And(ax == 2, z3.If(hi, z == n - 1, z == 0)),
            z3.If(
                dr == 0,
                next == (n - 1) - x,
                z3.If(dr == 1, next == x, next == (n - 1) - y),
            ),
            next == y,
        ),
    )


def edge_z(n: int, x: int, y: int, z: int, ax: int, hi: bool, dr: int) -> int:
    if ax == 0 and ((hi and x == n - 1) or (not hi and x == 0)):
        if dr == 0:
            return n - 1 - y
        elif dr == 1:
            return y
        elif dr == 2:
            return n - 1 - z
    elif ax == 1 and ((hi and y == n - 1) or (not hi and y == 0)):
        if dr == 0:
            return x
        elif dr == 1:
            return n - 1 - x
        elif dr == 2:
            return n - 1 - z
    return z


def z3_edge_z(
    n: int,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
    next: z3.ArithRef,
):
    return z3.If(
        z3.And(ax == 0, z3.If(hi, x == n - 1, x == 0)),
        z3.If(
            dr == 0,
            next == (n - 1) - y,
            z3.If(dr == 1, next == y, next == (n - 1) - z),
        ),
        z3.If(
            z3.And(ax == 1, z3.If(hi, y == n - 1, y == 0)),
            z3.If(
                dr == 0,
                next == x,
                z3.If(dr == 1, next == (n - 1) - x, next == (n - 1) - z),
            ),
            next == z,
        ),
    )


def edge_r(n: int, x: int, z: int, r: bool, ax: int, hi: bool, dr: int) -> bool:
    if dr != 2 and (
        (ax == 0 and ((hi and x == n - 1) or (not hi and x == 0)))
        or (ax == 2 and ((hi and z == n - 1) or (not hi and z == 0)))
    ):
        return not r
    return r


def z3_edge_r(
    n: int,
    x: z3.ArithRef,
    z: z3.ArithRef,
    r: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
    next: z3.BoolRef,
):
    return z3.If(
        z3.And(
            dr != 2,
            z3.Or(
                z3.And(ax == 0, z3.If(hi, x == n - 1, x == 0)),
                z3.And(ax == 2, z3.If(hi, z == n - 1, z == 0)),
            ),
        ),
        next != r,
        next == r,
    )
