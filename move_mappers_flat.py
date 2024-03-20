import z3


def z3_corner_x_hi(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
):
    return z3.If(
        z3.And(ax == 1, hi == y_hi),
        z3.If(
            dr == 0,
            z3.Not(z_hi),
            z3.If(dr == 1, z_hi, z3.Not(x_hi)),
        ),
        z3.If(
            z3.And(ax == 2, hi == z_hi),
            z3.If(
                dr == 0,
                y_hi,
                z3.If(dr == 1, z3.Not(y_hi), z3.Not(x_hi)),
            ),
            x_hi,
        ),
    )


def z3_corner_y_hi(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
):
    return z3.If(
        z3.And(ax == 0, hi == x_hi),
        z3.If(
            dr == 0,
            z_hi,
            z3.If(dr == 1, z3.Not(z_hi), z3.Not(y_hi)),
        ),
        z3.If(
            z3.And(ax == 2, hi == z_hi),
            z3.If(
                dr == 0,
                z3.Not(x_hi),
                z3.If(dr == 1, x_hi, z3.Not(y_hi)),
            ),
            y_hi,
        ),
    )


def z3_corner_z_hi(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
):
    return z3.If(
        z3.And(ax == 0, hi == x_hi),
        z3.If(
            dr == 0,
            z3.Not(y_hi),
            z3.If(dr == 1, y_hi, z3.Not(z_hi)),
        ),
        z3.If(
            z3.And(ax == 1, hi == y_hi),
            z3.If(
                dr == 0,
                x_hi,
                z3.If(dr == 1, z3.Not(x_hi), z3.Not(z_hi)),
            ),
            z_hi,
        ),
    )


def z3_corner_r(
    x_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    r: z3.ArithRef,
    cw: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
):
    # Condition for next == (r + 1) % 3
    add_one = z3.If(r == 0, 1, z3.If(r == 1, 2, 0))

    # Condition for next == (r - 1) % 3
    minus_one = z3.If(r == 0, 2, z3.If(r == 1, 0, 1))

    return z3.If(
        dr != 2,
        z3.If(
            z3.And(ax == 0, hi == x_hi),
            z3.If(cw, minus_one, add_one),
            z3.If(
                z3.And(ax == 2, hi == z_hi),
                z3.If(cw, add_one, minus_one),
                r,
            ),
        ),
        r,
    )


def z3_corner_cw(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    cw: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
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
        z3.Not(cw),
        cw,
    )


def z3_edge_a(
    a: z3.ArithRef,
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
):
    return z3.If(
        dr != 2,
        z3.If(
            a == 0,
            z3.If(
                z3.And(ax == 1, hi == y_hi),
                2,
                z3.If(z3.And(ax == 2, hi == x_hi), 1, a),
            ),
            z3.If(
                a == 1,
                z3.If(
                    z3.And(ax == 0, hi == x_hi),
                    2,
                    z3.If(z3.And(ax == 2, hi == y_hi), 0, a),
                ),
                z3.If(
                    z3.And(ax == 0, hi == x_hi),
                    1,
                    z3.If(z3.And(ax == 1, hi == y_hi), 0, a),
                ),
            ),
        ),
        a,
    )


def z3_edge_x_hi(
    a: z3.ArithRef,
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
):
    return z3.If(
        a == 0,
        z3.If(
            z3.And(ax == 1, hi == y_hi, dr != 1),
            z3.Not(x_hi),
            z3.If(
                z3.And(ax == 2, hi == x_hi),
                z3.If(
                    dr == 0,
                    y_hi,
                    z3.If(dr == 1, z3.Not(y_hi), x_hi),
                ),
                x_hi,
            ),
        ),
        z3.If(
            z3.And(a == 1, ax == 2, hi == y_hi),
            z3.If(dr == 2, z3.Not(x_hi), y_hi),
            z3.If(
                z3.And(a == 2, ax == 1, hi == y_hi, dr != 0),
                z3.Not(x_hi),
                x_hi,
            ),
        ),
    )


def z3_edge_y_hi(
    a: z3.ArithRef,
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    ax: z3.ArithRef,
    hi: z3.BoolRef,
    dr: z3.ArithRef,
):
    return z3.If(
        z3.And(a == 0, ax == 2, hi == x_hi),
        z3.If(dr == 2, z3.Not(y_hi), x_hi),
        z3.If(
            a == 1,
            z3.If(
                z3.And(ax == 0, hi == x_hi, dr != 0),
                z3.Not(y_hi),
                z3.If(
                    z3.And(ax == 2, hi == y_hi),
                    z3.If(
                        dr == 0,
                        z3.Not(x_hi),
                        z3.If(dr == 1, x_hi, y_hi),
                    ),
                    y_hi,
                ),
            ),
            z3.If(
                z3.And(a == 2, ax == 0, hi == x_hi, dr != 1),
                z3.Not(y_hi),
                y_hi,
            ),
        ),
    )


def z3_edge_r(a: z3.ArithRef, next_a: z3.ArithRef, r: z3.BoolRef):
    return z3.If(
        z3.Or(z3.And(a == 0, next_a == 1), z3.And(a == 1, next_a == 0)),
        z3.Not(r),
        r,
    )
