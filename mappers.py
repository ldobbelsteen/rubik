import z3


def generic_cubie_coord(
    n: int, x: int, y: int, z: int, ma: int, mi: int, md: int
) -> tuple[int, int, int]:
    """Generic cubie coordinate mapping based on 90 or 180 degree rotation matrices."""
    if ma == 0 and mi == y:
        if md == 0:
            return (z, y, n - 1 - x)  # clockwise
        elif md == 1:
            return (n - 1 - z, y, x)  # counterclockwise
        elif md == 2:
            return (n - 1 - x, y, n - 1 - z)  # 180 degree
    elif ma == 1 and mi == x:
        if md == 0:
            return (x, n - 1 - z, y)  # counterclockwise
        elif md == 1:
            return (x, z, n - 1 - y)  # clockwise
        elif md == 2:
            return (x, n - 1 - y, n - 1 - z)  # 180 degree
    elif ma == 2 and mi == z:
        if md == 0:
            return (y, n - 1 - x, z)  # clockwise
        elif md == 1:
            return (n - 1 - y, x, z)  # counterclockwise
        elif md == 2:
            return (n - 1 - x, n - 1 - y, z)  # 180 degree
    return (x, y, z)


def corner_x(n: int, x: bool, y: bool, z: bool, ma: int, mi: int, md: int) -> bool:
    if ma == 0 and (mi == n - 1 if y else mi == 0):
        if md == 0:
            return z
        elif md == 1:
            return not z
        elif md == 2:
            return not x
    elif ma == 2 and (mi == n - 1 if z else mi == 0):
        if md == 0:
            return y
        elif md == 1:
            return not y
        elif md == 2:
            return not x
    return x


def z3_corner_x(
    n: int,
    x: z3.BoolRef,
    y: z3.BoolRef,
    z: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_x: z3.BoolRef,
):
    return z3.If(
        z3.And(ma == 0, z3.If(y, mi == n - 1, mi == 0)),
        z3.If(
            md == 0,
            next_x == z,
            z3.If(md == 1, next_x != z, next_x != x),
        ),
        z3.If(
            z3.And(ma == 2, z3.If(z, mi == n - 1, mi == 0)),
            z3.If(
                md == 0,
                next_x == y,
                z3.If(md == 1, next_x != y, next_x != x),
            ),
            next_x == x,
        ),
    )


def corner_y(n: int, x: bool, y: bool, z: bool, ma: int, mi: int, md: int) -> bool:
    if ma == 1 and (mi == n - 1 if x else mi == 0):
        if md == 0:
            return not z
        elif md == 1:
            return z
        elif md == 2:
            return not y
    elif ma == 2 and (mi == n - 1 if z else mi == 0):
        if md == 0:
            return not x
        elif md == 1:
            return x
        elif md == 2:
            return not y
    return y


def z3_corner_y(
    n: int,
    x: z3.BoolRef,
    y: z3.BoolRef,
    z: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_y: z3.BoolRef,
):
    return z3.If(
        z3.And(ma == 1, z3.If(x, mi == n - 1, mi == 0)),
        z3.If(
            md == 0,
            next_y != z,
            z3.If(md == 1, next_y == z, next_y != y),
        ),
        z3.If(
            z3.And(ma == 2, z3.If(z, mi == n - 1, mi == 0)),
            z3.If(
                md == 0,
                next_y != x,
                z3.If(md == 1, next_y == x, next_y != y),
            ),
            next_y == y,
        ),
    )


def corner_z(n: int, x: bool, y: bool, z: bool, ma: int, mi: int, md: int) -> bool:
    if ma == 0 and (mi == n - 1 if y else mi == 0):
        if md == 0:
            return not x
        elif md == 1:
            return x
        elif md == 2:
            return not z
    elif ma == 1 and (mi == n - 1 if x else mi == 0):
        if md == 0:
            return y
        elif md == 1:
            return not y
        elif md == 2:
            return not z
    return z


def z3_corner_z(
    n: int,
    x: z3.BoolRef,
    y: z3.BoolRef,
    z: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_z: z3.BoolRef,
):
    return z3.If(
        z3.And(ma == 0, z3.If(y, mi == n - 1, mi == 0)),
        z3.If(
            md == 0,
            next_z != x,
            z3.If(md == 1, next_z == x, next_z != z),
        ),
        z3.If(
            z3.And(ma == 1, z3.If(x, mi == n - 1, mi == 0)),
            z3.If(
                md == 0,
                next_z == y,
                z3.If(md == 1, next_z != y, next_z != z),
            ),
            next_z == z,
        ),
    )


def corner_r(
    n: int, x: bool, z: bool, r: int, c: bool, ma: int, mi: int, md: int
) -> int:
    if md != 2:
        if ma == 1 and (mi == n - 1 if x else mi == 0):
            if c:
                return (r - 1) % 3
            else:
                return (r + 1) % 3
        elif ma == 2 and (mi == n - 1 if z else mi == 0):
            if c:
                return (r + 1) % 3
            else:
                return (r - 1) % 3
    return r


def z3_corner_r(
    n: int,
    x: z3.BoolRef,
    z: z3.BoolRef,
    r: z3.ArithRef,
    c: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_r: z3.ArithRef,
):
    # Condition for next_r == (r + 1) % 3
    add_one = z3.If(r == 0, next_r == 1, z3.If(r == 1, next_r == 2, next_r == 0))

    # Condition for next_r == (r - 1) % 3
    minus_one = z3.If(r == 0, next_r == 2, z3.If(r == 1, next_r == 0, next_r == 1))

    return z3.If(
        md != 2,
        z3.If(
            z3.And(ma == 1, z3.If(x, mi == n - 1, mi == 0)),
            z3.If(c, minus_one, add_one),
            z3.If(
                z3.And(ma == 2, z3.If(z, mi == n - 1, mi == 0)),
                z3.If(c, add_one, minus_one),
                next_r == r,
            ),
        ),
        next_r == r,
    )


def corner_c(
    n: int, x: bool, y: bool, z: bool, c: bool, ma: int, mi: int, md: int
) -> bool:
    if md != 2 and (
        (ma == 0 and (mi == n - 1 if y else mi == 0))
        or (ma == 1 and (mi == n - 1 if x else mi == 0))
        or (ma == 2 and (mi == n - 1 if z else mi == 0))
    ):
        return not c
    return c


def z3_corner_c(
    n: int,
    x: z3.BoolRef,
    y: z3.BoolRef,
    z: z3.BoolRef,
    c: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_c: z3.BoolRef,
):
    return z3.If(
        z3.And(
            md != 2,
            z3.Or(
                z3.And(ma == 0, z3.If(y, mi == n - 1, mi == 0)),
                z3.And(ma == 1, z3.If(x, mi == n - 1, mi == 0)),
                z3.And(ma == 2, z3.If(z, mi == n - 1, mi == 0)),
            ),
        ),
        next_c != c,
        next_c == c,
    )


def center_a(a: int, ma: int, mi: int, md: int) -> int:
    if mi == 1 and md != 2:
        if ma == 0:
            if a == 0:
                return 2
            elif a == 2:
                return 0
        elif ma == 1:
            if a == 1:
                return 2
            elif a == 2:
                return 1
        elif ma == 2:
            if a == 0:
                return 1
            elif a == 1:
                return 0
    return a


def z3_center_a(
    a: z3.ArithRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_a: z3.ArithRef,
):
    return z3.If(
        z3.And(mi == 1, md != 2),
        z3.If(
            ma == 0,
            z3.If(a == 0, next_a == 2, z3.If(a == 2, next_a == 0, next_a == a)),
            z3.If(
                ma == 1,
                z3.If(a == 1, next_a == 2, z3.If(a == 2, next_a == 1, next_a == a)),
                z3.If(a == 0, next_a == 1, z3.If(a == 1, next_a == 0, next_a == a)),
            ),
        ),
        next_a == a,
    )


def center_h(a: int, h: bool, ma: int, mi: int, md: int) -> bool:
    if mi == 1:
        if ma == 0:
            if (a == 0 and md != 1) or (a == 2 and md != 0):
                return not h
        elif ma == 1:
            if (a == 1 and md != 0) or (a == 2 and md != 1):
                return not h
        elif ma == 2:
            if (a == 0 and md != 1) or (a == 1 and md != 0):
                return not h
    return h


def z3_center_h(
    a: z3.ArithRef,
    h: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_h: z3.BoolRef,
):
    return z3.If(
        mi == 1,
        z3.If(
            ma == 0,
            z3.If(
                z3.Or(z3.And(a == 0, md != 1), z3.And(a == 2, md != 0)),
                next_h != h,
                next_h == h,
            ),
            z3.If(
                ma == 1,
                z3.If(
                    z3.Or(z3.And(a == 1, md != 0), z3.And(a == 2, md != 1)),
                    next_h != h,
                    next_h == h,
                ),
                z3.If(
                    z3.Or(z3.And(a == 0, md != 1), z3.And(a == 1, md != 0)),
                    next_h != h,
                    next_h == h,
                ),
            ),
        ),
        next_h == h,
    )


def edge_x(n: int, x: int, y: int, z: int, ma: int, mi: int, md: int) -> int:
    if ma == 0 and mi == y:
        if md == 0:
            return z
        elif md == 1:
            return n - 1 - z
        elif md == 2:
            return n - 1 - x
    elif ma == 2 and mi == z:
        if md == 0:
            return y
        elif md == 1:
            return n - 1 - y
        elif md == 2:
            return n - 1 - x
    return x


def z3_edge_x(
    n: int,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_x: z3.ArithRef,
):
    return z3.If(
        z3.And(ma == 0, mi == y),
        z3.If(
            md == 0,
            next_x == z,
            z3.If(md == 1, next_x == (n - 1) - z, next_x == (n - 1) - x),
        ),
        z3.If(
            z3.And(ma == 2, mi == z),
            z3.If(
                md == 0,
                next_x == y,
                z3.If(md == 1, next_x == (n - 1) - y, next_x == (n - 1) - x),
            ),
            next_x == x,
        ),
    )


def edge_y(n: int, x: int, y: int, z: int, ma: int, mi: int, md: int) -> int:
    if ma == 1 and mi == x:
        if md == 0:
            return n - 1 - z
        elif md == 1:
            return z
        elif md == 2:
            return n - 1 - y
    elif ma == 2 and mi == z:
        if md == 0:
            return n - 1 - x
        elif md == 1:
            return x
        elif md == 2:
            return n - 1 - y
    return y


def z3_edge_y(
    n: int,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_y: z3.ArithRef,
):
    return z3.If(
        z3.And(ma == 1, mi == x),
        z3.If(
            md == 0,
            next_y == (n - 1) - z,
            z3.If(md == 1, next_y == z, next_y == (n - 1) - y),
        ),
        z3.If(
            z3.And(ma == 2, mi == z),
            z3.If(
                md == 0,
                next_y == (n - 1) - x,
                z3.If(md == 1, next_y == x, next_y == (n - 1) - y),
            ),
            next_y == y,
        ),
    )


def edge_z(n: int, x: int, y: int, z: int, ma: int, mi: int, md: int) -> int:
    if ma == 0 and mi == y:
        if md == 0:
            return n - 1 - x
        elif md == 1:
            return x
        elif md == 2:
            return n - 1 - z
    elif ma == 1 and mi == x:
        if md == 0:
            return y
        elif md == 1:
            return n - 1 - y
        elif md == 2:
            return n - 1 - z
    return z


def z3_edge_z(
    n: int,
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_z: z3.ArithRef,
):
    return z3.If(
        z3.And(ma == 0, mi == y),
        z3.If(
            md == 0,
            next_z == (n - 1) - x,
            z3.If(md == 1, next_z == x, next_z == (n - 1) - z),
        ),
        z3.If(
            z3.And(ma == 1, mi == x),
            z3.If(
                md == 0,
                next_z == y,
                z3.If(md == 1, next_z == (n - 1) - y, next_z == (n - 1) - z),
            ),
            next_z == z,
        ),
    )


def edge_r(x: int, y: int, z: int, r: bool, ma: int, mi: int, md: int) -> bool:
    if md != 2 and (
        (ma == 2 and mi == z)
        or (mi == 1 and ((ma == 0 and mi == y) or (ma == 1 and mi == x)))
    ):
        return not r
    return r


def z3_edge_r(
    x: z3.ArithRef,
    y: z3.ArithRef,
    z: z3.ArithRef,
    r: z3.BoolRef,
    ma: z3.ArithRef,
    mi: z3.ArithRef,
    md: z3.ArithRef,
    next_r: z3.BoolRef,
):
    return z3.If(
        z3.And(
            md != 2,
            z3.Or(
                z3.And(ma == 2, mi == z),
                z3.And(
                    mi == 1,
                    z3.Or(z3.And(ma == 0, mi == y), z3.And(ma == 1, mi == x)),
                ),
            ),
        ),
        next_r != r,
        next_r == r,
    )
