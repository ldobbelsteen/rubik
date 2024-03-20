import z3

import move_mappers


def z3_corner_x_hi(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    axs: list[z3.ArithRef],
    his: list[z3.BoolRef],
    drs: list[z3.ArithRef],
):
    assert len(axs) == len(his) == len(drs)
    assert len(axs) > 0
    if len(axs) == 1:
        return move_mappers.z3_corner_x_hi(x_hi, y_hi, z_hi, axs[0], his[0], drs[0])
    else:
        return move_mappers.z3_corner_x_hi(
            z3_corner_x_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_y_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_z_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            axs[-1],
            his[-1],
            drs[-1],
        )


def z3_corner_y_hi(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    axs: list[z3.ArithRef],
    his: list[z3.BoolRef],
    drs: list[z3.ArithRef],
):
    assert len(axs) == len(his) == len(drs)
    assert len(axs) > 0
    if len(axs) == 1:
        return move_mappers.z3_corner_y_hi(x_hi, y_hi, z_hi, axs[0], his[0], drs[0])
    else:
        return move_mappers.z3_corner_y_hi(
            z3_corner_x_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_y_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_z_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            axs[-1],
            his[-1],
            drs[-1],
        )


def z3_corner_z_hi(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    axs: list[z3.ArithRef],
    his: list[z3.BoolRef],
    drs: list[z3.ArithRef],
):
    assert len(axs) == len(his) == len(drs)
    assert len(axs) > 0
    if len(axs) == 1:
        return move_mappers.z3_corner_z_hi(x_hi, y_hi, z_hi, axs[0], his[0], drs[0])
    else:
        return move_mappers.z3_corner_z_hi(
            z3_corner_x_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_y_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_z_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            axs[-1],
            his[-1],
            drs[-1],
        )


def z3_corner_r(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    r: z3.ArithRef,
    cw: z3.BoolRef,
    axs: list[z3.ArithRef],
    his: list[z3.BoolRef],
    drs: list[z3.ArithRef],
):
    assert len(axs) == len(his) == len(drs)
    assert len(axs) > 0
    if len(axs) == 1:
        return move_mappers.z3_corner_r(x_hi, z_hi, r, cw, axs[0], his[0], drs[0])
    else:
        return move_mappers.z3_corner_r(
            z3_corner_x_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_z_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_r(x_hi, y_hi, z_hi, r, cw, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_cw(x_hi, y_hi, z_hi, cw, axs[:-1], his[:-1], drs[:-1]),
            axs[-1],
            his[-1],
            drs[-1],
        )


def z3_corner_cw(
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    z_hi: z3.BoolRef,
    cw: z3.BoolRef,
    axs: list[z3.ArithRef],
    his: list[z3.BoolRef],
    drs: list[z3.ArithRef],
):
    assert len(axs) == len(his) == len(drs)
    assert len(axs) > 0
    if len(axs) == 1:
        return move_mappers.z3_corner_cw(x_hi, y_hi, z_hi, cw, axs[0], his[0], drs[0])
    else:
        return move_mappers.z3_corner_cw(
            z3_corner_x_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_y_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_z_hi(x_hi, y_hi, z_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_corner_cw(x_hi, y_hi, z_hi, cw, axs[:-1], his[:-1], drs[:-1]),
            axs[-1],
            his[-1],
            drs[-1],
        )


def z3_edge_a(
    a: z3.ArithRef,
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    axs: list[z3.ArithRef],
    his: list[z3.BoolRef],
    drs: list[z3.ArithRef],
):
    assert len(axs) == len(his) == len(drs)
    assert len(axs) > 0
    if len(axs) == 1:
        return move_mappers.z3_edge_a(a, x_hi, y_hi, axs[0], his[0], drs[0])
    else:
        return move_mappers.z3_edge_a(
            z3_edge_a(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_edge_x_hi(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_edge_y_hi(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            axs[-1],
            his[-1],
            drs[-1],
        )


def z3_edge_x_hi(
    a: z3.ArithRef,
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    axs: list[z3.ArithRef],
    his: list[z3.BoolRef],
    drs: list[z3.ArithRef],
):
    assert len(axs) == len(his) == len(drs)
    assert len(axs) > 0
    if len(axs) == 1:
        return move_mappers.z3_edge_x_hi(a, x_hi, y_hi, axs[0], his[0], drs[0])
    else:
        return move_mappers.z3_edge_x_hi(
            z3_edge_a(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_edge_x_hi(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_edge_y_hi(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            axs[-1],
            his[-1],
            drs[-1],
        )


def z3_edge_y_hi(
    a: z3.ArithRef,
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    axs: list[z3.ArithRef],
    his: list[z3.BoolRef],
    drs: list[z3.ArithRef],
):
    assert len(axs) == len(his) == len(drs)
    assert len(axs) > 0
    if len(axs) == 1:
        return move_mappers.z3_edge_y_hi(a, x_hi, y_hi, axs[0], his[0], drs[0])
    else:
        return move_mappers.z3_edge_y_hi(
            z3_edge_a(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_edge_x_hi(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_edge_y_hi(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            axs[-1],
            his[-1],
            drs[-1],
        )


def z3_edge_r(
    a: z3.ArithRef,
    x_hi: z3.BoolRef,
    y_hi: z3.BoolRef,
    r: z3.BoolRef,
    axs: list[z3.ArithRef],
    his: list[z3.BoolRef],
    drs: list[z3.ArithRef],
):
    assert len(axs) == len(his) == len(drs)
    assert len(axs) > 0
    if len(axs) == 1:
        next_a = move_mappers.z3_edge_a(a, x_hi, y_hi, axs[0], his[0], drs[0])
        return move_mappers.z3_edge_r(a, next_a, r)
    else:
        return move_mappers.z3_edge_r(
            z3_edge_a(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
            z3_edge_a(
                z3_edge_a(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
                z3_edge_x_hi(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
                z3_edge_y_hi(a, x_hi, y_hi, axs[:-1], his[:-1], drs[:-1]),
                axs[:-2],
                his[:-2],
                drs[:-2],
            ),
            z3_edge_r(a, x_hi, y_hi, r, axs[:-1], his[:-1], drs[:-1]),
        )
