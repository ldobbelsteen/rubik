import json
import sys
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue
import z3
from logic import State, move_name, list_edges, list_centers, list_corners
from misc import print_stamped
import move_mapping
import itertools


def k_upperbound(n: int):
    match n:
        case 1:
            return 0  # is always already solved
        case 2:
            return 11  # God's Number
        case 3:
            return 20  # God's Number
        case _:
            raise Exception(f"upperbound of {n} not set")


def z3_int(solver: z3.Optimize, name: str, low: int, high: int):
    """Create Z3 integer and add its value range to the solver. The range is
    inclusive on low and exclusive on high."""
    var = z3.Int(name)
    solver.add(var >= low)
    solver.add(var < high)
    return var


def solve_for_k(puzzle: State, k: int):
    """Solve a puzzle with a maximum number of moves. Return list of move names or nothing if not possible.
    Also returns, in both cases, the time it took to prepare the SAT model and the time it took to solve it."""
    prep_start = datetime.now()
    solver = z3.Optimize()
    n = puzzle.n

    # Nested list representing an n × n × n cube. The variables represent
    # the current location (or rotation if relevant) of that cubie in a state.
    states: list[
        list[
            list[
                list[
                    tuple[
                        z3.ArithRef,  # x-coordinate
                        z3.ArithRef,  # y-coordinate
                        z3.ArithRef,  # z-coordinate
                        z3.ArithRef | None,  # rotation
                    ]
                    | None  # internal cubie
                ]
            ]
        ]
    ] = [
        [[[None for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for _ in range(k + 1)
    ]

    corners = list_corners(n)
    centers = list_centers(n)
    edges = list_edges(n)

    # Populate the states with Z3 variables.
    for s in range(k + 1):
        for cx, cy, cz in corners:
            states[s][cx][cy][cz] = (
                z3_int(solver, f"corner({cx},{cy},{cz}) s({s}) x", 0, n),
                z3_int(solver, f"corner({cx},{cy},{cz}) s({s}) y", 0, n),
                z3_int(solver, f"corner({cx},{cy},{cz}) s({s}) z", 0, n),
                z3_int(solver, f"corner({cx},{cy},{cz}) s({s}) r", 0, 3),
            )
        for cx, cy, cz in centers:
            states[s][cx][cy][cz] = (
                z3_int(solver, f"center({cx},{cy},{cz}) s({s}) x", 0, n),
                z3_int(solver, f"center({cx},{cy},{cz}) s({s}) y", 0, n),
                z3_int(solver, f"center({cx},{cy},{cz}) s({s}) z", 0, n),
                None,
            )
        for cx, cy, cz in edges:
            states[s][cx][cy][cz] = (
                z3_int(solver, f"edge({cx},{cy},{cz}) s({s}) x", 0, n),
                z3_int(solver, f"edge({cx},{cy},{cz}) s({s}) y", 0, n),
                z3_int(solver, f"edge({cx},{cy},{cz}) s({s}) z", 0, n),
                z3_int(solver, f"edge({cx},{cy},{cz}) s({s}) r", 0, 3),
            )

    # Variables which together indicate the move at each step. NOTE: nothing move disabled
    mas = [z3_int(solver, f"s({s}) ma", 0, 2) for s in range(k)]
    mis = [z3_int(solver, f"s({s}) mi", 0, n) for s in range(k)]
    mds = [z3_int(solver, f"s({s}) md", 0, 2) for s in range(k)]

    def cubie(s: int, x: int, y: int, z: int):
        """Fetch a cubie from the states list."""
        cubie = states[s][x][y][z]
        assert cubie is not None
        return cubie

    def fix_state(s: int, state: State):
        """Return condition of a state being equal to a state object."""
        conditions = []

        for cx, cy, cz in corners:
            x, y, z, r = cubie(s, cx, cy, cz)
            conditions.append(x == state.coords[cx][cy][cz][0])
            conditions.append(y == state.coords[cx][cy][cz][1])
            conditions.append(z == state.coords[cx][cy][cz][2])
            conditions.append(r == state.rots[cx][cy][cz])

        for cx, cy, cz in centers:
            x, y, z, r = cubie(s, cx, cy, cz)
            conditions.append(x == state.coords[cx][cy][cz][0])
            conditions.append(y == state.coords[cx][cy][cz][1])
            conditions.append(z == state.coords[cx][cy][cz][2])

        for cx, cy, cz in edges:
            x, y, z, r = cubie(s, cx, cy, cz)
            conditions.append(x == state.coords[cx][cy][cz][0])
            conditions.append(y == state.coords[cx][cy][cz][1])
            conditions.append(z == state.coords[cx][cy][cz][2])
            conditions.append(r == state.rots[cx][cy][cz])

        return z3.And(conditions)

    def identical_states(s1: int, s2: int):
        """Return condition of two states being equal."""
        conditions = []

        for cx, cy, cz in corners:
            x1, y1, z1, r1 = cubie(s1, cx, cy, cz)
            x2, y2, z2, r2 = cubie(s2, cx, cy, cz)
            conditions.append(x1 == x2)
            conditions.append(y1 == y2)
            conditions.append(z1 == z2)
            conditions.append(r1 == r2)

        for cx, cy, cz in centers:
            x1, y1, z1, r1 = cubie(s1, cx, cy, cz)
            x2, y2, z2, r2 = cubie(s2, cx, cy, cz)
            conditions.append(x1 == x2)
            conditions.append(y1 == y2)
            conditions.append(z1 == z2)

        for cx, cy, cz in edges:
            x1, y1, z1, r1 = cubie(s1, cx, cy, cz)
            x2, y2, z2, r2 = cubie(s2, cx, cy, cz)
            conditions.append(x1 == x2)
            conditions.append(y1 == y2)
            conditions.append(z1 == z2)
            conditions.append(r1 == r2)

        return z3.And(conditions)

    # Fix the first state to the puzzle state.
    solver.add(fix_state(0, puzzle))

    # Fix the last state to a finished state.
    finished = State.finished(n)
    solver.add(fix_state(-1, finished))

    # Restrict color states when move is nothing.
    for s in range(k):
        solver.add(z3.Or(mis[s] != n, identical_states(s, s + 1)))

    # Only allow nothing move when complete.
    for s in range(k):
        solver.add(z3.Or(mis[s] != n, fix_state(s, finished)))

    # NOTE: temporary reference move mapping
    for s in range(k):
        ma, mi, md = mas[s], mis[s], mds[s]

        for cx, cy, cz in itertools.chain(corners, centers, edges):
            x, y, z, _ = cubie(s, cx, cy, cz)
            new_x, new_y, new_z, _ = cubie(s + 1, cx, cy, cz)

            # Restrictions for new_x.
            solver.add(
                z3.If(
                    ma == 0,
                    z3.If(
                        mi == y,
                        z3.If(
                            md == 0,
                            new_x == z,
                            z3.If(
                                md == 1,
                                new_x == n - 1 - z,
                                z3.If(md == 2, new_x == n - 1 - x, new_x == x),
                            ),
                        ),
                        new_x == x,
                    ),
                    z3.If(
                        ma == 1,
                        z3.If(
                            mi == x,
                            z3.If(
                                md == 0,
                                new_x == x,
                                z3.If(
                                    md == 1,
                                    new_x == x,
                                    z3.If(md == 2, new_x == x, new_x == x),
                                ),
                            ),
                            new_x == x,
                        ),
                        z3.If(
                            ma == 2,
                            z3.If(
                                mi == z,
                                z3.If(
                                    md == 0,
                                    new_x == y,
                                    z3.If(
                                        md == 1,
                                        new_x == n - 1 - y,
                                        z3.If(md == 2, new_x == n - 1 - x, new_x == x),
                                    ),
                                ),
                                new_x == x,
                            ),
                            new_x == x,
                        ),
                    ),
                )
            )

            # Restrictions for new_y.
            solver.add(
                z3.If(
                    ma == 0,
                    z3.If(
                        mi == y,
                        z3.If(
                            md == 0,
                            new_y == y,
                            z3.If(
                                md == 1,
                                new_y == y,
                                z3.If(md == 2, new_y == y, new_y == y),
                            ),
                        ),
                        new_y == y,
                    ),
                    z3.If(
                        ma == 1,
                        z3.If(
                            mi == x,
                            z3.If(
                                md == 0,
                                new_y == n - 1 - z,
                                z3.If(
                                    md == 1,
                                    new_y == z,
                                    z3.If(md == 2, new_y == n - 1 - x, new_y == y),
                                ),
                            ),
                            new_y == y,
                        ),
                        z3.If(
                            ma == 2,
                            z3.If(
                                mi == z,
                                z3.If(
                                    md == 0,
                                    new_y == n - 1 - x,
                                    z3.If(
                                        md == 1,
                                        new_y == x,
                                        z3.If(md == 2, new_y == n - 1 - y, new_y == y),
                                    ),
                                ),
                                new_y == y,
                            ),
                            new_y == y,
                        ),
                    ),
                )
            )

            # Restrictions for new_z.
            solver.add(
                z3.If(
                    ma == 0,
                    z3.If(
                        mi == y,
                        z3.If(
                            md == 0,
                            new_z == n - 1 - x,
                            z3.If(
                                md == 1,
                                new_z == x,
                                z3.If(md == 2, new_z == n - 1 - z, new_z == z),
                            ),
                        ),
                        new_z == z,
                    ),
                    z3.If(
                        ma == 1,
                        z3.If(
                            mi == x,
                            z3.If(
                                md == 0,
                                new_z == y,
                                z3.If(
                                    md == 1,
                                    new_z == n - 1 - y,
                                    z3.If(md == 2, new_z == n - 1 - z, new_z == z),
                                ),
                            ),
                            new_z == z,
                        ),
                        z3.If(
                            ma == 2,
                            z3.If(
                                mi == z,
                                z3.If(
                                    md == 0,
                                    new_z == z,
                                    z3.If(
                                        md == 1,
                                        new_z == z,
                                        z3.If(md == 2, new_z == z, new_z == z),
                                    ),
                                ),
                                new_z == z,
                            ),
                            new_z == z,
                        ),
                    ),
                )
            )

            # # Restrictions for new_x
            # solver.add(z3.Or(z3.Or([ma != 0, mi != y, md != 0]), new_x == z))
            # solver.add(z3.Or(z3.Or([ma != 0, mi != y, md != 1]), new_x == n - 1 - z))
            # solver.add(z3.Or(z3.Or([ma != 0, mi != y, md != 2]), new_x == n - 1 - x))
            # solver.add(z3.Or(z3.Or([ma != 1, mi != x]), new_x == x))
            # solver.add(z3.Or(z3.Or([ma != 2, mi != z, md != 0]), new_x == y))
            # solver.add(z3.Or(z3.Or([ma != 2, mi != z, md != 1]), new_x == n - 1 - y))
            # solver.add(z3.Or(z3.Or([ma != 2, mi != z, md != 2]), new_x == n - 1 - x))

            # # Restrictions for new_y
            # solver.add(z3.Or(z3.Or([ma != 0, mi != y]), new_y == y))
            # solver.add(z3.Or(z3.Or([ma != 1, mi != x, md != 0]), new_y == n - 1 - z))
            # solver.add(z3.Or(z3.Or([ma != 1, mi != x, md != 1]), new_y == z))
            # solver.add(z3.Or(z3.Or([ma != 1, mi != x, md != 2]), new_y == n - 1 - y))
            # solver.add(z3.Or(z3.Or([ma != 2, mi != z, md != 0]), new_y == n - 1 - x))
            # solver.add(z3.Or(z3.Or([ma != 2, mi != z, md != 1]), new_y == x))
            # solver.add(z3.Or(z3.Or([ma != 2, mi != z, md != 2]), new_y == n - 1 - y))

            # # Restrictions for new_z
            # solver.add(z3.Or(z3.Or([ma != 0, mi != y, md != 0]), new_z == n - 1 - x))
            # solver.add(z3.Or(z3.Or([ma != 0, mi != y, md != 1]), new_z == x))
            # solver.add(z3.Or(z3.Or([ma != 0, mi != y, md != 2]), new_z == n - 1 - z))
            # solver.add(z3.Or(z3.Or([ma != 1, mi != x, md != 0]), new_z == y))
            # solver.add(z3.Or(z3.Or([ma != 1, mi != x, md != 1]), new_z == n - 1 - y))
            # solver.add(z3.Or(z3.Or([ma != 1, mi != x, md != 2]), new_z == n - 1 - z))
            # solver.add(z3.Or(z3.Or([ma != 2, mi != z]), new_z == z))

        for cx, cy, cz in itertools.chain(corners, edges):
            x, y, z, r = cubie(s, cx, cy, cz)
            new_x, new_y, new_z, new_r = cubie(s + 1, cx, cy, cz)
            assert r is not None and new_r is not None

            solver.add(
                z3.If(
                    ma == 0,
                    z3.If(
                        mi == y,
                        z3.If(
                            md != 2,
                            z3.If(
                                r == 0,
                                new_r == 2,
                                z3.If(r == 2, new_r == 0, new_r == r),
                            ),
                            new_r == r,
                        ),
                        new_r == r,
                    ),
                    z3.If(
                        ma == 1,
                        z3.If(
                            mi == x,
                            z3.If(
                                md != 2,
                                z3.If(
                                    r == 1,
                                    new_r == 2,
                                    z3.If(r == 2, new_r == 1, new_r == r),
                                ),
                                new_r == r,
                            ),
                            new_r == r,
                        ),
                        z3.If(
                            ma == 2,
                            z3.If(
                                mi == z,
                                z3.If(
                                    md != 2,
                                    z3.If(
                                        r == 0,
                                        new_r == 1,
                                        z3.If(r == 1, new_r == 0, new_r == r),
                                    ),
                                    new_r == r,
                                ),
                                new_r == r,
                            ),
                            new_r == r,
                        ),
                    ),
                )
            )

            # # Restrictions for new_r
            # solver.add(z3.Or(z3.Or([ma != 0, mi != y, md == 2, r != 0]), new_r == 2))
            # solver.add(z3.Or(z3.Or([ma != 0, mi != y, md == 2, r != 2]), new_r == 0))
            # solver.add(z3.Or(z3.Or([ma != 1, mi != x, md == 2, r != 1]), new_r == 2))
            # solver.add(z3.Or(z3.Or([ma != 1, mi != x, md == 2, r != 2]), new_r == 1))
            # solver.add(z3.Or(z3.Or([ma != 2, mi != z, md == 2, r != 0]), new_r == 1))
            # solver.add(z3.Or(z3.Or([ma != 2, mi != z, md == 2, r != 1]), new_r == 0))

    # NOTE: temporarily commented out for reference testing
    # # Restrict color states using pre-generated move mappings file.
    # mappings = move_mapping.load(n)
    # for s in range(k):
    #     ma, mi, md = mas[s], mis[s], mds[s]

    #     def convert_exp(exp: int | str) -> int | z3.ArithRef:
    #         if isinstance(exp, int):
    #             return exp
    #         elif exp.isnumeric():
    #             return int(exp)
    #         elif "-" in exp:
    #             left, right = exp.split("-")
    #             return convert_exp(left.strip()) - convert_exp(right.strip())
    #         elif "+" in exp:
    #             left, right = exp.split("+")
    #             return convert_exp(left.strip()) + convert_exp(right.strip())
    #         else:
    #             match exp:
    #                 case "x":
    #                     return x
    #                 case "y":
    #                     return y
    #                 case "z":
    #                     return z
    #                 case "r":
    #                     assert r is not None
    #                     return r
    #                 case "ma":
    #                     return ma
    #                 case "mi":
    #                     return mi
    #                 case "md":
    #                     return md
    #                 case _:
    #                     raise Exception(f"invalid variable: {exp}")

    #     def convert_eq(left: str, eq: bool, right: int | str) -> bool | z3.BoolRef:
    #         if eq:
    #             return convert_exp(left) == convert_exp(right)
    #         else:
    #             return convert_exp(left) != convert_exp(right)

    #     # Add restrictions for all corners.
    #     for cx, cy, cz in corners:
    #         x, y, z, r = cubie(s, cx, cy, cz)
    #         new_x, new_y, new_z, new_r = cubie(s + 1, cx, cy, cz)

    #         # Mappings for x-coordinates.
    #         for inputs, output in mappings["corner_coord"]["x_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_x == convert_exp(output),
    #                 )
    #             )

    #         # Mappings for y-coordinates.
    #         for inputs, output in mappings["corner_coord"]["y_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_y == convert_exp(output),
    #                 )
    #             )

    #         # Mappings for z-coordinates.
    #         for inputs, output in mappings["corner_coord"]["z_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_z == convert_exp(output),
    #                 )
    #             )

    #         # Mappings for rotation.
    #         for inputs, output in mappings["corner_rotation"]["r_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_r == convert_exp(output),
    #                 )
    #             )

    #     # Add restrictions for all centers.
    #     for cx, cy, cz in centers:
    #         x, y, z, _ = cubie(s, cx, cy, cz)
    #         new_x, new_y, new_z, _ = cubie(s + 1, cx, cy, cz)

    #         # Mappings for x-coordinates.
    #         for inputs, output in mappings["center_coord"]["x_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_x == convert_exp(output),
    #                 )
    #             )

    #         # Mappings for y-coordinates.
    #         for inputs, output in mappings["center_coord"]["y_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_y == convert_exp(output),
    #                 )
    #             )

    #         # Mappings for z-coordinates.
    #         for inputs, output in mappings["center_coord"]["z_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_z == convert_exp(output),
    #                 )
    #             )

    #     # Add restrictions for all edges.
    #     for cx, cy, cz in edges:
    #         x, y, z, r = cubie(s, cx, cy, cz)
    #         new_x, new_y, new_z, new_r = cubie(s + 1, cx, cy, cz)

    #         # Mappings for x-coordinates.
    #         for inputs, output in mappings["edge_coord"]["x_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_x == convert_exp(output),
    #                 )
    #             )

    #         # Mappings for y-coordinates.
    #         for inputs, output in mappings["edge_coord"]["y_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_y == convert_exp(output),
    #                 )
    #             )

    #         # Mappings for z-coordinates.
    #         for inputs, output in mappings["edge_coord"]["z_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_z == convert_exp(output),
    #                 )
    #             )

    #         # Mappings for rotation.
    #         for inputs, output in mappings["edge_rotation"]["r_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     new_r == convert_exp(output),
    #                 )
    #             )

    # If between 1 and n moves ago we made a turn at an index and axis, a different axis has to have been turned in the meantime.
    for s in range(1, k):
        for rs in range(1, min(n + 1, s + 1)):
            solver.add(
                z3.Or(
                    mas[s - rs] != mas[s],
                    mis[s - rs] != mis[s],
                    z3.Or([mas[s] != mas[p] for p in range(s - rs + 1, s)]),
                )
            )

    # All subsequent moves in the same axis happen in ascending order of index.
    for s in range(1, k):
        solver.add(
            z3.Or(
                mas[s - 1] != mas[s],
                z3.And(
                    [
                        z3.And(
                            [z3.Or(mis[s - 1] != b, mis[s] != c) for c in range(1, b)]
                        )
                        for b in range(n, 1, -1)
                    ]
                ),
            )
        )

    # States cannot be repeated.
    for s1 in range(k + 1):
        for s2 in range(s1 + 1, k + 1):
            solver.add(z3.Not(identical_states(s1, s2)))

    # Check model and return moves if sat.
    prep_time = datetime.now() - prep_start
    solve_start = datetime.now()
    res = solver.check()
    solve_time = datetime.now() - solve_start
    moves: list[str] | None = None

    if res == z3.sat:
        moves = []
        model = solver.model()
        for s in range(k):
            moves.append(
                move_name(
                    n,
                    model.get_interp(mas[s]).as_long(),
                    model.get_interp(mis[s]).as_long(),
                    model.get_interp(mds[s]).as_long(),
                )
            )
        assert len(moves) == k

    return moves, prep_time, solve_time


def solve(files: list[str], process_count: int):
    """Solve a list of puzzles, efficiently distributing tasks among multiple processes."""

    with Manager() as manager:
        # List of puzzles to solve.
        puzzles = [State.from_str(open(file, "r").read()) for file in files]

        # List of n values for each of the puzzles.
        ns = [puzzles[i].n for i in range(len(puzzles))]

        # Generate any missing move mappings.
        for n in ns:
            move_mapping.generate(n)

        # List of upperbounds for k for each of the puzzles.
        k_upperbounds = [k_upperbound(ns[i]) for i in range(len(puzzles))]

        # Lists of prospects for k for each of the puzzles.
        k_prospects = [list(range(k_upperbounds[i] + 1)) for i in range(len(puzzles))]

        # List of currently found minimum size solutions for each of the puzzles.
        k_minima: list[list[str] | None] = [None for _ in range(len(puzzles))]

        # List of prep times for each tried prospect for each puzzle.
        prep_times: list[dict[int, timedelta]] = [{} for _ in range(len(puzzles))]

        # List of solving times for each tried prospect for each puzzle.
        solve_times: list[dict[int, timedelta]] = [{} for _ in range(len(puzzles))]

        # List of processes and their current tasks.
        processes: list[tuple[int, int, Process]] = []

        # Queue to output results onto.
        output: Queue[tuple[int, int, list[str] | None, timedelta, timedelta]] = (
            manager.Queue()
        )

        def spawn_new_process():
            def solve_wrapper(
                puzzle: State,
                k: int,
                i: int,
                output: Queue[tuple[int, int, list[str] | None, timedelta, timedelta]],
            ):
                solution, prep_time, solve_time = solve_for_k(puzzle, k)
                output.put((i, k, solution, prep_time, solve_time))

            for i in range(len(puzzles)):
                if len(k_prospects[i]) > 0:
                    k = k_prospects[i].pop(0)
                    process = Process(
                        target=solve_wrapper,
                        args=(
                            puzzles[i],
                            k,
                            i,
                            output,
                        ),
                    )
                    processes.append((i, k, process))
                    process.start()
                    break

        for _ in range(process_count):
            spawn_new_process()

        while len(processes) > 0:
            i, k, solution, prep_time, solve_time = output.get()
            prep_times[i][k] = prep_time
            solve_times[i][k] = solve_time

            if solution is None:
                print_stamped(
                    f"{files[i]}: unsat for k = {k} found in {solve_time} with {prep_time} prep time..."
                )
                k_prospects[i] = [p for p in k_prospects[i] if p > k]
                killed = 0
                for pi in range(len(processes) - 1, -1, -1):
                    if processes[pi][1] <= k:
                        processes.pop(pi)[2].kill()
                        killed += 1
                for _ in range(killed):
                    spawn_new_process()
            else:
                print_stamped(
                    f"{files[i]}: sat for k = {k} found in {solve_time} with {prep_time} prep time..."
                )
                current_minimum = k_minima[i]
                if current_minimum is None or k < len(current_minimum):
                    k_minima[i] = solution
                k_prospects[i] = [p for p in k_prospects[i] if p < k]
                killed = 0
                for pi in range(len(processes) - 1, -1, -1):
                    if processes[pi][1] >= k:
                        processes.pop(pi)[2].kill()
                        killed += 1
                for _ in range(killed):
                    spawn_new_process()

            if all([i != pi for pi, _, _ in processes]):
                minimum = k_minima[i]
                total_solve_time = sum(solve_times[i].values(), timedelta())
                total_prep_time = sum(prep_times[i].values(), timedelta())

                if minimum is None:
                    print_stamped(
                        f"{files[i]}: found no solution with k ≤ {k_upperbounds[i]} to be possible in {total_solve_time} with {total_prep_time} prep time"
                    )
                else:
                    print_stamped(
                        f"{files[i]}: minimum k = {len(minimum)} found in {total_solve_time} with {total_prep_time} prep time"
                    )

                result = {
                    "k": len(minimum) if minimum is not None else "n/a",
                    "moves": minimum if minimum is not None else "impossible",
                    "total_solve_time": str(total_solve_time),
                    "total_prep_time": str(total_prep_time),
                    "prep_time_per_k": {
                        k: str(t) for k, t in sorted(prep_times[i].items())
                    },
                    "solve_time_per_k": {
                        k: str(t) for k, t in sorted(solve_times[i].items())
                    },
                    "k_upperbound": k_upperbounds[i],
                    "process_count": process_count,
                }

                with open(f"{files[i]}.solution", "w") as solution_file:
                    solution_file.write(json.dumps(result, indent=4))


# e.g. python solve.py ./puzzles/n2-random10.txt
if __name__ == "__main__":
    solve([sys.argv[1]], cpu_count())
