import json
import sys
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue
import z3
from puzzle import Puzzle, move_name, list_edges, list_centers, list_corners
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


def solve_for_k(puzzle: Puzzle, k: int):
    """Solve a puzzle with a maximum number of moves. Return list of move names or nothing if not possible.
    Also returns, in both cases, the time it took to prepare the SAT model and the time it took to solve it."""
    prep_start = datetime.now()
    solver = z3.Optimize()
    n = puzzle.n

    # Nested list representing the n × n × n cube of cubicles for each state.
    cubicle_states: list[
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

    # Populate the cubicle states with Z3 variables.
    for s in range(k + 1):
        for x, y, z in corners:
            cubicle_states[s][x][y][z] = (
                z3_int(solver, f"corner({x},{y},{z}) s({s}) x", 0, n),
                z3_int(solver, f"corner({x},{y},{z}) s({s}) y", 0, n),
                z3_int(solver, f"corner({x},{y},{z}) s({s}) z", 0, n),
                z3_int(solver, f"corner({x},{y},{z}) s({s}) r", 0, 3),
            )
        for x, y, z in centers:
            cubicle_states[s][x][y][z] = (
                z3_int(solver, f"center({x},{y},{z}) s({s}) x", 0, n),
                z3_int(solver, f"center({x},{y},{z}) s({s}) y", 0, n),
                z3_int(solver, f"center({x},{y},{z}) s({s}) z", 0, n),
                None,
            )
        for x, y, z in edges:
            cubicle_states[s][x][y][z] = (
                z3_int(solver, f"edge({x},{y},{z}) s({s}) x", 0, n),
                z3_int(solver, f"edge({x},{y},{z}) s({s}) y", 0, n),
                z3_int(solver, f"edge({x},{y},{z}) s({s}) z", 0, n),
                z3_int(solver, f"edge({x},{y},{z}) s({s}) r", 0, 2),
            )

    # Variables which together indicate the move at each state.
    mas = [z3_int(solver, f"s({s}) ma", 0, 3) for s in range(k)]
    mis = [z3_int(solver, f"s({s}) mi", 0, n + 1) for s in range(k)]
    mds = [z3_int(solver, f"s({s}) md", 0, 3) for s in range(k)]

    def cubicle(s: int, x: int, y: int, z: int):
        """Get a cubicle, assuming it exists."""
        cubicle = cubicle_states[s][x][y][z]
        if cubicle is None:
            raise Exception(f"invalid cubicle: {({x},{y},{z})}")
        return cubicle

    def fix_state(s: int, puzzle: Puzzle):
        """Return condition of a state being equal to a puzzle object."""
        conditions = []

        for x, y, z in corners:
            xv, yv, zv, rv = cubicle(s, x, y, z)
            conditions.append(xv == puzzle.coords[x][y][z][0])
            conditions.append(yv == puzzle.coords[x][y][z][1])
            conditions.append(zv == puzzle.coords[x][y][z][2])
            conditions.append(rv == puzzle.rotations[x][y][z])
        for x, y, z in centers:
            xv, yv, zv, rv = cubicle(s, x, y, z)
            conditions.append(xv == puzzle.coords[x][y][z][0])
            conditions.append(yv == puzzle.coords[x][y][z][1])
            conditions.append(zv == puzzle.coords[x][y][z][2])
        for x, y, z in edges:
            xv, yv, zv, rv = cubicle(s, x, y, z)
            conditions.append(xv == puzzle.coords[x][y][z][0])
            conditions.append(yv == puzzle.coords[x][y][z][1])
            conditions.append(zv == puzzle.coords[x][y][z][2])
            conditions.append(rv == puzzle.rotations[x][y][z])

        return z3.And(conditions)

    def identical_states(s1: int, s2: int):
        """Return condition of two states being equal."""
        conditions = []

        for x, y, z in corners:
            x1, y1, z1, r1 = cubicle(s1, x, y, z)
            x2, y2, z2, r2 = cubicle(s2, x, y, z)
            conditions.append(x1 == x2)
            conditions.append(y1 == y2)
            conditions.append(z1 == z2)
            conditions.append(r1 == r2)
        for x, y, z in centers:
            x1, y1, z1, r1 = cubicle(s1, x, y, z)
            x2, y2, z2, r2 = cubicle(s2, x, y, z)
            conditions.append(x1 == x2)
            conditions.append(y1 == y2)
            conditions.append(z1 == z2)
        for x, y, z in edges:
            x1, y1, z1, r1 = cubicle(s1, x, y, z)
            x2, y2, z2, r2 = cubicle(s2, x, y, z)
            conditions.append(x1 == x2)
            conditions.append(y1 == y2)
            conditions.append(z1 == z2)
            conditions.append(r1 == r2)

        return z3.And(conditions)

    # Fix the first state to the puzzle state.
    solver.add(fix_state(0, puzzle))

    # Fix the last state to the finished state.
    finished = Puzzle.finished(n)
    solver.add(fix_state(-1, finished))

    # Restrict color states when move is nothing.
    for s in range(k):
        solver.add(z3.Or(mis[s] != n, identical_states(s, s + 1)))

    # Only allow nothing move when complete.
    for s in range(k):
        solver.add(z3.Or(mis[s] != n, fix_state(s, finished)))

    # Restrict cubicle states according to moves.
    # NOTE: naive reference implementation
    for s in range(k):
        mav, miv, mdv = mas[s], mis[s], mds[s]

        for x, y, z in itertools.chain(corners, centers, edges):
            xv, yv, zv, _ = cubicle(s, x, y, z)
            next_xv, next_yv, next_zv, _ = cubicle(s + 1, x, y, z)

            # Restrictions for next x.
            solver.add(
                z3.If(
                    mav == 0,
                    z3.If(
                        miv == yv,
                        z3.If(
                            mdv == 0,
                            next_xv == zv,
                            z3.If(
                                mdv == 1,
                                next_xv == n - 1 - zv,
                                z3.If(mdv == 2, next_xv == n - 1 - xv, next_xv == xv),
                            ),
                        ),
                        next_xv == xv,
                    ),
                    z3.If(
                        mav == 1,
                        z3.If(
                            miv == xv,
                            z3.If(
                                mdv == 0,
                                next_xv == xv,
                                z3.If(
                                    mdv == 1,
                                    next_xv == xv,
                                    z3.If(mdv == 2, next_xv == xv, next_xv == xv),
                                ),
                            ),
                            next_xv == xv,
                        ),
                        z3.If(
                            mav == 2,
                            z3.If(
                                miv == zv,
                                z3.If(
                                    mdv == 0,
                                    next_xv == yv,
                                    z3.If(
                                        mdv == 1,
                                        next_xv == n - 1 - yv,
                                        z3.If(
                                            mdv == 2,
                                            next_xv == n - 1 - xv,
                                            next_xv == xv,
                                        ),
                                    ),
                                ),
                                next_xv == xv,
                            ),
                            next_xv == xv,
                        ),
                    ),
                )
            )

            # Restrictions for next y.
            solver.add(
                z3.If(
                    mav == 0,
                    z3.If(
                        miv == yv,
                        z3.If(
                            mdv == 0,
                            next_yv == yv,
                            z3.If(
                                mdv == 1,
                                next_yv == yv,
                                z3.If(mdv == 2, next_yv == yv, next_yv == yv),
                            ),
                        ),
                        next_yv == yv,
                    ),
                    z3.If(
                        mav == 1,
                        z3.If(
                            miv == xv,
                            z3.If(
                                mdv == 0,
                                next_yv == n - 1 - zv,
                                z3.If(
                                    mdv == 1,
                                    next_yv == zv,
                                    z3.If(
                                        mdv == 2, next_yv == n - 1 - xv, next_yv == yv
                                    ),
                                ),
                            ),
                            next_yv == yv,
                        ),
                        z3.If(
                            mav == 2,
                            z3.If(
                                miv == zv,
                                z3.If(
                                    mdv == 0,
                                    next_yv == n - 1 - xv,
                                    z3.If(
                                        mdv == 1,
                                        next_yv == xv,
                                        z3.If(
                                            mdv == 2,
                                            next_yv == n - 1 - yv,
                                            next_yv == yv,
                                        ),
                                    ),
                                ),
                                next_yv == yv,
                            ),
                            next_yv == yv,
                        ),
                    ),
                )
            )

            # Restrictions for next z.
            solver.add(
                z3.If(
                    mav == 0,
                    z3.If(
                        miv == yv,
                        z3.If(
                            mdv == 0,
                            next_zv == n - 1 - xv,
                            z3.If(
                                mdv == 1,
                                next_zv == xv,
                                z3.If(mdv == 2, next_zv == n - 1 - zv, next_zv == zv),
                            ),
                        ),
                        next_zv == zv,
                    ),
                    z3.If(
                        mav == 1,
                        z3.If(
                            miv == xv,
                            z3.If(
                                mdv == 0,
                                next_zv == yv,
                                z3.If(
                                    mdv == 1,
                                    next_zv == n - 1 - yv,
                                    z3.If(
                                        mdv == 2, next_zv == n - 1 - zv, next_zv == zv
                                    ),
                                ),
                            ),
                            next_zv == zv,
                        ),
                        z3.If(
                            mav == 2,
                            z3.If(
                                miv == zv,
                                z3.If(
                                    mdv == 0,
                                    next_zv == zv,
                                    z3.If(
                                        mdv == 1,
                                        next_zv == zv,
                                        z3.If(mdv == 2, next_zv == zv, next_zv == zv),
                                    ),
                                ),
                                next_zv == zv,
                            ),
                            next_zv == zv,
                        ),
                    ),
                )
            )

        for x, y, z in corners:
            xv, yv, zv, rv = cubicle(s, x, y, z)
            next_xv, next_yv, next_zv, next_rv = cubicle(s + 1, x, y, z)
            assert rv is not None and next_rv is not None

            # Restrictions for next r.
            solver.add(
                z3.If(
                    mav == 1,
                    z3.If(
                        miv == xv,
                        z3.If(
                            mdv != 2,
                            z3.If(
                                rv == 0,
                                next_rv == 1,
                                z3.If(
                                    rv == 1,
                                    next_rv == 2,
                                    z3.If(rv == 2, next_rv == 0, next_rv == rv),
                                ),
                            ),
                            next_rv == rv,
                        ),
                        next_rv == rv,
                    ),
                    z3.If(
                        mav == 2,
                        z3.If(
                            miv == zv,
                            z3.If(
                                mdv != 2,
                                z3.If(
                                    rv == 0,
                                    next_rv == 2,
                                    z3.If(
                                        rv == 1,
                                        next_rv == 0,
                                        z3.If(rv == 2, next_rv == 1, next_rv == rv),
                                    ),
                                ),
                                next_rv == rv,
                            ),
                            next_rv == rv,
                        ),
                        next_rv == rv,
                    ),
                ),
            )

        for x, y, z in edges:
            xv, yv, zv, rv = cubicle(s, x, y, z)
            next_xv, next_yv, next_zv, next_rv = cubicle(s + 1, x, y, z)
            assert rv is not None and next_rv is not None

            # Restrictions for next r.
            solver.add(
                z3.If(
                    mav == 2,
                    z3.If(
                        miv == zv,
                        z3.If(
                            mdv != 2,
                            z3.If(
                                rv == 0,
                                next_rv == 1,
                                z3.If(rv == 1, next_rv == 0, next_rv == rv),
                            ),
                            next_rv == rv,
                        ),
                        next_rv == rv,
                    ),
                    next_rv == rv,
                )
            )

    # Restrict cubicle states according to moves.
    # NOTE: optimized implementation
    # mappings = move_mapping.load(n)
    # for s in range(k):
    #     mav, miv, mdv = mas[s], mis[s], mds[s]

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
    #                     return xv
    #                 case "y":
    #                     return yv
    #                 case "z":
    #                     return zv
    #                 case "r":
    #                     assert rv is not None
    #                     return rv
    #                 case "ma":
    #                     return mav
    #                 case "mi":
    #                     return miv
    #                 case "md":
    #                     return mdv
    #                 case _:
    #                     raise Exception(f"invalid variable: {exp}")

    #     def convert_eq(left: str, eq: bool, right: int | str) -> bool | z3.BoolRef:
    #         if eq:
    #             return convert_exp(left) == convert_exp(right)
    #         else:
    #             return convert_exp(left) != convert_exp(right)

    #     # Add restrictions for all corners.
    #     for x, y, z in corners:
    #         xv, yv, zv, rv = cubicle(s, x, y, z)
    #         next_xv, next_yv, next_zv, next_rv = cubicle(s + 1, x, y, z)

    #         # Mappings for x-coordinates.
    #         for inputs, output in mappings["corner_coord"]["x_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     next_xv == convert_exp(output),
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
    #                     next_yv == convert_exp(output),
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
    #                     next_zv == convert_exp(output),
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
    #                     next_rv == convert_exp(output),
    #                 )
    #             )

    #     # Add restrictions for all centers.
    #     for x, y, z in centers:
    #         xv, yv, zv, rv = cubicle(s, x, y, z)
    #         next_xv, next_yv, next_zv, next_rv = cubicle(s + 1, x, y, z)

    #         # Mappings for x-coordinates.
    #         for inputs, output in mappings["center_coord"]["x_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     next_xv == convert_exp(output),
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
    #                     next_yv == convert_exp(output),
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
    #                     next_zv == convert_exp(output),
    #                 )
    #             )

    #     # Add restrictions for all edges.
    #     for x, y, z in edges:
    #         xv, yv, zv, rv = cubicle(s, x, y, z)
    #         next_xv, next_yv, next_zv, next_rv = cubicle(s + 1, x, y, z)

    #         # Mappings for x-coordinates.
    #         for inputs, output in mappings["edge_coord"]["x_new"]:
    #             conditions = [
    #                 convert_eq(input, eq, val) for input, (eq, val) in inputs.items()
    #             ]
    #             solver.add(
    #                 z3.Or(
    #                     z3.Or([z3.Not(cond) for cond in conditions]),
    #                     next_xv == convert_exp(output),
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
    #                     next_yv == convert_exp(output),
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
    #                     next_zv == convert_exp(output),
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
    #                     next_rv == convert_exp(output),
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
        puzzles = [Puzzle.from_file(file) for file in files]

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
                puzzle: Puzzle,
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


# e.g. python solve.py ./puzzles/n2-random7.txt
if __name__ == "__main__":
    solve([sys.argv[1]], cpu_count())
