import json
import sys
from datetime import datetime, timedelta
from multiprocessing import Manager, Process, cpu_count
from queue import Queue
import z3
from misc import print_with_stamp
import move_mapping


def k_upperbound(n: int):
    match n:
        case 1:
            return 0  # is always already solved
        case 2:
            return 11  # God's Number
        case 3:
            return 20  # God's Number
        case 4:
            return 35  # guestimate
        case _:
            raise Exception(f"upperbound of {n} not set")


def face_name(f: int) -> str:
    """Convert a face index to its canonical name."""
    match f:
        case 0:
            return "front"
        case 1:
            return "right"
        case 2:
            return "back"
        case 3:
            return "left"
        case 4:
            return "top"
        case 5:
            return "bottom"
        case _:
            return "unknown"


def color_name(c: int) -> str:
    """Convert a color index to its canonical name."""
    match c:
        case 0:
            return "white"
        case 1:
            return "green"
        case 2:
            return "yellow"
        case 3:
            return "blue"
        case 4:
            return "red"
        case 5:
            return "orange"
        case _:
            return "unknown"


def move_name(n: int, ma: int, mi: int, md: int) -> str:
    """Convert a move to its canonical name."""
    if mi == n:
        return "nothing"
    if ma == 0:
        if md == 0:
            return f"quarter row {mi} left"
        elif md == 1:
            return f"quarter row {mi} right"
        elif md == 2:
            return f"half row {mi}"
    elif ma == 1:
        if md == 0:
            return f"quarter column {mi} up"
        elif md == 1:
            return f"quarter column {mi} down"
        elif md == 2:
            return f"half column {mi}"
    elif ma == 2:
        if md == 0:
            return f"quarter layer {mi} clockwise"
        elif md == 1:
            return f"quarter layer {mi} counterclockwise"
        elif md == 2:
            return f"half layer {mi}"
    return "unknown"


def extract_state(
    n: int, model: z3.ModelRef, state: list[list[list[z3.ArithRef]]]
) -> list[list[list[int]]]:
    """Extract the colors in a state from a model."""
    return [
        [
            [model.get_interp(state[f][y][x]).as_long() for x in range(n)]
            for y in range(n)
        ]
        for f in range(6)
    ]


def solve(starting_state: list[list[list[int]]], k: int):
    """Solve a puzzle with a maximum number of moves. Return list of move names or nothing if not possible."""
    n = len(starting_state[0])
    solver = z3.Optimize()

    def cell_idx(f: int, y: int, x: int):
        """Convert the coordinates on a cube to a flat index."""
        return x + y * n + f * n * n

    # Variables indicating the colors of the cube cells at each state.
    colors = [[z3.Int(f"c({s},{i})") for i in range(6 * n * n)] for s in range(k + 1)]

    # Variables which together indicate the move at each state.
    move_axes = [z3.Int(f"ma({s})") for s in range(len(colors) - 1)]
    move_indices = [z3.Int(f"mi({s})") for s in range(len(colors) - 1)]
    move_directions = [z3.Int(f"md({s})") for s in range(len(colors) - 1)]

    # Restrict color domains to the six colors.
    for s in range(len(colors)):
        for f in range(6):
            for y in range(n):
                for x in range(n):
                    solver.add(
                        z3.And(
                            colors[s][cell_idx(f, y, x)] >= 0,
                            colors[s][cell_idx(f, y, x)] < 6,
                        )
                    )

    # Restrict colors of first state to starting state.
    for f in range(6):
        for y in range(n):
            for x in range(n):
                solver.add(colors[0][cell_idx(f, y, x)] == starting_state[f][y][x])

    def is_complete(s: int):
        """Return restriction on whether a state is complete."""
        return z3.And(
            [
                colors[s][cell_idx(f, y, x)] == colors[s][cell_idx(f, 0, 0)]
                for f in range(6)
                for y in range(n)
                for x in range(n)
            ]
        )

    # Restrict cube to be complete at the end.
    solver.add(is_complete(-1))

    # Restrict moves to valid moves.
    for s in range(len(colors) - 1):
        solver.add(z3.And(move_axes[s] >= 0, move_axes[s] <= 2))
        solver.add(z3.And(move_indices[s] >= 0, move_indices[s] <= n))
        solver.add(z3.And(move_directions[s] >= 0, move_directions[s] <= 2))

    # Restrict color states when move is nothing.
    for s in range(len(colors) - 1):
        solver.add(
            z3.Or(
                move_indices[s] != n,
                z3.And(
                    [
                        colors[s][cell_idx(f, y, x)] == colors[s + 1][cell_idx(f, y, x)]
                        for f in range(6)
                        for y in range(n)
                        for x in range(n)
                    ]
                ),
            )
        )

    # Only allow nothing move when complete.
    for s in range(len(colors) - 1):
        solver.add(z3.Or(move_indices[s] != n, is_complete(s)))

    # Restrict color states using pre-generated move mappings file.
    mappings = move_mapping.load(n)
    for s in range(len(colors) - 1):
        for ma in mappings:
            for mi in mappings[ma]:
                for md in mappings[ma][mi]:
                    conditions = []
                    if ma is not None:
                        conditions.append(move_axes[s] == ma)
                    if mi is not None:
                        conditions.append(move_indices[s] == mi)
                    if md is not None:
                        conditions.append(move_directions[s] == md)

                    consequences = [
                        colors[s + 1][cell_idx(f_in, y_in, x_in)]
                        == colors[s][cell_idx(f_out, y_out, x_out)]
                        for (f_in, y_in, x_in), (f_out, y_out, x_out) in mappings[ma][
                            mi
                        ][md]
                    ]

                    if len(consequences) > 0:
                        solver.add(
                            z3.Or(
                                z3.Or([z3.Not(cond) for cond in conditions]),
                                z3.And(consequences),
                            )
                        )

    # If between 1 and n moves ago we made a turn at an index and axis, a different axis has to have been turned in the meantime.
    for s in range(1, len(colors) - 1):
        for r in range(1, min(n + 1, s + 1)):
            solver.add(
                z3.Or(
                    move_axes[s - r] != move_axes[s],
                    move_indices[s - r] != move_indices[s],
                    z3.Or([move_axes[s] != move_axes[p] for p in range(s - r + 1, s)]),
                )
            )

    # All subsequent moves in the same axis happen in ascending order of index.
    for s in range(1, len(colors) - 1):
        solver.add(
            z3.Or(
                move_axes[s - 1] != move_axes[s],
                z3.And(
                    [
                        z3.And(
                            [
                                z3.Or(move_indices[s - 1] != b, move_indices[s] != c)
                                for c in range(1, b)
                            ]
                        )
                        for b in range(n, 1, -1)
                    ]
                ),
            )
        )

    # States cannot be repeated.
    for s1 in range(len(colors)):
        for s2 in range(s1 + 1, len(colors)):
            solver.add(
                z3.Not(
                    z3.And(
                        [
                            colors[s1][cell_idx(f, y, x)]
                            == colors[s2][cell_idx(f, y, x)]
                            for f in range(6)
                            for y in range(n)
                            for x in range(n)
                        ]
                    )
                )
            )

    # Check model and return moves if sat.
    res = solver.check()
    if res == z3.sat:
        moves: list[str] = []
        model = solver.model()
        for s in range(len(colors) - 1):
            ma = model.get_interp(move_axes[s]).as_long()
            mi = model.get_interp(move_indices[s]).as_long()
            md = model.get_interp(move_directions[s]).as_long()
            moves.append(move_name(n, ma, mi, md))
        assert len(moves) == k
        return moves


def solve_puzzles(files: list[str], process_count: int):
    """Solve a list of puzzles, efficiently distributing tasks among multiple processes."""
    with Manager() as manager:
        # List of puzzles to solve
        puzzles: list[list[list[list[int]]]] = [
            eval(open(file, "r").read()) for file in files
        ]

        # List of n values for each of the puzzles.
        ns = [len(puzzles[i][0]) for i in range(len(puzzles))]

        # Generate any missing move mappings.
        for n in ns:
            move_mapping.generate(n)

        # List of upperbounds for k for each of the puzzles.
        k_upperbounds = [k_upperbound(ns[i]) for i in range(len(puzzles))]

        # Lists of prospects for k for each of the puzzles.
        k_prospects = [list(range(k_upperbounds[i] + 1)) for i in range(len(puzzles))]

        # List of minimum size solutions for each of the puzzles.
        k_minima: list[list[str] | None] = [None for _ in range(len(puzzles))]

        # List of cpu times for each tried prospect for each puzzle.
        cpu_times: list[dict[int, timedelta]] = [{} for _ in range(len(puzzles))]

        # List of processes and their current tasks.
        processes: list[tuple[int, int, Process]] = []

        # Queue to output results onto.
        output: Queue[tuple[int, int, datetime, list[str] | None]] = manager.Queue()

        def spawn_new_process():
            def solve_wrapper(
                starting_state: list[list[list[int]]],
                k: int,
                puzzle_index: int,
                start_time: datetime,
                output: Queue[tuple[int, int, datetime, list[str] | None]],
            ):
                solution = solve(starting_state, k)
                output.put((puzzle_index, k, start_time, solution))

            for puzzle_index in range(len(puzzles)):
                if len(k_prospects[puzzle_index]) > 0:
                    median = (
                        len(k_prospects[puzzle_index]) // 2
                    )  # Take median prospect, resulting in a kind of binary search.
                    k = k_prospects[puzzle_index].pop(median)
                    process = Process(
                        target=solve_wrapper,
                        args=(
                            puzzles[puzzle_index],
                            k,
                            puzzle_index,
                            datetime.now(),
                            output,
                        ),
                    )
                    processes.append((puzzle_index, k, process))
                    process.start()
                    break

        for _ in range(process_count):
            spawn_new_process()

        while len(processes) > 0:
            puzzle_index, k, start, minimum = output.get()
            cpu_time = datetime.now() - start
            cpu_times[puzzle_index][k] = cpu_time
            if minimum is None:
                print_with_stamp(
                    f"{files[puzzle_index]}: unsat for k = {k} found in {cpu_time}..."
                )
                k_prospects[puzzle_index] = [
                    p for p in k_prospects[puzzle_index] if p > k
                ]
                killed = 0
                for pi in range(len(processes) - 1, -1, -1):
                    if processes[pi][1] <= k:
                        processes.pop(pi)[2].kill()
                        killed += 1
                for _ in range(killed):
                    spawn_new_process()
            else:
                print_with_stamp(
                    f"{files[puzzle_index]}: sat for k = {k} found in {cpu_time}..."
                )
                current_minimum = k_minima[puzzle_index]
                if current_minimum is None or k < len(current_minimum):
                    k_minima[puzzle_index] = minimum
                k_prospects[puzzle_index] = [
                    p for p in k_prospects[puzzle_index] if p < k
                ]
                killed = 0
                for pi in range(len(processes) - 1, -1, -1):
                    if processes[pi][1] >= k:
                        processes.pop(pi)[2].kill()
                        killed += 1
                for _ in range(killed):
                    spawn_new_process()
            if all([puzzle_index != pi for pi, _, _ in processes]):
                minimum = k_minima[puzzle_index]
                total_cpu_time = sum(cpu_times[puzzle_index].values(), timedelta())

                if minimum is None:
                    print_with_stamp(
                        f"{files[puzzle_index]}: found no solution with k ≤ {k_upperbounds[puzzle_index]} to be possible in {total_cpu_time}"
                    )
                else:
                    print_with_stamp(
                        f"{files[puzzle_index]}: minimum k = {len(minimum)} found in {total_cpu_time}"
                    )

                result = {
                    "k": len(minimum) if minimum is not None else "n/a",
                    "moves": minimum if minimum is not None else "impossible",
                    "total_cpu_time": str(total_cpu_time),
                    "cpu_time_per_k": {
                        k: str(d) for k, d in sorted(cpu_times[puzzle_index].items())
                    },
                    "k_upperbound": k_upperbounds[puzzle_index],
                }

                result_file = open(files[puzzle_index] + ".result", "w")
                result_file.write(json.dumps(result, indent=4))
                result_file.close()


# e.g. python solve_puzzles.py ./puzzles/n2-random10.cube ./puzzles/n3-random9.cube ...
if __name__ == "__main__":
    solve_puzzles(sys.argv[1:], cpu_count())
