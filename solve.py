from datetime import datetime
import numpy as np
import sys
import z3
from multiprocessing import Process, Manager


def face_name(f: int):
    """Convert a face index to its canonical name."""
    if f == 0:
        return "front"
    elif f == 1:
        return "right"
    elif f == 2:
        return "back"
    elif f == 3:
        return "left"
    elif f == 4:
        return "top"
    elif f == 5:
        return "bottom"


def color_name(c: int):
    """Convert a color index to its canonical name."""
    if c == 0:
        return "white"
    elif c == 1:
        return "green"
    elif c == 2:
        return "yellow"
    elif c == 3:
        return "blue"
    elif c == 4:
        return "red"
    elif c == 5:
        return "orange"


def move_name(ma: int, md: int, mi: int):
    """Convert a move index to its canonical name."""
    if ma == 0:
        if md == 0:
            return f"quarter turn row {mi} to the left"
        elif md == 1:
            return f"quarter turn row {mi} to the right"
        elif md == 2:
            return f"half turn row {mi}"
    elif ma == 1:
        if md == 0:
            return f"quarter turn column {mi} upwards"
        elif md == 1:
            return f"quarter turn column {mi} downwards"
        elif md == 2:
            return f"half turn column {mi}"
    elif ma == 2:
        if md == 0:
            return f"quarter turn depth {mi} clockwise"
        elif md == 1:
            return f"quarter turn depth {mi} counterclockwise"
        elif md == 2:
            return f"half turn depth {mi}"


def state_to_cube(n, model, state):
    """Convert a state to a printable cube using a model."""
    return np.array([[[model[state[f][y][x]].as_long() for x in range(n)] for y in range(n)] for f in range(6)])


def impl(conditions: list, consequences: list):
    """Express an implication in a Z3 disjuction, which is more efficient."""
    return z3.Or([z3.Not(condition) for condition in conditions] + [z3.And(consequences)])


def solve(starting_state: list[list[list[int]]], max_moves: int):
    solver = z3.Optimize()
    n = len(starting_state[0])

    # List of states of the cube. The front, right, back and left faces face upwards.
    # The bottom and top faces both face upwards when rotating them towards you.
    color_states = [[[[z3.Int(f"{s},{f},{y},{x}") for x in range(n)] for y in range(n)] for f in range(6)] for s in range(max_moves + 1)]

    # Restrict color states domains.
    for s in range(len(color_states)):
        for f in range(6):
            for y in range(n):
                for x in range(n):
                    solver.add(z3.And(color_states[s][f][y][x] >= 0, color_states[s][f][y][x] < 6))

    # Restrict starting colors of cube to input.
    for f in range(6):
        for y in range(n):
            for x in range(n):
                solver.add(color_states[0][f][y][x] == starting_state[f][y][x])

    def is_complete(s: int):
        """Return restriction on whether a state is complete."""
        return z3.And(
            [z3.And([z3.And([color_states[s][f][y][x] == color_states[s][f][0][0] for x in range(n)]) for y in range(n)]) for f in range(6)]
        )

    # Restrict final state to being complete.
    solver.add(is_complete(-1))

    # Lists of variables which together indicate the move made at all states
    move_axes = z3.IntVector("move_axis", len(color_states) - 1)
    move_indices = z3.IntVector("move_index", len(color_states) - 1)
    move_directions = z3.IntVector("move_direction", len(color_states) - 1)

    # Restrict moves to valid moves.
    for i in range(len(color_states) - 1):
        solver.add(
            z3.And(
                move_axes[i] >= 0,
                move_axes[i] <= 2,
            )
        )
        solver.add(
            z3.And(
                move_indices[i] >= 0,
                move_indices[i] <= n,
            )
        )
        solver.add(
            z3.And(
                move_directions[i] >= 0,
                move_directions[i] <= 2,
            )
        )

    # Restrict color states when previous move is nothing.
    for s in range(1, len(color_states)):
        solver.add(
            z3.Implies(
                move_indices[s - 1] == n,
                z3.And([color_states[s][f][y][x] == color_states[s - 1][f][y][x] for f in range(6) for y in range(n) for x in range(n)]),
            )
        )

    # Restrict color states using shortcuts file.
    shortcuts: list[list[list[set[tuple[tuple[int | None, int | None, int | None], tuple[int, int, int]]]]]] = eval(
        open(f"shortcuts/dim{n}/shortcuts.txt", "r").read()
    )
    for s in range(1, len(color_states)):
        # Create storage for collecting implications together.
        implications = {
            ma: {mi: {md: set() for md in list(range(3)) + [None]} for mi in list(range(n)) + [None]} for ma in list(range(3)) + [None]
        }

        # Collect implications from shortcuts file.
        for f in range(6):
            for y in range(n):
                for x in range(n):
                    ss = shortcuts[f][y][x]
                    for shortcut, output in ss:
                        ma, mi, md = shortcut
                        implications[ma][mi][md].add(((f, y, x), output))

        # Insert implications into solver.
        for ma in list(range(3)) + [None]:
            for mi in list(range(n)) + [None]:
                for md in list(range(3)) + [None]:
                    conditions = []
                    if ma is not None:
                        conditions.append(move_axes[s - 1] == ma)
                    if mi is not None:
                        conditions.append(move_indices[s - 1] == mi)
                    if md is not None:
                        conditions.append(move_directions[s - 1] == md)
                    consequences = [
                        color_states[s][f][y][x] == color_states[s - 1][f_target][y_target][x_target]
                        for (f, y, x), (f_target, y_target, x_target) in implications[ma][mi][md]
                    ]
                    if len(consequences) == 0:
                        continue
                    solver.add(impl(conditions, consequences))

    # Only allow non-moves when complete.
    for s in range(0, len(color_states) - 1):
        solver.add((move_indices[s] == n) == is_complete(s))

    # If between 1 and n moves ago we made a turn at an index and axis, a different axis has to have been turned in the meantime.
    for s in range(1, len(color_states) - 1):
        for r in range(1, min(n + 1, s + 1)):
            solver.add(
                z3.Or(
                    move_axes[s - r] != move_axes[s],
                    move_indices[s - r] != move_indices[s],
                    z3.Or([move_axes[s] != move_axes[p] for p in range(s - r + 1, s)]),
                )
            )

    # All subsequent moves in the same axis happen in ascending order of index.
    for s in range(1, len(color_states) - 1):
        solver.add(
            z3.Or(
                [
                    move_axes[s - 1] != move_axes[s],
                    z3.And([z3.And([z3.Or(move_indices[s - 1] != b, move_indices[s] != c) for c in range(1, b)]) for b in range(n, 1, -1)]),
                ]
            )
        )

    # Check model and report moves if sat, else return unsat.
    res = solver.check()
    if res == z3.sat:
        moves = []
        model = solver.model()
        for i in range(len(color_states) - 1):
            mi = model[move_indices[i]].as_long()
            md = model[move_directions[i]].as_long()
            ma = model[move_axes[i]].as_long()
            if mi == n:
                continue
            moves.append(move_name(ma, md, mi))
        return moves
    else:
        return "unsat"


def main(puzzle_file: str, max_moves: int, minimize_cores: int):
    start = datetime.now()
    puzzle = eval(open(puzzle_file, "r").read())

    if minimize_cores == 0:
        solution = solve(puzzle, max_moves)
        if solution == "unsat":
            print(f"no solution possible, found in {datetime.now()-start}")
        else:
            print(f"solution of {len(solution)} moves found in {datetime.now()-start}: {', '.join(solution)}")
        exit()

    best_solution = None
    with Manager() as manager:
        prospects = list(range(max_moves + 1))  # try all max_moves from 0 to n (inclusive)
        processes: dict[int, Process] = {}  # store all running processes with their max_moves parameter
        solutions = manager.Queue()  # queue into which solutions are put

        def spawn_new_solver():
            """Take a prospect and spawn a solver process. The result is put into the solutions queue."""

            def solve_wrapper(starting_state, max_moves, return_queue):
                solution = solve(starting_state, max_moves)
                return_queue.put((max_moves, solution))

            if len(prospects) > 0:
                median = len(prospects) // 2
                max_moves = prospects.pop(median)
                process = Process(target=solve_wrapper, args=(puzzle, max_moves, solutions))
                assert not max_moves in processes
                processes[max_moves] = process
                process.start()

        # Spawn the specified number of cores
        for _ in range(minimize_cores):
            spawn_new_solver()

        while len(processes) > 0:
            parameter, solution = solutions.get()
            if solution == "unsat":
                # Filter out prospects which cannot be SAT anymore.
                prospects = [p for p in prospects if p > parameter]

                # Kill processes which cannot be SAT anymore and replace them.
                for running_parameter in list(processes.keys()):
                    if running_parameter <= parameter:
                        processes.pop(running_parameter).kill()
                        spawn_new_solver()
            else:
                # If solution is better, replace the best solution and report.
                if best_solution is None or len(solution) < len(best_solution):
                    best_solution = solution
                    print(f"solution of {len(best_solution)} moves found after {datetime.now()-start}, minimizing...")

                # Filter out prospects which are worse than the current best.
                prospects = [p for p in prospects if p < len(best_solution)]

                # Kill processes which cannot be better than the current best and replace them.
                for running_parameter in list(processes.keys()):
                    if running_parameter >= len(best_solution):
                        processes.pop(running_parameter).kill()
                        spawn_new_solver()

    if best_solution is None:
        solution = f"no solution possible, found in {datetime.now()-start}"
    else:
        solution = f"minimum size solution of {len(best_solution)} moves found in {datetime.now()-start} using {minimize_cores} cores and {max_moves} moves at maximum: {', '.join(best_solution)}"
    solution_file = open(puzzle_file + ".solution", "w")
    solution_file.write(solution)
    solution_file.close()
    print(solution)


# python solve_puzzle.py {puzzle.txt} {max_moves} {minimize_cores}
if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
