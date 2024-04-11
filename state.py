import itertools
import random
from typing import cast

import z3

from tools import b2s, s2b


class Move:
    """A class representing a move in the Rubik's cube. The move is represented by the
    axis, side and direction.
    """

    def __init__(self, ax: int, hi: bool, dr: int):
        """Create a new move with the given axis, side and direction."""
        self.ax = ax
        self.hi = hi
        self.dr = dr

    def inverse(self) -> "Move":
        """Return the inverse of the move."""
        if self.dr == 0:
            return Move(self.ax, self.hi, 1)
        elif self.dr == 1:
            return Move(self.ax, self.hi, 0)
        elif self.dr == 2:
            return Move(self.ax, self.hi, 2)
        raise Exception(f"invalid move direction: {self.dr}")

    def __str__(self) -> str:
        """Return the string representation of the move."""
        if self.ax == 0:
            if self.dr == 0:
                if self.hi:
                    return "R3"
                else:
                    return "L1"
            elif self.dr == 1:
                if self.hi:
                    return "R1"
                else:
                    return "L3"
            elif self.dr == 2:
                if self.hi:
                    return "R2"
                else:
                    return "L2"
        elif self.ax == 1:
            if self.dr == 0:
                if self.hi:
                    return "U3"
                else:
                    return "D1"
            elif self.dr == 1:
                if self.hi:
                    return "U1"
                else:
                    return "D3"
            elif self.dr == 2:
                if self.hi:
                    return "U2"
                else:
                    return "D2"
        elif self.ax == 2:
            if self.dr == 0:
                if self.hi:
                    return "B3"
                else:
                    return "F1"
            elif self.dr == 1:
                if self.hi:
                    return "B1"
                else:
                    return "F3"
            elif self.dr == 2:
                if self.hi:
                    return "B2"
                else:
                    return "F2"
        raise Exception(f"invalid move: ({self.ax}, {self.hi}, {self.dr})")

    @staticmethod
    def from_str(s: str) -> "Move":
        """Create a new move from the given string representation."""
        assert len(s) == 2

        # Parse the axis and side.
        match s[0]:
            case "F":
                ax = 2
                hi = False
            case "B":
                ax = 2
                hi = True
            case "L":
                ax = 0
                hi = False
            case "R":
                ax = 0
                hi = True
            case "D":
                ax = 1
                hi = False
            case "U":
                ax = 1
                hi = True
            case _:
                raise Exception(f"invalid move axis or side: {s}")

        # Parse the direction.
        match s[1]:
            case "1":
                if hi:
                    dr = 1
                else:
                    dr = 0
            case "3":
                if hi:
                    dr = 0
                else:
                    dr = 1
            case "2":
                dr = 2
            case _:
                raise Exception(f"invalid move direction: {s}")

        return Move(ax, hi, dr)

    def __eq__(self, other: "Move"):
        """Return whether two moves are equal."""
        return self.ax == other.ax and self.hi == other.hi and self.dr == other.dr

    def __hash__(self):
        """Return the hash of the move."""
        return hash((self.ax, self.hi, self.dr))

    def __lt__(self, other: "Move"):
        """Return whether one move is less than another."""
        return (self.ax, self.hi, self.dr) < (other.ax, other.hi, other.dr)

    @staticmethod
    def list_all() -> list["Move"]:
        """Return a list of all possible moves."""
        return [
            Move(*vs) for vs in itertools.product([0, 1, 2], [False, True], [0, 1, 2])
        ]


class MoveSeq:
    """A class representing a sequence of moves."""

    def __init__(self, moves: tuple[Move, ...]):
        """Create a new move sequence with the given moves."""
        self.moves = moves

    def __len__(self):
        """Return the length of the move sequence."""
        return len(self.moves)

    def __iter__(self):
        """Return an iterator over the moves in the sequence."""
        return iter(self.moves)

    def __str__(self):
        """Return the string representation of the move sequence."""
        return ";".join(map(str, self.moves))

    def __eq__(self, other: "MoveSeq"):
        """Return whether two move sequences are equal."""
        return self.moves == other.moves

    def __hash__(self):
        """Return the hash of the move sequence."""
        return hash(self.moves)

    def __lt__(self, other: "MoveSeq"):
        """Return whether one move sequence is less than another."""
        return self.moves < other.moves

    def inverted(self) -> "MoveSeq":
        """Return the inverse move sequence. Applying this to a puzzle will undo
        the moves in the sequence.
        """
        return MoveSeq(tuple(m.inverse() for m in reversed(self.moves)))

    def ax(self, s: int) -> int:
        """Return the axis of the move at the given index."""
        return self.moves[s].ax

    def hi(self, s: int) -> bool:
        """Return the side of the move at the given index."""
        return self.moves[s].hi

    def dr(self, s: int) -> int:
        """Return the direction of the move at the given index."""
        return self.moves[s].dr

    def extended(self, m: Move) -> "MoveSeq":
        """Return a new move sequence with the given move appended."""
        return MoveSeq((*self.moves, m))

    @staticmethod
    def random(k: int) -> "MoveSeq":
        """Create a new random move sequence of length k."""
        return MoveSeq(tuple(random.choices(Move.list_all(), k=k)))

    @staticmethod
    def from_str(s: str):
        """Create a new move sequence from the given string representation."""
        if s == "":
            return MoveSeq(())
        return MoveSeq(tuple(map(Move.from_str, s.split(";"))))


def generic_cubie_coord(
    n: int, x: int, y: int, z: int, m: Move
) -> tuple[int, int, int]:
    """Generic cubie coordinate mapping constructed from 90 or 180 degree rotation
    matrices. This serves as a reference implementation for mapping functions.
    """
    if m.ax == 0 and ((m.hi and x == n - 1) or (not m.hi and x == 0)):
        if m.dr == 0:
            return (x, z, n - 1 - y)
        elif m.dr == 1:
            return (x, n - 1 - z, y)
        elif m.dr == 2:
            return (x, n - 1 - y, n - 1 - z)
    elif m.ax == 1 and ((m.hi and y == n - 1) or (not m.hi and y == 0)):
        if m.dr == 0:
            return (n - 1 - z, y, x)
        elif m.dr == 1:
            return (z, y, n - 1 - x)
        elif m.dr == 2:
            return (n - 1 - x, y, n - 1 - z)
    elif m.ax == 2 and ((m.hi and z == n - 1) or (not m.hi and z == 0)):
        if m.dr == 0:
            return (y, n - 1 - x, z)
        elif m.dr == 1:
            return (n - 1 - y, x, z)
        elif m.dr == 2:
            return (n - 1 - x, n - 1 - y, z)
    return (x, y, z)


def cubie_type(n: int, x: int, y: int, z: int):
    """Determine the type of a cubie by its coordinates. 0 = corner, 1 = center,
    2 = edge and -1 = internal.
    """
    if (x in (0, n - 1)) and (y in (0, n - 1)) and (z in (0, n - 1)):
        return 0
    if (
        ((x in (0, n - 1)) and y > 0 and y < n - 1 and z > 0 and z < n - 1)
        or ((y in (0, n - 1)) and x > 0 and x < n - 1 and z > 0 and z < n - 1)
        or ((z in (0, n - 1)) and x > 0 and x < n - 1 and y > 0 and y < n - 1)
    ):
        return 1
    if x > 0 and x < n - 1 and y > 0 and y < n - 1 and z > 0 and z < n - 1:
        return -1
    return 2


class CornerState:
    """A class representing the state of a corner cubie in the Rubik's cube."""

    def __init__(self, n: int, x_hi: bool, y_hi: bool, z_hi: bool, r: int, cw: bool):
        """Create a new corner state with the given coordinates and orientation."""
        self.n = n
        self.x_hi = x_hi
        self.y_hi = y_hi
        self.z_hi = z_hi
        self.r = r
        self.cw = cw

    def coords(self) -> tuple[int, int, int]:
        """Return the coordinates of the corner cubie."""
        return (
            self.n - 1 if self.x_hi else 0,
            self.n - 1 if self.y_hi else 0,
            self.n - 1 if self.z_hi else 0,
        )

    @staticmethod
    def from_coords(n: int, x: int, y: int, z: int) -> "CornerState":
        """Create a new corner state from the given coordinates."""
        assert cubie_type(n, x, y, z) == 0
        return CornerState(n, x == n - 1, y == n - 1, z == n - 1, 0, False)

    def clockwise(self) -> bool:
        """Return whether the corner cubie is oriented clockwise."""
        return (
            (not self.x_hi and self.y_hi and not self.z_hi)
            or (not self.x_hi and not self.y_hi and self.z_hi)
            or (self.x_hi and not self.y_hi and not self.z_hi)
            or (self.x_hi and self.y_hi and self.z_hi)
        )

    def execute_move(self, move: Move) -> "CornerState":
        """Return the state of the corner cubie after executing the given move."""
        return CornerState(
            self.n,
            self.next_x_hi(move),
            self.next_y_hi(move),
            self.next_z_hi(move),
            self.next_r(move),
            self.next_cw(move),
        )

    @staticmethod
    def all_finished(n: int) -> tuple["CornerState", ...]:
        """Return a list of all finished corner states for the given cube size."""
        return tuple(
            [
                CornerState.from_coords(n, x, y, z)
                for x in range(n)
                for y in range(n)
                for z in range(n)
                if cubie_type(n, x, y, z) == 0
            ]
        )

    def __eq__(self, other: "CornerState"):
        """Return whether two corner states are equal."""
        return (
            self.n == other.n
            and self.x_hi == other.x_hi
            and self.y_hi == other.y_hi
            and self.z_hi == other.z_hi
            and self.r == other.r
            and self.cw == other.cw
        )

    def __hash__(self):
        """Return the hash of the corner state."""
        return hash((self.n, self.x_hi, self.y_hi, self.z_hi, self.r, self.cw))

    def __str__(self):
        """Return the string representation of the corner state."""
        return ";".join(
            [
                b2s(self.x_hi),
                b2s(self.y_hi),
                b2s(self.z_hi),
                str(self.r),
                b2s(self.cw),
            ]
        )

    @staticmethod
    def from_str(n: int, s: str):
        """Create a new corner state from the given string representation."""
        x_hi_raw, y_hi_raw, z_hi_raw, r_raw, cw_raw = s.split(";")
        return CornerState(
            n,
            s2b(x_hi_raw),
            s2b(y_hi_raw),
            s2b(z_hi_raw),
            int(r_raw),
            s2b(cw_raw),
        )

    def next_x_hi(self, m: Move) -> bool:
        """Return the next value of x_hi, given a move."""
        if m.ax == 1 and m.hi == self.y_hi:
            if m.dr == 0:
                return not self.z_hi
            elif m.dr == 1:
                return self.z_hi
            elif m.dr == 2:
                return not self.x_hi
        elif m.ax == 2 and m.hi == self.z_hi:
            if m.dr == 0:
                return self.y_hi
            elif m.dr == 1:
                return not self.y_hi
            elif m.dr == 2:
                return not self.x_hi
        return self.x_hi

    def next_y_hi(self, m: Move) -> bool:
        """Return the next value of y_hi, given a move."""
        if m.ax == 0 and m.hi == self.x_hi:
            if m.dr == 0:
                return self.z_hi
            elif m.dr == 1:
                return not self.z_hi
            elif m.dr == 2:
                return not self.y_hi
        elif m.ax == 2 and m.hi == self.z_hi:
            if m.dr == 0:
                return not self.x_hi
            elif m.dr == 1:
                return self.x_hi
            elif m.dr == 2:
                return not self.y_hi
        return self.y_hi

    def next_z_hi(self, m: Move) -> bool:
        """Return the next value of z_hi, given a move."""
        if m.ax == 0 and m.hi == self.x_hi:
            if m.dr == 0:
                return not self.y_hi
            elif m.dr == 1:
                return self.y_hi
            elif m.dr == 2:
                return not self.z_hi
        elif m.ax == 1 and m.hi == self.y_hi:
            if m.dr == 0:
                return self.x_hi
            elif m.dr == 1:
                return not self.x_hi
            elif m.dr == 2:
                return not self.z_hi
        return self.z_hi

    def next_r(self, m: Move) -> int:
        """Return the next value of r, given a move."""
        if m.dr != 2:
            if m.ax == 0 and m.hi == self.x_hi:
                if self.cw:
                    return (self.r - 1) % 3
                else:
                    return (self.r + 1) % 3
            elif m.ax == 2 and m.hi == self.z_hi:
                if self.cw:
                    return (self.r + 1) % 3
                else:
                    return (self.r - 1) % 3
        return self.r

    def next_cw(self, m: Move) -> bool:
        """Return the next value of cw, given a move."""
        if m.dr != 2 and (
            (m.ax == 0 and m.hi == self.x_hi)
            or (m.ax == 1 and m.hi == self.y_hi)
            or (m.ax == 2 and m.hi == self.z_hi)
        ):
            return not self.cw
        return self.cw


class EdgeState:
    """A class representing the state of an edge cubie in the Rubik's cube."""

    def __init__(self, n: int, a: int, x_hi: bool, y_hi: bool, r: bool):
        """Create a new edge state with the given coordinates and orientation."""
        self.n = n
        self.a = a
        self.x_hi = x_hi
        self.y_hi = y_hi
        self.r = r

    def coords(self) -> tuple[int, int, int]:
        """Return the coordinates of the edge cubie."""
        if self.a == 0:
            return (1, self.n - 1 if self.y_hi else 0, self.n - 1 if self.x_hi else 0)
        elif self.a == 1:
            return (self.n - 1 if self.x_hi else 0, 1, self.n - 1 if self.y_hi else 0)
        elif self.a == 2:
            return (self.n - 1 if self.x_hi else 0, self.n - 1 if self.y_hi else 0, 1)
        raise Exception(f"invalid edge axis: {self.a}")

    @staticmethod
    def from_coords(n: int, x: int, y: int, z: int) -> "EdgeState":
        """Create a new edge state from the given coordinates."""
        assert cubie_type(n, x, y, z) == 2
        if x == 1:
            return EdgeState(n, 0, z == n - 1, y == n - 1, False)
        elif y == 1:
            return EdgeState(n, 1, x == n - 1, z == n - 1, False)
        elif z == 1:
            return EdgeState(n, 2, x == n - 1, y == n - 1, False)
        raise Exception(f"invalid edge coords: ({x}, {y}, {z})")

    def execute_move(self, move: Move) -> "EdgeState":
        """Return the state of the edge cubie after executing the given move."""
        next_a = self.next_a(move)
        return EdgeState(
            self.n,
            next_a,
            self.next_x_hi(move),
            self.next_y_hi(move),
            self.next_r(next_a),
        )

    @staticmethod
    def all_finished(n: int) -> tuple["EdgeState", ...]:
        """Return a list of all finished edge states for the given cube size."""
        return tuple(
            [
                EdgeState.from_coords(n, x, y, z)
                for x in range(n)
                for y in range(n)
                for z in range(n)
                if cubie_type(n, x, y, z) == 2
            ]
        )

    def __eq__(self, other: "EdgeState"):
        """Return whether two edge states are equal."""
        return (
            self.n == other.n
            and self.a == other.a
            and self.x_hi == other.x_hi
            and self.y_hi == other.y_hi
            and self.r == other.r
        )

    def __hash__(self):
        """Return the hash of the edge state."""
        return hash((self.n, self.a, self.x_hi, self.y_hi, self.r))

    def __str__(self):
        """Return the string representation of the edge state."""
        return ";".join(
            [
                str(self.a),
                b2s(self.x_hi),
                b2s(self.y_hi),
                b2s(self.r),
            ]
        )

    @staticmethod
    def from_str(n: int, s: str):
        """Create a new edge state from the given string representation."""
        a_raw, x_hi_raw, y_hi_raw, r_raw = s.split(";")
        return EdgeState(n, int(a_raw), s2b(x_hi_raw), s2b(y_hi_raw), s2b(r_raw))

    def next_a(self, m: Move) -> int:
        """Return the next value of a, given a move."""
        if m.dr != 2:
            if self.a == 0:
                if m.ax == 1 and m.hi == self.y_hi:
                    return 2
                elif m.ax == 2 and m.hi == self.x_hi:
                    return 1
            elif self.a == 1:
                if m.ax == 0 and m.hi == self.x_hi:
                    return 2
                elif m.ax == 2 and m.hi == self.y_hi:
                    return 0
            elif self.a == 2:
                if m.ax == 0 and m.hi == self.x_hi:
                    return 1
                elif m.ax == 1 and m.hi == self.y_hi:
                    return 0
        return self.a

    def next_x_hi(self, m: Move) -> bool:
        """Return the next value of x_hi, given a move."""
        if self.a == 0:
            if m.ax == 1 and m.hi == self.y_hi and m.dr != 1:
                return not self.x_hi
            elif m.ax == 2 and m.hi == self.x_hi:
                if m.dr == 0:
                    return self.y_hi
                elif m.dr == 1:
                    return not self.y_hi
        elif self.a == 1:
            if m.ax == 2 and m.hi == self.y_hi:
                if m.dr == 2:
                    return not self.x_hi
                else:
                    return self.y_hi
        elif self.a == 2:
            if m.ax == 1 and m.hi == self.y_hi:
                if m.dr != 0:
                    return not self.x_hi
        return self.x_hi

    def next_y_hi(self, m: Move) -> bool:
        """Return the next value of y_hi, given a move."""
        if self.a == 0:
            if m.ax == 2 and m.hi == self.x_hi:
                if m.dr == 2:
                    return not self.y_hi
                else:
                    return self.x_hi
        elif self.a == 1:
            if m.ax == 0 and m.hi == self.x_hi:
                if m.dr != 0:
                    return not self.y_hi
            elif m.ax == 2 and m.hi == self.y_hi:
                if m.dr == 0:
                    return not self.x_hi
                elif m.dr == 1:
                    return self.x_hi
        elif self.a == 2:
            if m.ax == 0 and m.hi == self.x_hi:
                if m.dr != 1:
                    return not self.y_hi
        return self.y_hi

    def next_r(self, next_a: int) -> bool:
        """Return the next value of r, given the next axis."""
        if (self.a == 0 and next_a == 1) or (self.a == 1 and next_a == 0):
            return not self.r
        return self.r


class MoveZ3:
    """A class for representing moves in Z3."""

    def __init__(self, s: int, solver: z3.Solver):
        """Create a new move with the given solver."""
        self.ax = TernaryZ3.new(f"s({s}) ax", solver)
        self.hi = z3.Bool(f"s({s}) hi")
        self.dr = TernaryZ3.new(f"s({s}) dr", solver)


class TernaryZ3:
    """A class for representing ternary variables in Z3."""

    def __init__(self, b1: z3.BoolRef, b2: z3.BoolRef):
        """Create a new ternary variable with the given boolean variables."""
        self.b1 = b1
        self.b2 = b2

    @staticmethod
    def new(name: str, solver: z3.Solver):
        """Create a new ternary variable with the given name. Also disallow both values
        being true in the solver, since that would mean a value of 3.
        """
        b1 = z3.Bool(f"{name} b1")
        b2 = z3.Bool(f"{name} b2")
        solver.add(z3.Or(z3.Not(b1), z3.Not(b2)))
        return TernaryZ3(b1, b2)

    def __eq__(self, other: "int | TernaryZ3"):
        """Return the condition of two ternary variables being equal."""
        if isinstance(other, int):
            match other:
                case 0:
                    return z3.And(z3.Not(self.b1), z3.Not(self.b2))
                case 1:
                    return self.b2
                case 2:
                    return self.b1
                case _:
                    raise Exception(f"invalid ternary value: {other}")
        elif isinstance(other, TernaryZ3):
            return z3.And(self.b1 == other.b1, self.b2 == other.b2)

    def __ne__(self, other: "TernaryZ3 | int"):
        """Return the condition of two ternary variables being different."""
        if isinstance(other, int):
            match other:
                case 0:
                    return z3.Or(self.b1, self.b2)
                case 1:
                    return z3.Not(self.b2)
                case 2:
                    return z3.Not(self.b1)
                case _:
                    raise Exception(f"invalid ternary value: {other}")
        else:
            assert isinstance(other, TernaryZ3)
            return z3.Or(self.b1 != other.b1, self.b2 != other.b2)

    def __gt__(self, other: "TernaryZ3 | int"):
        """Return the condition of one ternary variable being greater than another."""
        if isinstance(other, int):
            match other:
                case 0:
                    return z3.Or(self.b1, self.b2)
                case 1:
                    return self.b1
                case 2:
                    return False
                case _:
                    raise Exception(f"invalid ternary value: {other}")
        else:
            assert isinstance(other, TernaryZ3)
            return z3.And(
                z3.Not(other.b1),
                z3.Or(
                    self.b1,
                    z3.And(self.b2, z3.Not(other.b2)),
                ),
            )

    def arith_value(self) -> z3.ArithRef:
        """Return the value of the ternary variable."""
        b1v = cast(z3.ArithRef, z3.If(self.b1, 2, 0))
        b2v = cast(z3.ArithRef, z3.If(self.b2, 1, 0))
        return b1v + b2v


class CornerStateZ3:
    """A class for representing corner states in Z3."""

    def __init__(
        self,
        x_hi: z3.BoolRef,
        y_hi: z3.BoolRef,
        z_hi: z3.BoolRef,
        r: TernaryZ3,
        cw: z3.BoolRef,
    ):
        """Create a new corner state with the given variables."""
        self.x_hi = x_hi
        self.y_hi = y_hi
        self.z_hi = z_hi
        self.r = r
        self.cw = cw

    @staticmethod
    def new(s: int, x_hi: bool, y_hi: bool, z_hi: bool, solver: z3.Solver):
        """Create a new corner state with the given coordinates and orientation."""
        return CornerStateZ3(
            z3.Bool(f"corner({x_hi},{y_hi},{z_hi}) s({s}) x_hi"),
            z3.Bool(f"corner({x_hi},{y_hi},{z_hi}) s({s}) y_hi"),
            z3.Bool(f"corner({x_hi},{y_hi},{z_hi}) s({s}) z_hi"),
            TernaryZ3.new(f"corner({x_hi},{y_hi},{z_hi}) s({s}) r", solver),
            z3.Bool(f"corner({x_hi},{y_hi},{z_hi}) s({s}) c"),
        )

    def __eq__(self, other: "CornerState | CornerStateZ3"):
        """Return the conditions for two corner states being equal."""
        return z3.And(
            [
                self.x_hi == other.x_hi,
                self.y_hi == other.y_hi,
                self.z_hi == other.z_hi,
                self.r == other.r,
                self.cw == other.cw,
            ]
        )

    def __ne__(self, other: "CornerState | CornerStateZ3"):
        """Return the conditions for two corner states being different."""
        return z3.Or(
            [
                self.x_hi != other.x_hi,
                self.y_hi != other.y_hi,
                self.z_hi != other.z_hi,
                self.r != other.r,
                self.cw != other.cw,
            ]
        )


class EdgeStateZ3:
    """A class for representing edge states in Z3."""

    def __init__(self, a: TernaryZ3, x_hi: z3.BoolRef, y_hi: z3.BoolRef, r: z3.BoolRef):
        """Create a new edge state with the given variables."""
        self.a = a
        self.x_hi = x_hi
        self.y_hi = y_hi
        self.r = r

    @staticmethod
    def new(s: int, a: int, x_hi: bool, y_hi: bool, solver: z3.Solver):
        """Create a new edge state with the given coordinates and orientation."""
        return EdgeStateZ3(
            TernaryZ3.new(f"edge({a},{x_hi},{y_hi}) s({s}) a", solver),
            z3.Bool(f"edge({a},{x_hi},{y_hi}) s({s}) x_hi"),
            z3.Bool(f"edge({a},{x_hi},{y_hi}) s({s}) y_hi"),
            z3.Bool(f"edge({a},{x_hi},{y_hi}) s({s}) r"),
        )

    def __eq__(self, other: "EdgeState | EdgeStateZ3"):
        """Return the conditions for two edge states being equal."""
        return z3.And(
            [
                self.a == other.a,
                self.x_hi == other.x_hi,
                self.y_hi == other.y_hi,
                self.r == other.r,
            ]
        )

    def __ne__(self, other: "EdgeState | EdgeStateZ3"):
        """Return the conditions for two edge states being different."""
        return z3.Or(
            [
                self.a != other.a,
                self.x_hi != other.x_hi,
                self.y_hi != other.y_hi,
                self.r != other.r,
            ]
        )
