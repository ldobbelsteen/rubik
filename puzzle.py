import argparse
import itertools

from PIL import Image, ImageDraw

import move_mappers
from tools import rotate_list

CubicleCoords = tuple[int, int, int]  # (x, y, z)
FaceletCoords = tuple[int, int, int]  # (f, y, x)

CornerState = tuple[bool, bool, bool, int, bool]  # (x_hi, y_hi, z_hi, r, cw)
CenterState = tuple[int, bool]  # (a, hi)
EdgeState = tuple[int, int, int, bool]  # (x, y, z, r)

Move = tuple[int, bool, int]  # (ax, hi, dr)
MoveSeq = tuple[Move, ...]


# The global face ordering by which the desired color ordering is achieved.
# First come top/bottom, second front/back and third left/right.
FACE_ORDERING = [4, 5, 0, 2, 3, 1]


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
            return "up"
        case 5:
            return "down"
    raise Exception(f"invalid face: {f}")


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
    raise Exception(f"invalid color: {c}")


def move_name(move: Move) -> str:
    """Convert a move to its canonical name."""
    ax, hi, dr = move
    if ax == 0:
        if dr == 0:
            if hi:
                return "R3"
            else:
                return "L1"
        elif dr == 1:
            if hi:
                return "R1"
            else:
                return "L3"
        elif dr == 2:
            if hi:
                return "R2"
            else:
                return "L2"
    elif ax == 1:
        if dr == 0:
            if hi:
                return "U3"
            else:
                return "D1"
        elif dr == 1:
            if hi:
                return "U1"
            else:
                return "D3"
        elif dr == 2:
            if hi:
                return "U2"
            else:
                return "D2"
    elif ax == 2:
        if dr == 0:
            if hi:
                return "B3"
            else:
                return "F1"
        elif dr == 1:
            if hi:
                return "B1"
            else:
                return "F3"
        elif dr == 2:
            if hi:
                return "B2"
            else:
                return "F2"
    raise Exception(f"invalid move: {move}")


def parse_move(canonical_name: str) -> Move:
    """Convert a canonical move name to its internal representation."""
    assert len(canonical_name) == 2

    # Parse the axis and side.
    match canonical_name[0]:
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
            raise Exception(f"invalid move axis or side: {canonical_name}")

    # Parse the direction.
    match canonical_name[1]:
        case "1":
            dr = 0
        case "3":
            dr = 1
        case "2":
            dr = 2
        case _:
            raise Exception(f"invalid move direction: {canonical_name}")

    return (ax, hi, dr)


def inverse_move(move: Move) -> Move:
    """Get the move that undoes a move."""
    ax, hi, dr = move
    if dr == 0:
        return (ax, hi, 1)
    elif dr == 1:
        return (ax, hi, 0)
    elif dr == 2:
        return (ax, hi, 2)
    raise Exception(f"invalid move: {move}")


def all_moves() -> list[Move]:
    """List all valid moves."""
    return list(itertools.product([0, 1, 2], [False, True], [0, 1, 2]))


def cubicle_type(n: int, cubicle: CubicleCoords):
    """Determine the type of a cubicle by its coordinates. 0 = corner, 1 = center,
    2 = edge and -1 = internal."""
    x, y, z = cubicle
    if (x == 0 or x == n - 1) and (y == 0 or y == n - 1) and (z == 0 or z == n - 1):
        return 0
    if (
        ((x == 0 or x == n - 1) and y > 0 and y < n - 1 and z > 0 and z < n - 1)
        or ((y == 0 or y == n - 1) and x > 0 and x < n - 1 and z > 0 and z < n - 1)
        or ((z == 0 or z == n - 1) and x > 0 and x < n - 1 and y > 0 and y < n - 1)
    ):
        return 1
    if x > 0 and x < n - 1 and y > 0 and y < n - 1 and z > 0 and z < n - 1:
        return -1
    return 2


def cubicle_colors(n: int, cubicle: CubicleCoords) -> list[int]:
    """Get the list of colors of a cubicle in a finished puzzle. The list is sorted
    by the global face ordering."""
    x, y, z = cubicle
    colors = set()
    if x == 0:
        colors.add(3)
    if x == n - 1:
        colors.add(1)
    if y == 0:
        colors.add(5)
    if y == n - 1:
        colors.add(4)
    if z == 0:
        colors.add(0)
    if z == n - 1:
        colors.add(2)
    return [f for f in FACE_ORDERING if f in colors]


def cubicle_facelets(n: int, cubicle: CubicleCoords) -> list[FaceletCoords]:
    """Get the list of facelets of a cubicle. The list is sorted by the global
    face ordering."""
    x, y, z = cubicle
    facelets = []
    for ff in FACE_ORDERING:
        for fy in range(n):
            for fx in range(n):
                facelet = (ff, fy, fx)
                if facelet_cubicle(n, facelet) == (x, y, z):
                    facelets.append(facelet)
    return facelets


def facelet_cubicle(n: int, facelet: FaceletCoords) -> CubicleCoords:
    """Get the cubicle on which a facelet is located."""
    f, y, x = facelet
    match f:
        case 0:
            return (x, y, 0)
        case 1:
            return (n - 1, y, x)
        case 2:
            return (n - 1 - x, y, n - 1)
        case 3:
            return (0, y, n - 1 - x)
        case 4:
            return (x, n - 1, y)
        case 5:
            return (x, 0, n - 1 - y)
    raise Exception(f"invalid face: {f}")


def encode_corner(n: int, corner: CubicleCoords) -> CornerState:
    """Convert a corner cubie's coordinates into our encoding in its finished state."""
    x, y, z = corner
    return (x == n - 1, y == n - 1, z == n - 1, 0, False)


def decode_corner(n: int, corner: CornerState) -> CubicleCoords:
    """Extract the coordinates of a corner cubie from our encoding."""
    x, y, z, _, _ = corner
    return (n - 1 if x else 0, n - 1 if y else 0, n - 1 if z else 0)


def encode_center(n: int, center: CubicleCoords) -> CenterState:
    """Convert a center cubie's coordinates into our encoding in its finished state."""
    x, y, z = center
    if x != 1:
        return (0, x == n - 1)
    elif y != 1:
        return (1, y == n - 1)
    elif z != 1:
        return (2, z == n - 1)
    raise Exception(f"invalid center: {center}")


def decode_center(n: int, center: CenterState) -> CubicleCoords:
    """Extract the coordinates of a center cubie from our encoding."""
    a, h = center
    if a == 0:
        return (n - 1 if h else 0, 1, 1)
    elif a == 1:
        return (1, n - 1 if h else 0, 1)
    elif a == 2:
        return (1, 1, n - 1 if h else 0)
    raise Exception(f"invalid center: {center}")


def encode_edge(edge: CubicleCoords) -> EdgeState:
    """Convert an edge cubie's coordinates into our encoding in its finished state."""
    x, y, z = edge
    return (x, y, z, False)


def decode_edge(edge: EdgeState) -> CubicleCoords:
    """Extract the coordinates of an edge cubie from our encoding."""
    x, y, z, _ = edge
    return (x, y, z)


def corner_clockwise(corner: CornerState) -> bool:
    """Determine whether a corner cubicle's colors are labeled clockwise or not.
    This is a result of the global face ordering."""
    x, y, z, _, _ = corner
    return (
        (not x and y and not z)
        or (not x and not y and z)
        or (x and not y and not z)
        or (x and y and z)
    )


class FinishedState:
    def __init__(self, n: int):
        self.n = n
        self.corners: list[CornerState] = []
        self.centers: list[CenterState] = []
        self.edges: list[EdgeState] = []

        for x in range(n):
            for y in range(n):
                for z in range(n):
                    cubicle = (x, y, z)
                    match cubicle_type(n, cubicle):
                        case 0:
                            self.corners.append(encode_corner(n, cubicle))
                        case 1:
                            self.centers.append(encode_center(n, cubicle))
                        case 2:
                            self.edges.append(encode_edge(cubicle))

    def corner_idx(self, corner: CubicleCoords):
        return self.corners.index(encode_corner(self.n, corner))

    def center_idx(self, center: CubicleCoords):
        return self.centers.index(encode_center(self.n, center))

    def edge_idx(self, edge: CubicleCoords):
        return self.edges.index(encode_edge(edge))


class Puzzle:
    def __init__(
        self,
        n: int,
        finished_state: FinishedState,
        corner_states: list[CornerState],
        center_states: list[CenterState],
        edge_states: list[EdgeState],
    ):
        self.n = n
        self.finished_state = finished_state
        self.corner_states = corner_states
        self.center_states = center_states
        self.edge_states = edge_states

    @staticmethod
    def from_file(path: str):
        with open(path, "r") as file:
            content = file.read()
            return Puzzle.from_str(content)

    @staticmethod
    def from_str(s: str):
        n = int((len(s) / 6) ** 0.5)
        if len(s) != 6 * n * n:
            raise Exception("invalid puzzle string length")
        if n != 2 and n != 3:
            raise Exception(f"n = {n} not supported")

        return Puzzle.from_facelet_colors(
            n,
            [
                [
                    [int(s[f * n * n + y * n + x]) for x in range(n)]
                    for y in reversed(range(n))
                ]
                for f in range(6)
            ],  # extract facelet colors from string
        )

    @staticmethod
    def from_facelet_colors(n: int, facelet_colors: list[list[list[int]]]):
        finished = FinishedState(n)

        # Extract the cubicle colors from the facelet representation.
        corner_colors = [[] for _ in finished.corners]
        center_colors = [[] for _ in finished.centers]
        edge_colors = [[] for _ in finished.edges]
        for ff in FACE_ORDERING:
            for fy in range(n):
                for fx in range(n):
                    facelet = (ff, fy, fx)
                    cubicle = facelet_cubicle(n, facelet)
                    match cubicle_type(n, cubicle):
                        case 0:
                            corner_colors[finished.corner_idx(cubicle)].append(
                                facelet_colors[ff][fy][fx]
                            )
                        case 1:
                            center_colors[finished.center_idx(cubicle)].append(
                                facelet_colors[ff][fy][fx]
                            )
                        case 2:
                            edge_colors[finished.edge_idx(cubicle)].append(
                                facelet_colors[ff][fy][fx]
                            )

        corner_states: list[CornerState | None] = [None] * len(finished.corners)
        for i, corner in enumerate(finished.corners):
            colors = corner_colors[i]
            for i2, corner2 in enumerate(finished.corners):
                colors2 = cubicle_colors(n, decode_corner(n, corner2))
                if set(colors) == set(colors2):
                    x_hi, y_hi, z_hi, _, _ = corner
                    r = colors2.index(colors[0])
                    cw = corner_clockwise(corner) != corner_clockwise(corner2)
                    assert corner_states[i2] is None
                    corner_states[i2] = (x_hi, y_hi, z_hi, r, cw)

        center_states: list[CenterState | None] = [None] * len(finished.centers)
        for i, center in enumerate(finished.centers):
            colors = center_colors[i]
            for i2, center2 in enumerate(finished.centers):
                colors2 = cubicle_colors(n, decode_center(n, center2))
                if set(colors) == set(colors2):
                    assert center_states[i2] is None
                    center_states[i2] = center

        edge_states: list[EdgeState | None] = [None] * len(finished.edges)
        for i, edge in enumerate(finished.edges):
            colors = edge_colors[i]
            for i2, edge2 in enumerate(finished.edges):
                colors2 = cubicle_colors(n, decode_edge(edge2))
                if set(colors) == set(colors2):
                    x, y, z, _ = edge
                    r = colors2.index(colors[0]) == 1
                    assert edge_states[i2] is None
                    edge_states[i2] = (x, y, z, r)

        assert None not in corner_states
        assert None not in center_states
        assert None not in edge_states
        return Puzzle(n, finished, corner_states, center_states, edge_states)  # type: ignore

    @staticmethod
    def finished(n: int):
        state = FinishedState(n)
        return Puzzle(
            n,
            state,
            state.corners.copy(),
            state.centers.copy(),
            state.edges.copy(),
        )

    def copy(self):
        return Puzzle(
            self.n,
            self.finished_state,
            self.corner_states.copy(),
            self.center_states.copy(),
            self.edge_states.copy(),
        )

    def __eq__(self, other: "Puzzle"):
        return (
            self.n == other.n
            and self.corner_states == other.corner_states
            and self.center_states == other.center_states
            and self.edge_states == other.edge_states
        )

    def __hash__(self):
        return hash(
            (
                self.n,
                tuple(self.corner_states),
                tuple(self.center_states),
                tuple(self.edge_states),
            )
        )

    def to_str(self):
        facelet_colors = [
            [
                [self.facelet_color((f, y, x)) for x in range(self.n)]
                for y in range(self.n)
            ]
            for f in range(6)
        ]

        flattened = [
            facelet_colors[f][y][x]
            for f in range(6)
            for y in reversed(range(self.n))
            for x in range(self.n)
        ]

        return "".join([str(c) for c in flattened])

    def execute_move(self, move: Move):
        ax, hi, dr = move

        for i, (x_hi, y_hi, z_hi, r, cw) in enumerate(self.corner_states):
            self.corner_states[i] = (
                move_mappers.corner_x_hi(x_hi, y_hi, z_hi, ax, hi, dr),
                move_mappers.corner_y_hi(x_hi, y_hi, z_hi, ax, hi, dr),
                move_mappers.corner_z_hi(x_hi, y_hi, z_hi, ax, hi, dr),
                move_mappers.corner_r(x_hi, z_hi, r, cw, ax, hi, dr),
                move_mappers.corner_cw(x_hi, y_hi, z_hi, cw, ax, hi, dr),
            )

        for i, (x, y, z, r) in enumerate(self.edge_states):
            self.edge_states[i] = (
                move_mappers.edge_x(self.n, x, y, z, ax, hi, dr),
                move_mappers.edge_y(self.n, x, y, z, ax, hi, dr),
                move_mappers.edge_z(self.n, x, y, z, ax, hi, dr),
                move_mappers.edge_r(self.n, x, z, r, ax, hi, dr),
            )

    def facelet_color(self, facelet: FaceletCoords) -> int:
        cubicle = facelet_cubicle(self.n, facelet)

        match cubicle_type(self.n, cubicle):
            case 0:
                for i, corner in enumerate(self.finished_state.corners):
                    origin_corner = self.corner_states[i]
                    if cubicle == decode_corner(self.n, origin_corner):
                        colors = cubicle_colors(self.n, decode_corner(self.n, corner))
                        _, _, _, r, cw = origin_corner
                        first_color = colors[r]

                        # If the cubie's direction has changed, reverse the
                        # color list to adhere to ordering.
                        if cw:
                            colors.reverse()

                        # Rotate until the color is in the first slot.
                        while colors[0] != first_color:
                            rotate_list(colors)

                        # Get the facelet index of the facelet in question and return.
                        fi = cubicle_facelets(self.n, cubicle).index(facelet)
                        return colors[fi]
            case 1:
                for i, center in enumerate(self.finished_state.centers):
                    origin_center = self.center_states[i]
                    if cubicle == decode_center(self.n, origin_center):
                        colors = cubicle_colors(self.n, decode_center(self.n, center))
                        return colors[0]
            case 2:
                for i, edge in enumerate(self.finished_state.edges):
                    origin_edge = self.edge_states[i]
                    if cubicle == decode_edge(origin_edge):
                        colors = cubicle_colors(self.n, decode_edge(edge))
                        fi = cubicle_facelets(self.n, cubicle).index(facelet)
                        _, _, _, r = origin_edge
                        if not r:
                            return colors[fi]
                        else:
                            return colors[1 - fi]

        raise Exception(f"invalid facelet: {facelet}")

    def print(self):
        facelet_size = 48
        image_size = (3 * self.n * facelet_size, 4 * self.n * facelet_size)
        im = Image.new(mode="RGB", size=image_size)
        draw = ImageDraw.Draw(im)

        def draw_face(start_x: int, start_y: int, f: int, rotate=False):
            for y in range(self.n):
                for x in range(self.n):
                    if rotate:
                        color = self.facelet_color((f, self.n - 1 - y, self.n - 1 - x))
                    else:
                        color = self.facelet_color((f, y, x))
                    draw.rectangle(
                        (
                            start_x + (x * facelet_size),
                            start_y + ((self.n - 1 - y) * facelet_size),
                            start_x + ((x + 1) * facelet_size),
                            start_y + ((self.n - y) * facelet_size),
                        ),
                        fill=color_name(color),
                        outline="black",
                        width=4,
                    )

        draw_face(1 * self.n * facelet_size, 1 * self.n * facelet_size, 0)
        draw_face(2 * self.n * facelet_size, 1 * self.n * facelet_size, 1)
        draw_face(1 * self.n * facelet_size, 3 * self.n * facelet_size, 2, True)
        draw_face(0 * self.n * facelet_size, 1 * self.n * facelet_size, 3)
        draw_face(1 * self.n * facelet_size, 0 * self.n * facelet_size, 4)
        draw_face(1 * self.n * facelet_size, 2 * self.n * facelet_size, 5)
        im.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    Puzzle.from_file(args.path).print()
