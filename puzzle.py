import argparse
import itertools

from PIL import Image, ImageDraw

import move_mappers
from misc import rotate_list

Corner = tuple[bool, bool, bool, int, bool]  # (x, y, z, r, c)
Center = tuple[int, bool]  # (a, h)
Edge = tuple[int, int, int, bool]  # (x, y, z, r)

SUPPORTED_NS = {2, 3}

# The global face ordering by which the desired rotation ordering is achieved.
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
            return "top"
        case 5:
            return "bottom"
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


def move_name(ma: int, mi: int, md: int) -> str:
    """Convert a move to its canonical name."""
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
    raise Exception(f"invalid move: ({ma},{mi},{md})")


def list_all_moves(n: int):
    """List all possible moves given n."""
    return list(itertools.product([0, 1, 2], range(n), [0, 1, 2]))


def cubicle_type(n: int, x: int, y: int, z: int):
    """Determine the type of a cubicle by its coordinates. 0 = corner, 1 = center,
    2 = edge and -1 = internal."""
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


def cubicle_colors(n: int, x: int, y: int, z: int) -> list[int]:
    """Get the list of colors of a cubicle in a finished cube. The list is sorted
    by the global face ordering."""
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


def cubicle_facelets(n: int, x: int, y: int, z: int) -> list[tuple[int, int, int]]:
    """Get the list of facelets of a cubicle. The list is sorted by the global
    face ordering."""
    facelets = []
    for ff in FACE_ORDERING:
        for fy in range(n):
            for fx in range(n):
                if facelet_cubicle(n, ff, fy, fx) == (x, y, z):
                    facelets.append((ff, fy, fx))
    return facelets


def facelet_cubicle(n: int, f: int, y: int, x: int) -> tuple[int, int, int]:
    """Get the cubicle on which a facelet is located."""
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
        case _:
            raise Exception(f"invalid face: {f}")


def coord_to_corner(n: int, x: int, y: int, z: int) -> Corner:
    """Convert a corner cubie's coordinates to its encoding."""
    return (x == n - 1, y == n - 1, z == n - 1, 0, False)


def corner_to_coord(n: int, corner: Corner) -> tuple[int, int, int]:
    """Convert a corner's encoding to its cubie coordinates."""
    x, y, z, _, _ = corner
    return (n - 1 if x else 0, n - 1 if y else 0, n - 1 if z else 0)


def corner_clockwise(corner: Corner) -> bool:
    """Determine whether a corner cubie's colors are labeled clockwise or not."""
    x, y, z, _, _ = corner
    return (
        (not x and y and not z)
        or (not x and not y and z)
        or (x and not y and not z)
        or (x and y and z)
    )


def coord_to_center(n: int, x: int, y: int, z: int) -> Center:
    """Convert a center cubie's coordinates to its encoding."""
    if x != 1:
        return (0, x == n - 1)
    elif y != 1:
        return (1, y == n - 1)
    elif z != 1:
        return (2, z == n - 1)
    raise Exception(f"coord not center: ({x},{y},{z})")


def center_to_coord(n: int, center: Center) -> tuple[int, int, int]:
    """Convert a centers's encoding to its cubie coordinates."""
    a, h = center
    if a == 0:
        return (n - 1 if h else 0, 1, 1)
    elif a == 1:
        return (1, n - 1 if h else 0, 1)
    elif a == 2:
        return (1, 1, n - 1 if h else 0)
    raise Exception(f"invalid center axis: {a}")


def coord_to_edge(x: int, y: int, z: int) -> Edge:
    """Convert an edge cubie's coordinates to its encoding."""
    return (x, y, z, False)


def edge_to_coord(edge: Edge) -> tuple[int, int, int]:
    """Convert a edge's encoding to its cubie coordinates."""
    x, y, z, _ = edge
    return (x, y, z)


class Cubicles:
    def __init__(self, n: int):
        self.corners: list[Corner] = []
        self.centers: list[Center] = []
        self.edges: list[Edge] = []

        for x in range(n):
            for y in range(n):
                for z in range(n):
                    match cubicle_type(n, x, y, z):
                        case 0:
                            self.corners.append(coord_to_corner(n, x, y, z))
                        case 1:
                            self.centers.append(coord_to_center(n, x, y, z))
                        case 2:
                            self.edges.append(coord_to_edge(x, y, z))


class Puzzle:
    def __init__(
        self,
        n: int,
        cubicles: Cubicles,
        corners: list[Corner],
        centers: list[Center],
        edges: list[Edge],
    ):
        self.n = n
        self.cubicles = cubicles
        self.corners = corners
        self.centers = centers
        self.edges = edges

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
        if n not in SUPPORTED_NS:
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
        cubicles = Cubicles(n)

        # Extract the cubicle colors from the facelet representation.
        corner_colors = [[] for _ in cubicles.corners]
        center_colors = [[] for _ in cubicles.centers]
        edge_colors = [[] for _ in cubicles.edges]
        for ff in FACE_ORDERING:
            for fy in range(n):
                for fx in range(n):
                    x, y, z = facelet_cubicle(n, ff, fy, fx)
                    match cubicle_type(n, x, y, z):
                        case 0:
                            corner_colors[
                                cubicles.corners.index(coord_to_corner(n, x, y, z))
                            ].append(facelet_colors[ff][fy][fx])
                        case 1:
                            center_colors[
                                cubicles.centers.index(coord_to_center(n, x, y, z))
                            ].append(facelet_colors[ff][fy][fx])
                        case 2:
                            edge_colors[
                                cubicles.edges.index(coord_to_edge(x, y, z))
                            ].append(facelet_colors[ff][fy][fx])

        corners: list[Corner | None] = [None] * len(cubicles.corners)
        for i, corner in enumerate(cubicles.corners):
            colors = corner_colors[i]
            for origin_i, origin_corner in enumerate(cubicles.corners):
                origin_colors = cubicle_colors(n, *corner_to_coord(n, origin_corner))
                if set(origin_colors) == set(colors):
                    x, y, z, _, _ = corner
                    r = origin_colors.index(colors[0])
                    c = corner_clockwise(corner) != corner_clockwise(origin_corner)
                    assert corners[origin_i] is None
                    corners[origin_i] = (x, y, z, r, c)

        centers: list[Center | None] = [None] * len(cubicles.centers)
        for i, center in enumerate(cubicles.centers):
            colors = center_colors[i]
            for origin_i, origin_center in enumerate(cubicles.centers):
                origin_colors = cubicle_colors(n, *center_to_coord(n, origin_center))
                if set(origin_colors) == set(colors):
                    a, h = center
                    assert centers[origin_i] is None
                    centers[origin_i] = (a, h)

        edges: list[Edge | None] = [None] * len(cubicles.edges)
        for i, edge in enumerate(cubicles.edges):
            colors = edge_colors[i]
            for origin_i, origin_edge in enumerate(cubicles.edges):
                origin_colors = cubicle_colors(n, *edge_to_coord(origin_edge))
                if set(origin_colors) == set(colors):
                    x, y, z, _ = edge
                    r = origin_colors.index(colors[0]) == 1
                    assert edges[origin_i] is None
                    edges[origin_i] = (x, y, z, r)

        assert None not in corners
        assert None not in centers
        assert None not in edges
        return Puzzle(n, cubicles, corners, centers, edges)  # type: ignore

    @staticmethod
    def finished(n: int):
        cubicles = Cubicles(n)
        return Puzzle(
            n,
            cubicles,
            cubicles.corners.copy(),
            cubicles.centers.copy(),
            cubicles.edges.copy(),
        )

    def copy(self):
        return Puzzle(
            self.n,
            self.cubicles,
            self.corners.copy(),
            self.centers.copy(),
            self.edges.copy(),
        )

    def __eq__(self, other: "Puzzle"):
        return (
            self.n == other.n
            and self.corners == other.corners
            and self.centers == other.centers
            and self.edges == other.edges
        )

    def __hash__(self):
        return hash(
            (
                self.n,
                tuple(self.corners),
                tuple(self.centers),
                tuple(self.edges),
            )
        )

    def to_str(self):
        facelet_colors = [
            [
                [self.facelet_color(f, y, x) for x in range(self.n)]
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

    def execute_move(self, ma: int, mi: int, md: int):
        for i, (x, y, z, r, c) in enumerate(self.corners):
            self.corners[i] = (
                move_mappers.corner_x(self.n, x, y, z, ma, mi, md),
                move_mappers.corner_y(self.n, x, y, z, ma, mi, md),
                move_mappers.corner_z(self.n, x, y, z, ma, mi, md),
                move_mappers.corner_r(self.n, x, z, r, c, ma, mi, md),
                move_mappers.corner_c(self.n, x, y, z, c, ma, mi, md),
            )

        for i, (a, h) in enumerate(self.centers):
            self.centers[i] = (
                move_mappers.center_a(a, ma, mi, md),
                move_mappers.center_h(a, h, ma, mi, md),
            )

        for i, (x, y, z, r) in enumerate(self.edges):
            self.edges[i] = (
                move_mappers.edge_x(self.n, x, y, z, ma, mi, md),
                move_mappers.edge_y(self.n, x, y, z, ma, mi, md),
                move_mappers.edge_z(self.n, x, y, z, ma, mi, md),
                move_mappers.edge_r(x, y, z, r, ma, mi, md),
            )

    def facelet_color(self, ff: int, fy: int, fx: int) -> int:
        x, y, z = facelet_cubicle(self.n, ff, fy, fx)

        match cubicle_type(self.n, x, y, z):
            case 0:
                for i, corner in enumerate(self.cubicles.corners):
                    origin_corner = self.corners[i]
                    _, _, _, o_r, o_c = origin_corner
                    o_x, o_y, o_z = corner_to_coord(self.n, origin_corner)
                    if (o_x, o_y, o_z) == (x, y, z):
                        colors = cubicle_colors(
                            self.n, *corner_to_coord(self.n, corner)
                        )
                        first_color = colors[o_r]

                        # If the cubie's direction has changed, reverse the
                        # color list to adhere to ordering.
                        if o_c:
                            colors.reverse()

                        # Rotate until the color is in the first slot.
                        while colors[0] != first_color:
                            rotate_list(colors)

                        # Get the facelet index of the facelet in question and return.
                        fi = cubicle_facelets(self.n, x, y, z).index((ff, fy, fx))
                        return colors[fi]
            case 1:
                for i, center in enumerate(self.cubicles.centers):
                    origin_center = self.centers[i]
                    o_x, o_y, o_z = center_to_coord(self.n, origin_center)
                    if (o_x, o_y, o_z) == (x, y, z):
                        colors = cubicle_colors(
                            self.n, *center_to_coord(self.n, center)
                        )
                        return colors[0]
            case 2:
                for i, edge in enumerate(self.cubicles.edges):
                    origin_edge = self.edges[i]
                    _, _, _, o_r = origin_edge
                    o_x, o_y, o_z = edge_to_coord(origin_edge)
                    if (o_x, o_y, o_z) == (x, y, z):
                        colors = cubicle_colors(self.n, *edge_to_coord(edge))
                        fi = cubicle_facelets(self.n, x, y, z).index((ff, fy, fx))
                        if not o_r:
                            return colors[fi]
                        else:
                            return colors[1 - fi]

        raise Exception(f"invalid facelet: ({ff},{fy},{fx})")

    def print(self):
        facelet_size = 48
        image_size = (3 * self.n * facelet_size, 4 * self.n * facelet_size)
        im = Image.new(mode="RGB", size=image_size)
        draw = ImageDraw.Draw(im)

        def draw_face(start_x: int, start_y: int, f: int, rotate=False):
            for y in range(self.n):
                for x in range(self.n):
                    if rotate:
                        color = self.facelet_color(f, self.n - 1 - y, self.n - 1 - x)
                    else:
                        color = self.facelet_color(f, y, x)
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
