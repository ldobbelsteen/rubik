import sys

from PIL import Image, ImageDraw

from misc import rotate_list

# The only supported values for n.
SUPPORTED_NS = {2, 3}

# The global face ordering by which the desired rotation ordering is achieved.
# First come top/bottom, second frontback and third left/right.
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


def move_name(n: int, ma: int, mi: int, md: int) -> str:
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


def cubie_type(n: int, x: int, y: int, z: int):
    """Determine the type of a cubie by its coordinates. 0 = corner, 1 = center,
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


def cubie_colors(n: int, x: int, y: int, z: int) -> list[int]:
    """Get the list of colors of a cubie in a finished cube. The list is sorted
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


def corner_clockwise(n: int, x: int, y: int, z: int) -> bool:
    """Determine whether a corner cubie's colors are labeled clockwise or not."""

    # Only these four are clockwise. The other four are counterclockwise.
    return (
        (x == 0 and y == n - 1 and z == 0)
        or (x == 0 and y == 0 and z == n - 1)
        or (x == n - 1 and y == 0 and z == 0)
        or (x == n - 1 and y == n - 1 and z == n - 1)
    )


def cubie_facelets(n: int, x: int, y: int, z: int) -> list[tuple[int, int, int]]:
    """Get the list of facelets of a cubie. The list is sorted by the global
    face ordering."""
    facelets = []
    for ff in FACE_ORDERING:
        for fy in range(n):
            for fx in range(n):
                if facelet_cubie(n, ff, fy, fx) == (x, y, z):
                    facelets.append((ff, fy, fx))
    return facelets


def facelet_cubie(n: int, f: int, y: int, x: int) -> tuple[int, int, int]:
    """Get the cubie on which a facelet is located."""
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


# def coord_mapping(
#     n: int, x: int, y: int, z: int, ma: int, mi: int, md: int
# ) -> tuple[int, int, int]:
#     if ma == 0 and mi == y:
#         if md == 0:
#             return (z, y, n - 1 - x)  # clockwise
#         elif md == 1:
#             return (n - 1 - z, y, x)  # counterclockwise
#         elif md == 2:
#             return (n - 1 - x, y, n - 1 - z)  # 180 degree
#     elif ma == 1 and mi == x:
#         if md == 0:
#             return (x, n - 1 - z, y)  # counterclockwise
#         elif md == 1:
#             return (x, z, n - 1 - y)  # clockwise
#         elif md == 2:
#             return (x, n - 1 - y, n - 1 - z)  # 180 degree
#     elif ma == 2 and mi == z:
#         if md == 0:
#             return (y, n - 1 - x, z)  # clockwise
#         elif md == 1:
#             return (n - 1 - y, x, z)  # counterclockwise
#         elif md == 2:
#             return (n - 1 - x, n - 1 - y, z)  # 180 degree
#     return (x, y, z)


def x_mapping(n: int, x: int, y: int, z: int, ma: int, mi: int, md: int) -> int:
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


def y_mapping(n: int, x: int, y: int, z: int, ma: int, mi: int, md: int) -> int:
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


def z_mapping(n: int, x: int, y: int, z: int, ma: int, mi: int, md: int) -> int:
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


def corner_r_mapping(x: int, z: int, r: int, c: bool, ma: int, mi: int, md: int) -> int:
    if md != 2:
        if ma == 1 and mi == x:
            if c:
                return (r - 1) % 3
            else:
                return (r + 1) % 3
        elif ma == 2 and mi == z:
            if c:
                return (r + 1) % 3
            else:
                return (r - 1) % 3
    return r


def corner_c_mapping(
    x: int, y: int, z: int, c: bool, ma: int, mi: int, md: int
) -> bool:
    if md != 2 and (
        (ma == 0 and mi == y) or (ma == 1 and mi == x) or (ma == 2 and mi == z)
    ):
        return not c
    return c


def edge_r_mapping(
    n: int, x: int, y: int, z: int, r: bool, ma: int, mi: int, md: int
) -> bool:
    assert n == 3
    if md != 2 and (
        (ma == 2 and mi == z)
        or (mi == 1 and ((ma == 0 and mi == y) or (ma == 1 and mi == x)))
    ):
        return not r
    return r


class Cubies:
    def __init__(self, n: int):
        self.corners = [
            (x, y, z)
            for x in range(n)
            for y in range(n)
            for z in range(n)
            if cubie_type(n, x, y, z) == 0
        ]
        self.centers = [
            (x, y, z)
            for x in range(n)
            for y in range(n)
            for z in range(n)
            if cubie_type(n, x, y, z) == 1
        ]
        self.edges = [
            (x, y, z)
            for x in range(n)
            for y in range(n)
            for z in range(n)
            if cubie_type(n, x, y, z) == 2
        ]


class Puzzle:
    def __init__(
        self,
        n: int,
        cubies: Cubies,
        corners: list[tuple[int, int, int, int, bool]],
        centers: list[tuple[int, int, int]],
        edges: list[tuple[int, int, int, bool]],
    ):
        self.n = n
        self.cubies = cubies
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
            ],  # Extract facelet colors from string.
        )

    @staticmethod
    def from_facelet_colors(n: int, facelet_colors: list[list[list[int]]]):
        finished = Puzzle.finished(n)
        cubies = finished.cubies

        # Extract the cubie colors from the facelet representation.
        corner_colors = [[] for _ in cubies.corners]
        center_colors = [[] for _ in cubies.centers]
        edge_colors = [[] for _ in cubies.edges]
        for ff in FACE_ORDERING:
            for fy in range(n):
                for fx in range(n):
                    x, y, z = facelet_cubie(n, ff, fy, fx)
                    match cubie_type(n, x, y, z):
                        case 0:
                            corner_colors[cubies.corners.index((x, y, z))].append(
                                facelet_colors[ff][fy][fx]
                            )
                        case 1:
                            center_colors[cubies.centers.index((x, y, z))].append(
                                facelet_colors[ff][fy][fx]
                            )
                        case 2:
                            edge_colors[cubies.edges.index((x, y, z))].append(
                                facelet_colors[ff][fy][fx]
                            )

        corners = finished.corners
        for i, (x, y, z) in enumerate(cubies.corners):
            colors = corner_colors[i]
            for o_i, (o_x, o_y, o_z) in enumerate(cubies.corners):
                origin_colors = cubie_colors(n, o_x, o_y, o_z)
                if set(origin_colors) == set(colors):
                    r = origin_colors.index(colors[0])
                    c = corner_clockwise(n, x, y, z) != corner_clockwise(
                        n, o_x, o_y, o_z
                    )
                    corners[o_i] = (x, y, z, r, c)

        centers = finished.centers
        for i, (x, y, z) in enumerate(cubies.centers):
            colors = center_colors[i]
            for o_i, (o_x, o_y, o_z) in enumerate(cubies.centers):
                origin_colors = cubie_colors(n, o_x, o_y, o_z)
                if set(origin_colors) == set(colors):
                    r = origin_colors.index(colors[0])
                    centers[o_i] = (x, y, z)

        edges = finished.edges
        for i, (x, y, z) in enumerate(cubies.edges):
            colors = edge_colors[i]
            for o_i, (o_x, o_y, o_z) in enumerate(cubies.edges):
                origin_colors = cubie_colors(n, o_x, o_y, o_z)
                if set(origin_colors) == set(colors):
                    r = origin_colors.index(colors[0])
                    edges[o_i] = (x, y, z, r == 1)

        return Puzzle(n, cubies, corners, centers, edges)

    @staticmethod
    def finished(n: int):
        cubies = Cubies(n)
        return Puzzle(
            n,
            cubies,
            [(x, y, z, 0, False) for x, y, z in cubies.corners],
            [(x, y, z) for x, y, z in cubies.centers],
            [(x, y, z, False) for x, y, z in cubies.edges],
        )

    def copy(self):
        return Puzzle(
            self.n,
            self.cubies,
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
            (self.n, tuple(self.corners), tuple(self.centers), tuple(self.edges))
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
                x_mapping(self.n, x, y, z, ma, mi, md),
                y_mapping(self.n, x, y, z, ma, mi, md),
                z_mapping(self.n, x, y, z, ma, mi, md),
                corner_r_mapping(x, z, r, c, ma, mi, md),
                corner_c_mapping(x, y, z, c, ma, mi, md),
            )

        for i, (x, y, z) in enumerate(self.centers):
            self.centers[i] = (
                x_mapping(self.n, x, y, z, ma, mi, md),
                y_mapping(self.n, x, y, z, ma, mi, md),
                z_mapping(self.n, x, y, z, ma, mi, md),
            )

        for i, (x, y, z, r) in enumerate(self.edges):
            self.edges[i] = (
                x_mapping(self.n, x, y, z, ma, mi, md),
                y_mapping(self.n, x, y, z, ma, mi, md),
                z_mapping(self.n, x, y, z, ma, mi, md),
                edge_r_mapping(self.n, x, y, z, r, ma, mi, md),
            )

    def facelet_color(self, ff: int, fy: int, fx: int) -> int:
        x, y, z = facelet_cubie(self.n, ff, fy, fx)

        if (x, y, z) in self.cubies.corners:
            for c_i, (c_x, c_y, c_z) in enumerate(self.cubies.corners):
                o_x, o_y, o_z, o_r, o_c = self.corners[c_i]
                if (o_x, o_y, o_z) == (x, y, z):
                    colors = cubie_colors(self.n, c_x, c_y, c_z)
                    first_color = colors[o_r]

                    # If the cubie's direction has changed, reverse the
                    # color list to adhere to ordering.
                    if o_c:
                        colors.reverse()

                    # Rotate until the color is in the first slot.
                    while colors[0] != first_color:
                        rotate_list(colors)

                    # Get the facelet index of the facelet in question and return.
                    fi = cubie_facelets(self.n, x, y, z).index((ff, fy, fx))
                    return colors[fi]

        if (x, y, z) in self.cubies.centers:
            for c_i, (c_x, c_y, c_z) in enumerate(self.cubies.centers):
                o_x, o_y, o_z = self.centers[c_i]
                if (o_x, o_y, o_z) == (x, y, z):
                    colors = cubie_colors(self.n, c_x, c_y, c_z)
                    return colors[0]

        if (x, y, z) in self.cubies.edges:
            for c_i, (c_x, c_y, c_z) in enumerate(self.cubies.edges):
                o_x, o_y, o_z, o_r = self.edges[c_i]
                if (o_x, o_y, o_z) == (x, y, z):
                    colors = cubie_colors(self.n, c_x, c_y, c_z)
                    fi = cubie_facelets(self.n, x, y, z).index((ff, fy, fx))
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


# e.g. python puzzle.py ./puzzles/n2-random4.txt
if __name__ == "__main__":
    Puzzle.from_file(sys.argv[1]).print()
