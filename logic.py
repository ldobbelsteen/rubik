from PIL import Image, ImageDraw


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
    raise Exception(f"invalid face: {f}")


def first_cubie_facelet(n: int, cx: int, cy: int, cz: int) -> tuple[int, int, int]:
    """Deterministically get the first facelet of a cubie."""
    for f in range(6):
        for y in range(n):
            for x in range(n):
                if facelet_cubie(n, f, y, x) == (cx, cy, cz):
                    return (f, y, x)
    raise Exception(f"invalid cubie: ({cx},{cy},{cz})")


def cubie_colors_finished(n: int, x: int, y: int, z: int) -> set[int]:
    """Get the set of colors of a cubie in a finished cube."""
    result = set()
    if x == 0:
        result.add(3)
    if x == n - 1:
        result.add(1)
    if y == 0:
        result.add(5)
    if y == n - 1:
        result.add(4)
    if z == 0:
        result.add(0)
    if z == n - 1:
        result.add(2)
    return result


def rotate_list(ls: list[int]):
    """Rotate a list by appending its last element to the front."""
    last = ls.pop()
    ls.insert(0, last)


def list_corners(n: int) -> list[tuple[int, int, int]]:
    """Get a list of coordinates of the corners."""
    return [
        (x, y, z)
        for x in range(n)
        for y in range(n)
        for z in range(n)
        if cubie_type(n, x, y, z) == 0
    ]


def list_centers(n: int) -> list[tuple[int, int, int]]:
    """Get a list of coordinates of the centers."""
    return [
        (x, y, z)
        for x in range(n)
        for y in range(n)
        for z in range(n)
        if cubie_type(n, x, y, z) == 1
    ]


def list_edges(n: int) -> list[tuple[int, int, int]]:
    """Get a list of coordinates of the edges."""
    return [
        (x, y, z)
        for x in range(n)
        for y in range(n)
        for z in range(n)
        if cubie_type(n, x, y, z) == 2
    ]


def coord_mapping(
    n: int, x: int, y: int, z: int, ma: int, mi: int, md: int
) -> tuple[int, int, int]:
    if ma == 0:
        if mi == y:
            if md == 0:
                return (z, y, n - 1 - x)  # clockwise
            elif md == 1:
                return (n - 1 - z, y, x)  # counterclockwise
            elif md == 2:
                return (n - 1 - x, y, n - 1 - z)  # 180 degree
    elif ma == 1:
        if mi == x:
            if md == 0:
                return (x, n - 1 - z, y)  # counterclockwise
            elif md == 1:
                return (x, z, n - 1 - y)  # clockwise
            elif md == 2:
                return (x, n - 1 - y, n - 1 - z)  # 180 degree
    elif ma == 2:
        if mi == z:
            if md == 0:
                return (y, n - 1 - x, z)  # clockwise
            elif md == 1:
                return (n - 1 - y, x, z)  # counterclockwise
            elif md == 2:
                return (n - 1 - x, n - 1 - y, z)  # 180 degree
    return (x, y, z)


def corner_rotation_mapping(
    x: int, y: int, z: int, r: int, ma: int, mi: int, md: int
) -> tuple[int]:
    if ma == 0:
        if mi == y:
            if md != 2:
                if r == 0:
                    return (2,)
                elif r == 2:
                    return (0,)
    elif ma == 1:
        if mi == x:
            if md != 2:
                if r == 1:
                    return (2,)
                elif r == 2:
                    return (1,)
    elif ma == 2:
        if mi == z:
            if md != 2:
                if r == 0:
                    return (1,)
                elif r == 1:
                    return (0,)
    return (r,)


def edge_rotation_mapping(
    x: int, y: int, z: int, r: int, ma: int, mi: int, md: int
) -> tuple[int]:
    if ma == 0:
        if mi == y:
            if md != 2:
                if r == 0:
                    return (2,)
                elif r == 2:
                    return (0,)
    elif ma == 1:
        if mi == x:
            if md != 2:
                if r == 1:
                    return (2,)
                elif r == 2:
                    return (1,)
    elif ma == 2:
        if mi == z:
            if md != 2:
                if r == 0:
                    return (1,)
                elif r == 1:
                    return (0,)
    return (r,)


class State:
    def __init__(
        self,
        n: int,
        coords: list[list[list[tuple[int, int, int]]]],
        rots: list[list[list[int]]],
    ):
        self.n = n
        self.coords = coords
        self.rots = rots

    def facelet_color(self, f: int, y: int, x: int) -> int:
        return -1  # TODO

    def cubie_colors(self, x: int, y: int, z: int) -> list[int]:
        for ox in range(self.n):
            for oy in range(self.n):
                for oz in range(self.n):
                    if self.coords[ox][oy][oz] == (x, y, z):
                        colors = sorted(cubie_colors_finished(self.n, ox, oy, oz))

                        if self.rots[ox][oy][oz] != -1:
                            for _ in range(self.rots[ox][oy][oz]):
                                rotate_list(colors)

                        return colors

        raise Exception(f"invalid cubie coords: ({x},{y},{z})")

    @staticmethod
    def from_str(s: str):
        n = int((len(s) / 6) ** 0.5)
        assert len(s) == 6 * n * n

        # Extract the facelet colors from the string.
        facelet_colors = [
            [
                [int(s[f * n * n + (n - 1 - y) * n + x]) for x in range(n)]
                for y in range(n)
            ]
            for f in range(6)
        ]

        # Extract the cubie colors from the facelet representation.
        cubie_colors = [[[set() for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for f in range(6):
            for y in range(n):
                for x in range(n):
                    cx, cy, cz = facelet_cubie(n, f, y, x)
                    cubie_colors[cx][cy][cz].add(facelet_colors[f][y][x])

        finished = State.finished(n)

        def find_origin_cubie(
            x: int, y: int, z: int, colors: set[int]
        ) -> tuple[tuple[int, int, int], int | None]:
            for ox in range(n):
                for oy in range(n):
                    for oz in range(n):
                        origin_colors = finished.cubie_colors(ox, oy, oz)
                        if colors == set(origin_colors):
                            assert len(colors) > 0

                            if len(colors) == 1:
                                return (ox, oy, oz), None

                            ff, fy, fx = first_cubie_facelet(n, x, y, z)
                            for r in range(len(origin_colors)):
                                if origin_colors[0] == facelet_colors[ff][fy][fx]:
                                    return (ox, oy, oz), r
                                rotate_list(origin_colors)

                            raise Exception(f"invalid color set permutation: {colors}")

            raise Exception(f"invalid color set: {colors}")

        # Extract cubie coords and rotations by finding the color sets in a finished cube.
        coords: list[list[list[tuple[int, int, int]]]] = [
            [[(-1, -1, -1) for z in range(n)] for y in range(n)] for x in range(n)
        ]
        rots: list[list[list[int]]] = [
            [[-1 for _ in range(n)] for _ in range(n)] for _ in range(n)
        ]
        for x in range(n):
            for y in range(n):
                for z in range(n):
                    colors = cubie_colors[x][y][z]
                    if len(colors) == 0:
                        continue
                    (cx, cy, cz), r = find_origin_cubie(x, y, z, colors)
                    coords[cx][cy][cz] = (x, y, z)
                    if r is not None:
                        rots[cx][cy][cz] = r

        return State(n, coords, rots)

    def to_str(self):
        colors = [
            self.facelet_color(f, y, x)
            for f in range(6)
            for y in range(self.n)
            for x in range(self.n)
        ]
        return "".join(map(str, colors))

    @staticmethod
    def finished(n: int):
        coords = [[[(x, y, z) for z in range(n)] for y in range(n)] for x in range(n)]
        rots = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        return State(n, coords, rots)

    def execute_move(self, ma: int, mi: int, md: int):
        for x in range(self.n):
            for y in range(self.n):
                for z in range(self.n):
                    type = cubie_type(self.n, x, y, z)

                    # Map the coordinates.
                    if type != -1:
                        self.coords[x][y][z] = coord_mapping(
                            self.n,
                            self.coords[x][y][z][0],
                            self.coords[x][y][z][1],
                            self.coords[x][y][z][2],
                            ma,
                            mi,
                            md,
                        )

                    # Map the rotation.
                    if type == 0:
                        self.rots[x][y][z] = corner_rotation_mapping(
                            self.coords[x][y][z][0],
                            self.coords[x][y][z][1],
                            self.coords[x][y][z][2],
                            self.rots[x][y][z],
                            ma,
                            mi,
                            md,
                        )[0]
                    elif type == 2:
                        self.rots[x][y][z] = edge_rotation_mapping(
                            self.coords[x][y][z][0],
                            self.coords[x][y][z][1],
                            self.coords[x][y][z][2],
                            self.rots[x][y][z],
                            ma,
                            mi,
                            md,
                        )[0]

    def print(self):
        facelet_size = 48
        image_size = (3 * self.n * facelet_size, 4 * self.n * facelet_size)
        im = Image.new(mode="RGB", size=image_size)
        draw = ImageDraw.Draw(im)

        def draw_face(start_x: int, start_y: int, f: int):
            for y in range(self.n):
                for x in range(self.n):
                    draw.rectangle(
                        (
                            start_x + (x * facelet_size),
                            start_y + (y * facelet_size),
                            start_x + ((x + 1) * facelet_size),
                            start_y + ((y + 1) * facelet_size),
                        ),
                        fill=color_name(self.facelet_color(f, y, x)),
                        outline="black",
                        width=4,
                    )

        draw_face(1 * self.n * facelet_size, 1 * self.n * facelet_size, 0)
        draw_face(2 * self.n * facelet_size, 1 * self.n * facelet_size, 1)
        draw_face(1 * self.n * facelet_size, 3 * self.n * facelet_size, 2)
        draw_face(0 * self.n * facelet_size, 1 * self.n * facelet_size, 3)
        draw_face(1 * self.n * facelet_size, 0 * self.n * facelet_size, 4)
        draw_face(1 * self.n * facelet_size, 2 * self.n * facelet_size, 5)
        im.show()
