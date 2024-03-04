from PIL import Image, ImageDraw
import ast


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
    assert False


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
    assert False


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
    assert False


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


def list_corners(n: int) -> list[tuple[int, int, int]]:
    return [
        (x, y, z)
        for x in range(n)
        for y in range(n)
        for z in range(n)
        if cubie_type(n, x, y, z) == 0
    ]


def list_centers(n: int) -> list[tuple[int, int, int]]:
    return [
        (x, y, z)
        for x in range(n)
        for y in range(n)
        for z in range(n)
        if cubie_type(n, x, y, z) == 1
    ]


def list_edges(n: int) -> list[tuple[int, int, int]]:
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
    # TODO: invent a way to have rotation values in domain {0, 1} instead of
    # {0, 1, 2}. This is now just a copy of the corner rotation mapping.
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

    @staticmethod
    def from_str(s: str):
        n_raw, coords_raw, rots_raw = s.split("\t")
        n = int(n_raw)
        coords = ast.literal_eval(coords_raw)
        rots = ast.literal_eval(rots_raw)
        return State(n, coords, rots)

    def to_str(self):
        return "\t".join((str(self.n), str(self.coords), str(self.rots)))

    def copy(self):
        return State(
            self.n,
            [
                [[self.coords[x][y][z] for z in range(self.n)] for y in range(self.n)]
                for x in range(self.n)
            ],
            [
                [[self.rots[x][y][z] for z in range(self.n)] for y in range(self.n)]
                for x in range(self.n)
            ],
        )

    @staticmethod
    def finished(n: int):
        coords = [
            [
                [
                    (x, y, z) if cubie_type(n, x, y, z) != -1 else (-1, -1, -1)
                    for z in range(n)
                ]
                for y in range(n)
            ]
            for x in range(n)
        ]
        rots = [
            [
                [
                    0
                    if cubie_type(n, x, y, z) == 0 or cubie_type(n, x, y, z) == 2
                    else -1
                    for z in range(n)
                ]
                for y in range(n)
            ]
            for x in range(n)
        ]
        return State(n, coords, rots)

    def execute_move(self, ma: int, mi: int, md: int):
        new = self.copy()
        for x in range(self.n):
            for y in range(self.n):
                for z in range(self.n):
                    new_x, new_y, new_z = coord_mapping(self.n, x, y, z, ma, mi, md)
                    new.coords[new_x][new_y][new_z] = self.coords[x][y][z]
                    match cubie_type(self.n, x, y, z):
                        case 0:
                            rot = self.rots[x][y][z]
                            new_rot = corner_rotation_mapping(x, y, z, rot, ma, mi, md)
                            new.rots[new_x][new_y][new_z] = new_rot[0]
                        case 2:
                            rot = self.rots[x][y][z]
                            new_rot = edge_rotation_mapping(x, y, z, rot, ma, mi, md)
                            new.rots[new_x][new_y][new_z] = new_rot[0]
        return new

    @staticmethod
    def from_compact_str(s: str):
        return State(0, [], [])  # TODO

    def to_compact_str(self):
        return ""  # TODO

    def color(self, f: int, y: int, x: int) -> int:
        return -1  # TODO

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
                        fill=color_name(self.color(f, y, x)),
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
