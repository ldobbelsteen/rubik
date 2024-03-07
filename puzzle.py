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


# Global face ordering to guarantee rotation ordering.
# FIRST top/bottom, SECOND front/back, THIRD left/right.
FACE_ORDERING = [4, 5, 0, 2, 3, 1]


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
    """Get the list of colors of a cubicle in a finished cube. The list is
    sorted by the global face ordering."""
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
    raise Exception(f"invalid face: {f}")


def facelet_colors_to_encoding(n: int, facelet_colors: list[list[list[int]]]):
    """Convert facelet colors of a cube to our coord and rotation encoding."""

    # Extract the cubie colors from the facelet representation.
    cubie_colors = [[[[] for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for ff in FACE_ORDERING:
        for fy in range(n):
            for fx in range(n):
                x, y, z = facelet_cubicle(n, ff, fy, fx)
                cubie_colors[x][y][z].append(facelet_colors[ff][fy][fx])

    def find_cubicle(colors: list[int]) -> tuple[int, int, int, int]:
        for cx in range(n):
            for cy in range(n):
                for cz in range(n):
                    labeling = cubicle_colors(n, cx, cy, cz)
                    if set(colors) == set(labeling):
                        return cx, cy, cz, labeling.index(colors[0])
        raise Exception(f"invalid color list: {colors}")

    # Extract cubie coords and rotations by finding the color sets.
    coords: list[list[list[tuple[int, int, int]]]] = [
        [[(-1, -1, -1) for _ in range(n)] for _ in range(n)] for _ in range(n)
    ]
    rotations = [[[-1 for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for x in range(n):
        for y in range(n):
            for z in range(n):
                type = cubicle_type(n, x, y, z)
                if type == -1:
                    continue

                colors = cubie_colors[x][y][z]
                cx, cy, cz, r = find_cubicle(colors)
                coords[cx][cy][cz] = (x, y, z)

                if type == 0 or type == 2:
                    rotations[cx][cy][cz] = r

    return coords, rotations


def list_corners(n: int) -> list[tuple[int, int, int]]:
    """Get a list of coordinates of the corners."""
    return [
        (x, y, z)
        for x in range(n)
        for y in range(n)
        for z in range(n)
        if cubicle_type(n, x, y, z) == 0
    ]


def list_centers(n: int) -> list[tuple[int, int, int]]:
    """Get a list of coordinates of the centers."""
    return [
        (x, y, z)
        for x in range(n)
        for y in range(n)
        for z in range(n)
        if cubicle_type(n, x, y, z) == 1
    ]


def list_edges(n: int) -> list[tuple[int, int, int]]:
    """Get a list of coordinates of the edges."""
    return [
        (x, y, z)
        for x in range(n)
        for y in range(n)
        for z in range(n)
        if cubicle_type(n, x, y, z) == 2
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
    x: int, z: int, r: int, ma: int, mi: int, md: int
) -> tuple[int]:
    if ma == 1:
        if mi == x:
            if md != 2:
                if r == 0:
                    return (1,)
                elif r == 1:
                    return (2,)
                elif r == 2:
                    return (0,)
    elif ma == 2:
        if mi == z:
            if md != 2:
                if r == 0:
                    return (2,)
                elif r == 1:
                    return (0,)
                elif r == 2:
                    return (1,)
    return (r,)


def edge_rotation_mapping(z: int, r: int, ma: int, mi: int, md: int) -> tuple[int]:
    if ma == 2:
        if mi == z:
            if md != 2:
                if r == 0:
                    return (1,)
                elif r == 1:
                    return (0,)
    return (r,)


class Puzzle:
    def __init__(
        self,
        n: int,
        coords: list[list[list[tuple[int, int, int]]]],
        rotations: list[list[list[int]]],
    ):
        self.n = n
        self.coords = coords
        self.rotations = rotations

    def facelet_color(self, ff: int, fy: int, fx: int) -> int:
        x, y, z = facelet_cubicle(self.n, ff, fy, fx)
        for cx in range(self.n):
            for cy in range(self.n):
                for cz in range(self.n):
                    if self.coords[cx][cy][cz] == (x, y, z):
                        colors = cubicle_colors(self.n, cx, cy, cz)
                        type = cubicle_type(self.n, cx, cy, cz)

                        # NOTE: debugging
                        if ff == 0 and fy == 0 and fx == 1:
                            print(f"x,y,z {(x, y, z)}")
                            print(f"cx,cy,cz {(cx, cy, cz)}")
                            print(f"type {type}")
                            print(colors)

                        if type == 1:
                            return colors[0]
                        elif type == 2:
                            assert len(colors) == 2

                            r = self.rotations[cx][cy][cz]
                            assert r == 0 or r == 1

                            fi = cubicle_facelets(self.n, x, y, z).index((ff, fy, fx))
                            assert fi == 0 or fi == 1

                            if fi == 0:
                                return colors[r]
                            elif r == 0:
                                return colors[1]
                            else:
                                return colors[0]
                        elif type == 0:
                            r = self.rotations[cx][cy][cz]
                            fi = cubicle_facelets(self.n, x, y, z).index((ff, fy, fx))
                            ci = (fi + r) % len(colors)

                            # NOTE: debugging
                            if ff == 0 and fy == 0 and fx == 1:
                                print(f"r {r}")
                                print(f"fi {fi}")
                                print(f"ci {ci}")
                                print(f"cci {colors[ci]}")

                            return colors[ci]
        raise Exception(f"invalid facelet: ({ff},{fy},{fx})")

    @staticmethod
    def from_str(s: str):
        n = int((len(s) / 6) ** 0.5)
        assert len(s) == 6 * n * n

        # Extract the facelet colors from the string.
        facelet_colors = [
            [
                [int(s[f * n * n + y * n + x]) for x in range(n)]
                for y in reversed(range(n))
            ]
            for f in range(6)
        ]

        print(facelet_colors)  # NOTE: debugging

        coords, rotations = facelet_colors_to_encoding(n, facelet_colors)
        return Puzzle(n, coords, rotations)

    def to_str(self):
        facelet_colors = [
            [
                [self.facelet_color(f, y, x) for x in range(self.n)]
                for y in range(self.n)
            ]
            for f in range(6)
        ]

        print(facelet_colors)  # NOTE: debugging

        colors = [
            facelet_colors[f][y][x]
            for f in range(6)
            for y in reversed(range(self.n))
            for x in range(self.n)
        ]
        return "".join(map(str, colors))

    @staticmethod
    def finished(n: int):
        return Puzzle(
            n,
            [
                [
                    [
                        (x, y, z) if cubicle_type(n, x, y, z) != -1 else (-1, -1, -1)
                        for z in range(n)
                    ]
                    for y in range(n)
                ]
                for x in range(n)
            ],
            [
                [
                    [
                        0
                        if cubicle_type(n, x, y, z) == 0
                        or cubicle_type(n, x, y, z) == 2
                        else -1
                        for z in range(n)
                    ]
                    for y in range(n)
                ]
                for x in range(n)
            ],
        )

    def execute_move(self, ma: int, mi: int, md: int):
        for x in range(self.n):
            for y in range(self.n):
                for z in range(self.n):
                    type = cubicle_type(self.n, x, y, z)

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
                        self.rotations[x][y][z] = corner_rotation_mapping(
                            self.coords[x][y][z][0],
                            self.coords[x][y][z][2],
                            self.rotations[x][y][z],
                            ma,
                            mi,
                            md,
                        )[0]
                    elif type == 2:
                        self.rotations[x][y][z] = edge_rotation_mapping(
                            self.coords[x][y][z][2],
                            self.rotations[x][y][z],
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
