import argparse
import os

from PIL import Image, ImageDraw

from state import CornerState, EdgeState, Move, cubie_type
from tools import natural_sorted, rotate_list

PUZZLE_DIR = "./puzzles"

# The global face ordering by which the desired color ordering is achieved. First come
# top/bottom, second front/back and third left/right. The indices correspond to the face
# indices in the face_name function. This ordering determines the order of the facelets
# associated with a cubie. For example, the BRU corner has its facelets ordered by first
# its up facelet, second its back facelet and third its right facelet.
FACE_ORDERING: list[int] = [4, 5, 0, 2, 3, 1]

# The default center cubie colors of a cube. The indices correspond to the face indices
# in the face_name function. The values correspond to the color indices in
# the color_name function.
DEFAULT_CENTER_COLORS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def face_name(f: int) -> str:
    """Convert a face index to its canonical name."""
    match f:
        case 0:
            return "F"
        case 1:
            return "R"
        case 2:
            return "B"
        case 3:
            return "L"
        case 4:
            return "U"
        case 5:
            return "D"
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


def all_puzzles_names() -> list[str]:
    """List all puzzles in the puzzle directory."""
    return natural_sorted([filename for filename in os.listdir(PUZZLE_DIR)])


def cubie_colors(
    n: int,
    x: int,
    y: int,
    z: int,
    center_colors: tuple[int, ...],
) -> list[int]:
    """Get the list of colors of a cubie in a finished puzzle. The list is sorted
    by the global face ordering. Also takes into account the center colors of the
    finished puzzle.
    """
    faces = set()
    if x == 0:
        faces.add(3)
    if x == n - 1:
        faces.add(1)
    if y == 0:
        faces.add(5)
    if y == n - 1:
        faces.add(4)
    if z == 0:
        faces.add(0)
    if z == n - 1:
        faces.add(2)
    sorted = [f for f in FACE_ORDERING if f in faces]
    return [center_colors[f] for f in sorted]


def cubie_facelets(n: int, x: int, y: int, z: int) -> list[tuple[int, int, int]]:
    """Get the list of facelets of a cubie. The list is sorted by the global face
    ordering. This is done by iterating over all facelets (in the global face order)
    and accumulating all facelets belonging to the given cubie.
    """
    return [
        (ff, fy, fx)
        for ff in FACE_ORDERING
        for fy in range(n)
        for fx in range(n)
        if facelet_cubie(n, ff, fy, fx) == (x, y, z)
    ]


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


def extract_center_colors(
    n: int,
    facelet_colors: list[list[list[int]]],
) -> tuple[int, ...]:
    """Extract the center colors of each face from the facelet color repesentation
    of a puzzle. Only works for n == 3.
    """
    assert n == 3
    return tuple([facelet_colors[f][1][1] for f in range(6)])


class Puzzle:
    """A Python representation of the state of a Rubik's cube. The cube is represented
    by its corner, edge and center states. Moves can be executed on the puzzle and it
    can be converted to and from different formats (facelet colors, string, etc.).
    """

    def __init__(
        self,
        n: int,
        name: str,
        corners: tuple[CornerState, ...],
        edges: tuple[EdgeState, ...],
        center_colors: tuple[int, ...],
    ):
        """Create a new puzzle with the given corner, edge and center states."""
        self.n = n
        self.name = name
        self.corners = corners
        self.edges = edges
        self.center_colors = center_colors

    def execute_move(self, move: Move) -> "Puzzle":
        """Apply a move to the puzzle and return the new state."""
        return Puzzle(
            self.n,
            self.name,
            tuple([c.execute_move(move) for c in self.corners]),
            tuple([e.execute_move(move) for e in self.edges]),
            self.center_colors,
        )

    def is_finished(self):
        """Check whether the puzzle is in a finished state. This is the case if all
        corners and edges are in their finished state (the center colors cannot change).
        """
        return self.corners == CornerState.all_finished(
            self.n
        ) and self.edges == EdgeState.all_finished(self.n)

    def is_valid(self) -> bool:
        """Check whether the puzzle is in a valid state (not decisive)."""
        # Check that corner x_hi, y_hi and z_hi are unique.
        corner_xyzs: set[tuple[bool, bool, bool]] = set()
        for corner in self.corners:
            corner_xyz = (corner.x_hi, corner.y_hi, corner.z_hi)
            if corner_xyz in corner_xyzs:
                return False
            corner_xyzs.add(corner_xyz)

        # Check that edge a, x_hi and y_hi are unique.
        edge_axys: set[tuple[int, bool, bool]] = set()
        for edge in self.edges:
            edge_axy = (edge.a, edge.x_hi, edge.y_hi)
            if edge_axy in edge_axys:
                return False
            edge_axys.add(edge_axy)

        # Check that the sum of changed and unchanged directions are both
        # even and sum up to 8.
        changed, unchanged = 0, 0
        for corner in self.corners:
            if corner.cw:
                changed += 1
            else:
                unchanged += 1
        if changed % 2 != 0 or unchanged % 2 != 0 or changed + unchanged != 8:
            return False

        return True

    @staticmethod
    def finished(n: int, name: str, center_colors: tuple[int, ...]):
        """Create a finished puzzle given its center colors."""
        return Puzzle(
            n,
            name,
            CornerState.all_finished(n),
            EdgeState.all_finished(n),
            center_colors,
        )

    def to_facelet_colors(self):
        """Get the facelet color representation of the puzzle."""
        return [
            [
                [self.facelet_color(f, y, x) for x in range(self.n)]
                for y in range(self.n)
            ]
            for f in range(6)
        ]

    def __str__(self):
        """Convert the puzzle state into a string by flattening its facelet color
        representation and concatenating the colors.
        """
        facelet_colors = self.to_facelet_colors()
        flattened = [
            facelet_colors[f][y][x]
            for f in range(6)
            for y in reversed(range(self.n))
            for x in range(self.n)
        ]
        return "".join([str(c) for c in flattened])

    @staticmethod
    def from_str(s: str, name: str):
        """Parse a puzzle from the string representation."""
        n = int((len(s) / 6) ** 0.5)
        if len(s) != 6 * n * n:
            raise Exception("invalid puzzle string length")
        if n not in (2, 3):
            raise Exception(f"n = {n} not supported")

        # Extract facelet colors from string
        facelet_colors = [
            [
                [int(s[f * n * n + y * n + x]) for x in range(n)]
                for y in reversed(range(n))
            ]
            for f in range(6)
        ]

        return Puzzle.from_facelet_colors(n, name, facelet_colors)

    def to_file(self):
        """Write the string representation of the puzzle to a file."""
        os.makedirs(PUZZLE_DIR, exist_ok=True)
        with open(os.path.join(PUZZLE_DIR, self.name), "w") as file:
            file.write(str(self))

    @staticmethod
    def from_file(name: str):
        """Parse a puzzle from the string representation in a file."""
        with open(os.path.join(PUZZLE_DIR, name)) as file:
            return Puzzle.from_str(file.read(), name)

    def facelet_color(self, ff: int, fy: int, fx: int) -> int:
        """Get the color of a facelet in the puzzle given its coordinates."""
        corner_fin = CornerState.all_finished(self.n)
        edge_fin = EdgeState.all_finished(self.n)
        x, y, z = facelet_cubie(self.n, ff, fy, fx)

        match cubie_type(self.n, x, y, z):
            case 0:
                for i, corner in enumerate(corner_fin):
                    origin_corner = self.corners[i]
                    if origin_corner.coords() == (x, y, z):
                        colors = cubie_colors(
                            self.n,
                            *corner.coords(),
                            self.center_colors,
                        )
                        first_color = colors[origin_corner.r]

                        # If the cubie's clockwise direction has changed, reverse the
                        # color list to adhere to ordering.
                        if origin_corner.cw:
                            colors.reverse()

                        # Rotate until the first color is in the first slot.
                        while colors[0] != first_color:
                            rotate_list(colors)

                        # Get the facelet index of the facelet in question and return.
                        fi = cubie_facelets(self.n, x, y, z).index((ff, fy, fx))
                        return colors[fi]
            case 1:
                return self.center_colors[ff]
            case 2:
                for i, edge in enumerate(edge_fin):
                    origin_edge = self.edges[i]
                    if origin_edge.coords() == (x, y, z):
                        colors = cubie_colors(
                            self.n, *edge.coords(), self.center_colors
                        )
                        fi = cubie_facelets(self.n, x, y, z).index((ff, fy, fx))
                        if not origin_edge.r:
                            return colors[fi]
                        else:
                            return colors[1 - fi]

        raise Exception(f"invalid facelet: ({ff}, {fy}, {fx})")

    @staticmethod
    def from_facelet_colors(
        n: int,
        name: str,
        facelet_colors: list[list[list[int]]],
    ):
        """Extract the states from a facelet color representation. This is done by
        matching the colors of the facelets to the colors of the cubies in the
        finished state.
        """
        if n == 3:
            center_colors = extract_center_colors(n, facelet_colors)
        else:
            # When n = 2, there are no center cubies, so we assume the default center
            # colors for the finished state. NOTE: that this is not a perfect solution,
            # since the puzzle can be in a solved state with different center colors.
            # However, since this software mainly focuses on 3x3x3 cubes, this is not
            # a big issue.
            center_colors = DEFAULT_CENTER_COLORS

        corner_fin = CornerState.all_finished(n)
        edge_fin = EdgeState.all_finished(n)

        # Extract the colors of the facelets of each cubie from the facelet color
        # representation. This is done by the global face ordering.
        corner_colors = [[] for _ in corner_fin]
        edge_colors = [[] for _ in edge_fin]
        for ff in FACE_ORDERING:
            for fy in range(n):
                for fx in range(n):
                    x, y, z = facelet_cubie(n, ff, fy, fx)
                    match cubie_type(n, x, y, z):
                        case 0:
                            corner1 = CornerState.from_coords(n, x, y, z)
                            corner_colors[corner_fin.index(corner1)].append(
                                facelet_colors[ff][fy][fx]
                            )
                        case 2:
                            edge = EdgeState.from_coords(n, x, y, z)
                            edge_colors[edge_fin.index(edge)].append(
                                facelet_colors[ff][fy][fx]
                            )

        # Extract the corner states from the color colors. This is done by matching the
        # (unique) set of colors associated with each cubie with a cubie in
        # the finished state. Second, the rotation and clockwise change are determined.
        corner_states: list[CornerState | None] = [None] * len(corner_fin)
        for i, corner1 in enumerate(corner_fin):
            colors1 = corner_colors[i]
            for j, corner2 in enumerate(corner_fin):
                colors2 = cubie_colors(n, *corner2.coords(), center_colors)
                if set(colors1) == set(colors2):
                    r = colors2.index(colors1[0])
                    cw = corner1.clockwise() != corner2.clockwise()
                    assert corner_states[j] is None
                    corner_states[j] = CornerState(
                        n,
                        corner1.x_hi,
                        corner1.y_hi,
                        corner1.z_hi,
                        r,
                        cw,
                    )
                    break
            else:
                raise Exception(f"corner '{corner1}' could not be mapped")

        # Extract the edge states from the color colors. This is done by matching the
        # (unique) set of colors associated with each cubie with a cubie in
        # the finished state. Second, the rotation is determined.
        edge_states: list[EdgeState | None] = [None] * len(edge_fin)
        for i, edge in enumerate(edge_fin):
            colors1 = edge_colors[i]
            for j, edge2 in enumerate(edge_fin):
                colors2 = cubie_colors(n, *edge2.coords(), center_colors)
                if set(colors1) == set(colors2):
                    r = colors2.index(colors1[0]) == 1
                    assert edge_states[j] is None
                    edge_states[j] = EdgeState(n, edge.a, edge.x_hi, edge.y_hi, r)
                    break
            else:
                raise Exception(f"edge '{edge}' could not be mapped")

        # Cast the possibly None list to definite corners and edges.
        corners = tuple([s for s in corner_states if s is not None])
        edges = tuple([s for s in edge_states if s is not None])
        assert len(corners) == len(corner_states) and len(edges) == len(edge_states)

        return Puzzle(n, name, corners, edges, center_colors)

    def print(self):
        """Print the puzzle in a human-readable image using matplotlib."""
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

    def __eq__(self, other: "Puzzle"):
        """Check whether two puzzles are equal."""
        return (
            self.n == other.n
            and self.corners == other.corners
            and self.edges == other.edges
            and self.center_colors == other.center_colors
        )

    def __hash__(self):
        """Get a hash of the puzzle."""
        return hash((self.n, self.corners, self.edges, self.center_colors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    Puzzle.from_file(args.path).print()
