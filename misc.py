from datetime import datetime
import os
from PIL import Image, ImageDraw
import re


def print_stamped(s: str):
    """Print with a timestamp."""
    print(f"[{datetime.now().isoformat(' ', 'seconds')}] {s}")


def create_parent_directory(file_path: str):
    """Create the parent directory of a file if it does not exist yet."""
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def natural_sorted(ls: list[str]):
    """Sort a list of strings 'naturally', such that strings with numbers in them
    are sorted increasingly instead of alphabetically."""

    def convert(s: str):
        return int(s) if s.isdigit() else s.lower()

    def alphanum_key(key: str):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(ls, key=alphanum_key)


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


class CoordIdxMap:
    """Class to map the coordinates of a corner, edge or center cubie to a unique
    flat index and the other way around for a specific n."""

    def __init__(self, n: int):
        self.n = n

        self.corners: list[tuple[int, int, int]] = []
        self.edges: list[tuple[int, int, int]] = []
        self.centers: list[tuple[int, int, int]] = []

        # Add the eight corner cubies.
        for x in [0, n - 1]:
            for y in [0, n - 1]:
                for z in [0, n - 1]:
                    self.corners.append((x, y, z))

        # Add the bottom edge cubies.
        for z in range(1, n - 1):
            self.edges.append((0, 0, z))
        for z in range(1, n - 1):
            self.edges.append((n - 1, 0, z))
        for x in range(1, n - 1):
            self.edges.append((x, 0, 0))
        for x in range(1, n - 1):
            self.edges.append((x, 0, n - 1))

        # Add the top edge cubies.
        for z in range(1, n - 1):
            self.edges.append((0, n - 1, z))
        for z in range(1, n - 1):
            self.edges.append((n - 1, n - 1, z))
        for x in range(1, n - 1):
            self.edges.append((x, n - 1, 0))
        for x in range(1, n - 1):
            self.edges.append((x, n - 1, n - 1))

        # Add the edge cubies from the layer in between.
        for y in range(1, n - 1):
            for x in [0, n - 1]:
                for z in [0, n - 1]:
                    self.edges.append((x, y, z))

        # Add the center cubies for each of the six sides.
        for x in range(1, n - 1):
            for y in range(1, n - 1):
                for z in [0, n - 1]:
                    self.centers.append((x, y, z))
        for y in range(1, n - 1):
            for z in range(1, n - 1):
                for x in [0, n - 1]:
                    self.centers.append((x, y, z))
        for x in range(1, n - 1):
            for z in range(1, n - 1):
                for y in [0, n - 1]:
                    self.centers.append((x, y, z))

        assert len(self.corners) == 8
        assert len(self.edges) == 12 * n - 24
        assert len(self.centers) == 6 * (n - 2) ** 2

    def corner_idx(self, x: int, y: int, z: int) -> int:
        return self.corners.index((x, y, z))

    def corner_coord(self, idx: int) -> tuple[int, int, int]:
        return self.corners[idx]

    def edge_idx(self, x: int, y: int, z: int) -> int:
        return self.edges.index((x, y, z))

    def edge_coord(self, idx: int) -> tuple[int, int, int]:
        return self.edges[idx]

    def center_idx(self, x: int, y: int, z: int) -> int:
        return self.centers.index((x, y, z))

    def center_coord(self, idx: int) -> tuple[int, int, int]:
        return self.centers[idx]


class State:
    def __init__(
        self,
        n: int,
        map: CoordIdxMap,
        corners: list[tuple[int, int]],
        edges: list[tuple[int, int]],
        centers: list[int],
    ):
        self.n = n
        self.map = map
        self.corners = corners
        self.edges = edges
        self.centers = centers

    @staticmethod
    def from_str(s: str):
        parts = s.split(" ")
        n = int(parts[0])

        corners: list[tuple[int, int]] = []
        for v in parts[1].split(","):
            loc, rot = v.split("r")
            corners.append((int(loc), int(rot)))

        edges: list[tuple[int, int]] = []
        for v in parts[2].split(","):
            loc, rot = v.split("r")
            edges.append((int(loc), int(rot)))

        centers = [int(v) for v in parts[3].split(",")]

        return State(
            n,
            CoordIdxMap(n),
            corners,
            edges,
            centers,
        )

    def to_str(self):
        return " ".join(
            [
                str(self.n),
                ",".join([f"{v[0]}r{v[1]}" for v in self.corners]),
                ",".join([f"{v[0]}r{v[1]}" for v in self.edges]),
                ",".join([str(v) for v in self.centers]),
            ]
        )

    @staticmethod
    def finished(n: int):
        map = CoordIdxMap(n)
        return State(
            n,
            map,
            [(map.corner_idx(x, y, z), 0) for x, y, z in map.corners],
            [(map.edge_idx(x, y, z), 0) for x, y, z in map.edges],
            [map.center_idx(x, y, z) for x, y, z in map.centers],
        )

    def color(self, f: int, y: int, x: int) -> int:
        return -1  # TODO

    def execute_move(self, ma: int, mi: int, md: int):
        new_corners = [(int(-1), int(-1))] * len(self.corners)
        for i, (ci, cr) in enumerate(self.corners):
            x, y, z = self.map.corner_coord(i)
            new_x, new_y, new_z = corner_move_location_mapping(
                self.n, x, y, z, ma, mi, md
            )
            new_i = self.map.corner_idx(new_x, new_y, new_z)
            new_r = move_rotation_mapping(cr, ma, md)
            new_corners[new_i] = (ci, new_r)

        new_edges = [(int(-1), int(-1))] * len(self.edges)
        for i, (ei, er) in enumerate(self.edges):
            x, y, z = self.map.edge_coord(i)
            new_x, new_y, new_z = edge_move_location_mapping(
                self.n, x, y, z, ma, mi, md
            )
            new_i = self.map.edge_idx(new_x, new_y, new_z)
            new_r = move_rotation_mapping(er, ma, md)
            new_edges[new_i] = (ei, new_r)

        new_centers = [0] * len(self.centers)
        for i, ci in enumerate(self.centers):
            x, y, z = self.map.center_coord(i)
            new_x, new_y, new_z = center_move_location_mapping(
                self.n, x, y, z, ma, mi, md
            )
            new_i = self.map.center_idx(new_x, new_y, new_z)
            new_centers[new_i] = ci

        return State(self.n, self.map, new_corners, new_edges, new_centers)

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


def corner_move_location_mapping(
    n: int, x: int, y: int, z: int, ma: int, mi: int, md: int
) -> tuple[int, int, int]:
    if ma == 0:
        if mi == 0 and y == 0:
            if md == 0:
                if x == 0:
                    if z == 0:
                        return (0, 0, n - 1)
                    elif z == n - 1:
                        return (n - 1, 0, n - 1)
                elif x == n - 1:
                    if z == 0:
                        return (0, 0, 0)
                    elif z == n - 1:
                        return (n - 1, 0, 0)
            elif md == 1:
                if x == 0:
                    if z == 0:
                        return (0, 0, n - 1)
                    elif z == n - 1:
                        return (n - 1, 0, n - 1)
                elif x == n - 1:
                    if z == 0:
                        return (0, 0, 0)
                    elif z == n - 1:
                        return (n - 1, 0, 0)
            elif md == 2:
                if x == 0:
                    if z == 0:
                        return (n - 1, 0, n - 1)
                    elif z == n - 1:
                        return (n - 1, 0, 0)
                elif x == n - 1:
                    if z == 0:
                        return (0, 0, n - 1)
                    elif z == n - 1:
                        return (0, 0, 0)
        elif mi == n - 1 and y == n - 1:
            if md == 0:
                if x == 0:
                    if z == 0:
                        return (0, n - 1, n - 1)
                    elif z == n - 1:
                        return (n - 1, n - 1, n - 1)
                elif x == n - 1:
                    if z == 0:
                        return (0, n - 1, 0)
                    elif z == n - 1:
                        return (n - 1, n - 1, 0)
            elif md == 1:
                if x == 0:
                    if z == 0:
                        return (0, n - 1, n - 1)
                    elif z == 0:
                        return (n - 1, n - 1, n - 1)
                elif x == n - 1:
                    if z == 0:
                        return (0, n - 1, 0)
                    elif z == n - 1:
                        return (n - 1, n - 1, 0)
            elif md == 2:
                if x == 0:
                    if z == 0:
                        return (n - 1, n - 1, n - 1)
                    elif z == n - 1:
                        return (n - 1, n - 1, 0)
                elif x == n - 1:
                    if z == 0:
                        return (0, n - 1, n - 1)
                    elif z == n - 1:
                        return (0, n - 1, 0)
    elif ma == 1:
        if mi == 0 and x == 0:
            pass  # TODO
        elif mi == n - 1 and x == n - 1:
            pass  # TODO
    elif ma == 2:
        if mi == 0 and z == 0:
            pass  # TODO
        elif mi == n - 1 and z == n - 1:
            pass  # TODO
    return (x, y, z)


def edge_move_location_mapping(
    n: int, x: int, y: int, z: int, ma: int, mi: int, md: int
) -> tuple[int, int, int]:
    if ma == 0:
        if mi == y:
            if mi == 0:
                pass  # TODO
            elif mi == n - 1:
                pass  # TODO
            else:
                pass  # TODO
    elif ma == 1:
        if mi == x:
            if mi == 0:
                pass  # TODO
            elif mi == n - 1:
                pass  # TODO
            else:
                pass  # TODO
    elif ma == 2:
        if mi == z:
            if mi == 0:
                pass  # TODO
            elif mi == n - 1:
                pass  # TODO
            else:
                pass  # TODO
    return (x, y, z)


def center_move_location_mapping(
    n: int, x: int, y: int, z: int, ma: int, mi: int, md: int
) -> tuple[int, int, int]:
    if ma == 0:
        if mi == 1 and y == 1:
            if md == 0:
                if x == 0:
                    return (1, 1, 2)
                elif x == 1:
                    if z == 0:
                        return (0, 1, 1)
                    elif z == 2:
                        return (2, 1, 1)
                elif x == 2:
                    return (1, 1, 0)
            elif md == 1:
                if x == 0:
                    return (1, 1, 0)
                elif x == 1:
                    if z == 0:
                        return (2, 1, 1)
                    elif z == 2:
                        return (0, 1, 1)
                elif x == 2:
                    return (1, 1, 2)
            elif md == 2:
                if x == 0:
                    return (2, 1, 1)
                elif x == 1:
                    if z == 0:
                        return (1, 1, 2)
                    else:
                        return (1, 1, 0)
                elif x == 2:
                    return (0, 1, 1)
    elif ma == 1:
        if mi == 1 and x == 1:
            pass  # TODO
    elif ma == 2:
        if mi == 1 and z == 1:
            pass  # TODO
    return (x, y, z)


def move_rotation_mapping(r: int, ma: int, md: int) -> int:
    if ma == 0:
        if md == 0 or md == 1:
            if r == 0:
                return 2
            elif r == 2:
                return 0
    elif ma == 1:
        if md == 0 or md == 1:
            if r == 1:
                return 2
            elif r == 2:
                return 1
    elif ma == 2:
        if md == 0 or md == 1:
            if r == 0:
                return 1
            elif r == 1:
                return 0
    return r
