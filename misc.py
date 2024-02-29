from datetime import datetime
import os
from PIL import Image, ImageDraw
import re


def print_stamped(s: str):
    print(f"[{datetime.now().isoformat(' ', 'seconds')}] {s}")


def create_parent_directory(file_path: str):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def natural_sorted(ls: list[str]):
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
        case _:
            return "unknown"


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
        case _:
            return "unknown"


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
    return "unknown"


class State:
    def __init__(
        self,
        n: int,
        corner_locations: list[int],
        corner_rotations: list[int],
        edge_locations: list[int],
        edge_rotations: list[int],
        center_locations: list[int],
    ):
        self.n = n
        self.corner_locations = corner_locations
        self.corner_rotations = corner_rotations
        self.edge_locations = edge_locations
        self.edge_rotations = edge_rotations
        self.center_locations = center_locations

    @staticmethod
    def from_str(s: str):
        parts = s.split(" ")
        corner_locations = [int(v) for v in parts[0].split(",")]
        corner_rotations = [int(v) for v in parts[1].split(",")]
        edge_locations = [int(v) for v in parts[2].split(",")]
        edge_rotations = [int(v) for v in parts[3].split(",")]
        center_locations = [int(v) for v in parts[4].split(",")]
        n = int((len(edge_locations) + 24) / 12)
        return State(
            n,
            corner_locations,
            corner_rotations,
            edge_locations,
            edge_rotations,
            center_locations,
        )

    def to_str(self):
        return " ".join(
            [
                ",".join([str(v) for v in self.corner_locations]),
                ",".join([str(v) for v in self.corner_rotations]),
                ",".join([str(v) for v in self.edge_locations]),
                ",".join([str(v) for v in self.edge_rotations]),
                ",".join([str(v) for v in self.center_locations]),
            ]
        )

    @staticmethod
    def finished(n: int):
        n_corners = 8
        n_edges = 12 * n - 24
        n_centers = 6 * (n - 2) ** 2
        return State(
            n,
            [i for i in range(n_corners)],
            [0 for _ in range(n_corners)],
            [i for i in range(n_edges)],
            [0 for _ in range(n_edges)],
            [i for i in range(n_centers)],
        )

    def is_corner(self, x: int, y: int, z: int):
        return False  # TODO

    def is_edge(self, x: int, y: int, z: int):
        return False  # TODO

    def is_center(self, x: int, y: int, z: int):
        return False  # TODO

    def corner_idx(self, x: int, y: int, z: int):
        assert x == 0 or x == self.n - 1
        assert y == 0 or y == self.n - 1
        assert z == 0 or z == self.n - 1
        x = 1 if x > 0 else 0
        y = 1 if y > 0 else 0
        z = 1 if z > 0 else 0
        return z + 2 * y + 2 * 2 * x

    def edge_idx(self, x: int, y: int, z: int):
        if y == 0:
            if x == 0:
                return (0 * (self.n - 2)) + (z - 1)
            elif x == self.n - 1:
                return (1 * (self.n - 2)) + (z - 1)
            elif z == 0:
                return (2 * (self.n - 2)) + (x - 1)
            elif z == self.n - 1:
                return (3 * (self.n - 2)) + (x - 1)
        elif y == self.n - 1:
            if x == 0:
                return (4 * (self.n - 2)) + (z - 1)
            elif x == self.n - 1:
                return (5 * (self.n - 2)) + (z - 1)
            elif z == 0:
                return (6 * (self.n - 2)) + (x - 1)
            elif z == self.n - 1:
                return (7 * (self.n - 2)) + (x - 1)
        else:
            if x == 0:
                if z == 0:
                    return (8 * (self.n - 2)) + (y - 1)
                elif z == self.n - 1:
                    return (9 * (self.n - 2)) + (y - 1)
            elif x == self.n - 1:
                if z == 0:
                    return (10 * (self.n - 2)) + (y - 1)
                elif z == self.n - 1:
                    return (11 * (self.n - 2)) + (y - 1)
        assert False

    def center_idx(self, x: int, y: int, z: int):
        if y == 0:
            return 5
        elif y == self.n // 2:
            if x == 0:
                return 3
            elif x == self.n // 2:
                if z == 0:
                    return 0
                elif z == self.n - 1:
                    return 2
            elif x == self.n - 1:
                return 1
        elif y == self.n - 1:
            return 4
        assert False

    def color(self, f: int, y: int, x: int) -> int:
        return -1  # TODO

    def execute_move(self, ma: int, mi: int, md: int):
        new_corner_locations = [-1] * len(self.corner_locations)
        new_corner_rotations = [-1] * len(self.corner_rotations)
        new_edge_locations = [-1] * len(self.edge_locations)
        new_edge_rotations = [-1] * len(self.edge_rotations)
        new_center_locations = [-1] * len(self.center_locations)
        for x in range(self.n):
            for y in range(self.n):
                for z in range(self.n):
                    if self.is_corner(x, y, z):
                        idx = self.corner_idx(x, y, z)
                        new_x, new_y, new_z, new_r = corner_move_mapping(
                            x, y, z, self.corner_rotations[idx], ma, mi, md
                        )
                        new_idx = self.corner_idx(new_x, new_y, new_z)
                        new_corner_locations[new_idx] = self.corner_locations[idx]
                        new_corner_rotations[new_idx] = new_r
                    elif self.is_edge(x, y, z):
                        idx = self.edge_idx(x, y, z)
                        new_x, new_y, new_z, new_r = edge_move_mapping(
                            x, y, z, self.edge_rotations[idx], ma, mi, md
                        )
                        new_idx = self.edge_idx(new_x, new_y, new_z)
                        new_edge_locations[new_idx] = self.edge_locations[idx]
                        new_edge_rotations[new_idx] = new_r
                    elif self.is_center(x, y, z):
                        idx = self.center_idx(x, y, z)
                        new_x, new_y, new_z = center_move_mapping(x, y, z, ma, mi, md)
                        new_idx = self.center_idx(new_x, new_y, new_z)
                        new_center_locations[new_idx] = self.center_locations[idx]
        return State(
            self.n,
            new_corner_locations,
            new_corner_rotations,
            new_edge_locations,
            new_edge_rotations,
            new_center_locations,
        )

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


def corner_move_mapping(
    x: int, y: int, z: int, r: int, ma: int, mi: int, md: int
) -> tuple[int, int, int, int]:
    if ma == 0:
        if mi == 0 and y == 0:
            if md == 0:
                if x == 0:
                    if z == 0:
                        return (0, 0, 2, r)
                    elif z == 2:
                        return (2, 0, 2, r)
                elif x == 2:
                    if z == 0:
                        return (0, 0, 0, r)
                    elif z == 2:
                        return (2, 0, 0, r)
            elif md == 1:
                if x == 0:
                    if z == 0:
                        return (0, 0, 2, r)
                    elif z == 2:
                        return (2, 0, 2, r)
                elif x == 2:
                    if z == 0:
                        return (0, 0, 0, r)
                    elif z == 2:
                        return (2, 0, 0, r)
            elif md == 2:
                if x == 0:
                    if z == 0:
                        return (2, 0, 2, r)
                    elif z == 2:
                        return (2, 0, 0, r)
                elif x == 2:
                    if z == 0:
                        return (0, 0, 2, r)
                    elif z == 2:
                        return (0, 0, 0, r)
        if mi == 2 and y == 2:
            if md == 0:
                if x == 0:
                    if z == 0:
                        return (0, 2, 2, r)
                    elif z == 2:
                        return (2, 2, 2, r)
                elif x == 2:
                    if z == 0:
                        return (0, 2, 0, r)
                    elif z == 2:
                        return (2, 2, 0, r)
            elif md == 1:
                if x == 0:
                    if z == 0:
                        return (0, 2, 2, r)
                    elif z == 0:
                        return (2, 2, 2, r)
                elif x == 2:
                    if z == 0:
                        return (0, 2, 0, r)
                    elif z == 2:
                        return (2, 2, 0, r)
            elif md == 2:
                if x == 0:
                    if z == 0:
                        return (2, 2, 2, r)
                    elif z == 2:
                        return (2, 2, 0, r)
                elif x == 2:
                    if z == 0:
                        return (0, 2, 2, r)
                    elif z == 2:
                        return (0, 2, 0, r)
    elif ma == 1:
        if mi == 0 and x == 0:
            pass  # TODO
        if mi == 2 and x == 2:
            pass  # TODO
    elif ma == 2:
        if mi == 0 and z == 0:
            pass  # TODO
        if mi == 2 and z == 2:
            pass  # TODO
    return (x, y, z, r)


def edge_move_mapping(
    x: int, y: int, z: int, r: int, ma: int, mi: int, md: int
) -> tuple[int, int, int, int]:
    if ma == 0:
        if mi == 0 and y == 0:
            pass  # TODO
        elif mi == 1 and y == 1:
            pass  # TODO
        elif mi == 2 and y == 2:
            pass  # TODO
    elif ma == 1:
        if mi == 0 and x == 0:
            pass  # TODO
        elif mi == 1 and x == 1:
            pass  # TODO
        elif mi == 2 and x == 2:
            pass  # TODO
    elif ma == 2:
        if mi == 0 and z == 0:
            pass  # TODO
        elif mi == 1 and z == 1:
            pass  # TODO
        elif mi == 2 and z == 2:
            pass  # TODO
    return (x, y, z, r)


def center_move_mapping(
    x: int, y: int, z: int, ma: int, mi: int, md: int
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
