from datetime import datetime
import os
import math
from PIL import Image, ImageDraw
import re


def print_with_stamp(s: str):
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
    UNSET = 9

    def __init__(self, n: int, colors: list[int]):
        self.n = n
        self.colors = colors

    @staticmethod
    def from_str(s: str):
        n_raw = math.sqrt(len(s) / 6)
        if not n_raw.is_integer():
            raise Exception(f"invalid state string size: {s}")
        return State(int(n_raw), [State.UNSET if c == "*" else int(c) for c in s])

    def to_str(self):
        return "".join(
            [
                "*" if self.is_unset(f, y, x) else str(self.get_color(f, y, x))
                for f in range(6)
                for y in range(self.n)
                for x in range(self.n)
            ]
        )

    def to_str_only_corners(self):
        return "".join(
            [
                "*"
                if self.is_unset(f, y, x) or not self.is_corner_cell(x, y)
                else str(self.get_color(f, y, x))
                for f in range(6)
                for y in range(self.n)
                for x in range(self.n)
            ]
        )

    @staticmethod
    def finished(n: int):
        return State(
            n,
            [0] * n**2 + [1] * n**2 + [2] * n**2 + [3] * n**2 + [4] * n**2 + [5] * n**2,
        )

    def cell_idx(self, f: int, y: int, x: int):
        return x + y * self.n + f * self.n * self.n

    def is_unset(self, f: int, y: int, x: int):
        idx = self.cell_idx(f, y, x)
        return self.colors[idx] == State.UNSET

    def get_color(self, f: int, y: int, x: int) -> int:
        idx = self.cell_idx(f, y, x)
        result = self.colors[idx]
        assert result != State.UNSET
        return result

    def is_corner_cell(self, x: int, y: int) -> bool:
        return (
            (y == 0 and x == 0)
            or (y == 0 and x == self.n - 1)
            or (y == self.n - 1 and x == 0)
            or (y == self.n - 1 and x == self.n - 1)
        )

    def execute_move(self, mi: int, ma: int, md: int):
        new_colors = [State.UNSET] * len(self.colors)
        for f in range(6):
            for y in range(self.n):
                for x in range(self.n):
                    idx = self.cell_idx(f, y, x)
                    map_f, map_y, map_x = mapping(self.n, ma, mi, md, f, y, x)
                    map_idx = self.cell_idx(map_f, map_y, map_x)
                    new_colors[idx] = self.colors[map_idx]
        return State(self.n, new_colors)

    def print(self):
        square_size = 48
        image_size = (3 * self.n * square_size, 4 * self.n * square_size)
        im = Image.new(mode="RGB", size=image_size)
        draw = ImageDraw.Draw(im)

        def draw_face(start_x: int, start_y: int, f: int):
            for y in range(self.n):
                for x in range(self.n):
                    draw.rectangle(
                        (
                            start_x + (x * square_size),
                            start_y + (y * square_size),
                            start_x + ((x + 1) * square_size),
                            start_y + ((y + 1) * square_size),
                        ),
                        fill=color_name(self.get_color(f, y, x)),
                        outline="black",
                        width=4,
                    )

        draw_face(1 * self.n * square_size, 1 * self.n * square_size, 0)
        draw_face(2 * self.n * square_size, 1 * self.n * square_size, 1)
        draw_face(
            1 * self.n * square_size,
            3 * self.n * square_size,
            2,
        )
        draw_face(0 * self.n * square_size, 1 * self.n * square_size, 3)
        draw_face(1 * self.n * square_size, 0 * self.n * square_size, 4)
        draw_face(1 * self.n * square_size, 2 * self.n * square_size, 5)
        im.show()


def mapping(
    n: int, ma: int, mi: int, md: int, f: int, y: int, x: int
) -> tuple[int, int, int]:
    """Get the coordinates of the location of a cell in the previous state given
    its new location and the last move based on the physics of a Rubik's cube.
    As for the orientation: the front, right, back and left faces face upwards,
    and the bottom and top faces both face upwards when rotating them towards
    you."""
    if ma == 0:
        if f == 4 and mi == 0:
            if md == 0:
                return (4, n - 1 - x, y)
            elif md == 1:
                return (4, x, n - 1 - y)
            elif md == 2:
                return (4, n - 1 - y, n - 1 - x)
        elif f == 5 and mi == n - 1:
            if md == 0:
                return (5, x, n - 1 - y)
            elif md == 1:
                return (5, n - 1 - x, y)
            elif md == 2:
                return (5, n - 1 - y, n - 1 - x)
        elif f == 0 and mi == y:
            if md == 0:
                return (1, y, x)
            elif md == 1:
                return (3, y, x)
            elif md == 2:
                return (2, y, x)
        elif f == 1 and mi == y:
            if md == 0:
                return (2, y, x)
            elif md == 1:
                return (0, y, x)
            elif md == 2:
                return (3, y, x)
        elif f == 2 and mi == y:
            if md == 0:
                return (3, y, x)
            elif md == 1:
                return (1, y, x)
            elif md == 2:
                return (0, y, x)
        elif f == 3 and mi == y:
            if md == 0:
                return (0, y, x)
            elif md == 1:
                return (2, y, x)
            elif md == 2:
                return (1, y, x)
    elif ma == 1:
        if f == 3 and mi == 0:
            if md == 0:
                return (3, x, n - 1 - y)
            elif md == 1:
                return (3, n - 1 - x, y)
            elif md == 2:
                return (3, n - 1 - y, n - 1 - x)
        elif f == 1 and mi == n - 1:
            if md == 0:
                return (1, n - 1 - x, y)
            elif md == 1:
                return (1, x, n - 1 - y)
            elif md == 2:
                return (1, n - 1 - y, n - 1 - x)
        elif f == 0 and mi == x:
            if md == 0:
                return (5, y, x)
            elif md == 1:
                return (4, y, x)
            elif md == 2:
                return (2, n - 1 - y, n - 1 - x)
        elif f == 5 and mi == x:
            if md == 0:
                return (2, n - 1 - y, n - 1 - x)
            elif md == 1:
                return (0, y, x)
            elif md == 2:
                return (4, y, x)
        elif f == 2 and mi == n - 1 - x:
            if md == 0:
                return (4, n - 1 - y, n - 1 - x)
            elif md == 1:
                return (5, n - 1 - y, n - 1 - x)
            elif md == 2:
                return (0, n - 1 - y, n - 1 - x)
        elif f == 4 and mi == x:
            if md == 0:
                return (0, y, x)
            elif md == 1:
                return (2, n - 1 - y, n - 1 - x)
            elif md == 2:
                return (5, y, x)
    elif ma == 2:
        if f == 0 and mi == 0:
            if md == 0:
                return (0, n - 1 - x, y)
            elif md == 1:
                return (0, x, n - 1 - y)
            elif md == 2:
                return (0, n - 1 - y, n - 1 - x)
        elif f == 2 and mi == n - 1:
            if md == 0:
                return (2, x, n - 1 - y)
            elif md == 1:
                return (2, n - 1 - x, y)
            elif md == 2:
                return (2, n - 1 - y, n - 1 - x)
        elif f == 1 and mi == x:
            if md == 0:
                return (4, n - 1 - x, y)
            elif md == 1:
                return (5, x, n - 1 - y)
            elif md == 2:
                return (3, n - 1 - y, n - 1 - x)
        elif f == 4 and mi == n - 1 - y:
            if md == 0:
                return (3, n - 1 - x, y)
            elif md == 1:
                return (1, x, n - 1 - y)
            elif md == 2:
                return (5, n - 1 - y, n - 1 - x)
        elif f == 3 and mi == n - 1 - x:
            if md == 0:
                return (5, n - 1 - x, y)
            elif md == 1:
                return (4, x, n - 1 - y)
            elif md == 2:
                return (1, n - 1 - y, n - 1 - x)
        elif f == 5 and mi == y:
            if md == 0:
                return (1, n - 1 - x, y)
            elif md == 1:
                return (3, x, n - 1 - y)
            elif md == 2:
                return (4, n - 1 - y, n - 1 - x)
    return (f, y, x)
