from datetime import datetime
import numpy as np
import os
import math
from PIL import Image, ImageDraw


def print_with_stamp(s: str):
    print(f"[{datetime.now().isoformat(' ', 'seconds')}] {s}")


def create_parent_directory(file_path: str):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)


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
    def __init__(self, s: str):
        face_size, rem = divmod(len(s), 6)
        if rem != 0:
            raise Exception(f"invalid state string size: {s}")

        self.n = int(math.sqrt(face_size))
        if self.n**2 != face_size:
            raise Exception(f"invalid state string size: {s}")

        flat = np.array([int(c) for c in s])
        self.state = np.array(
            [flat_face.reshape(-1, self.n) for flat_face in flat.reshape(-1, face_size)]
        )

    def to_str(self):
        return "".join([str(c) for c in self.state.flatten()])

    @staticmethod
    def finished(n: int):
        return State(
            "0" * n**2 + "1" * n**2 + "2" * n**2 + "3" * n**2 + "4" * n**2 + "5" * n**2
        )

    def get_color(self, f: int, y: int, x: int):
        return self.state[f, y, x]

    def reverse_move(self, mi: int, ma: int, md: int):
        if md == 0:
            self.execute_move(mi, ma, 1)
        elif md == 1:
            self.execute_move(mi, ma, 0)
        elif md == 2:
            self.execute_move(mi, ma, 2)
        else:
            raise Exception("invalid move direction")

    def execute_move(self, mi: int, ma: int, md: int):
        if ma == 0:
            if md == 0:
                front_cache = np.copy(self.state[0][mi])
                self.state[0][mi] = self.state[1][mi]
                self.state[1][mi] = self.state[2][mi]
                self.state[2][mi] = self.state[3][mi]
                self.state[3][mi] = front_cache
                if mi == 0:
                    self.state[4] = np.rot90(self.state[4], k=3)
                if mi == self.n - 1:
                    self.state[5] = np.rot90(self.state[5], k=1)
            elif md == 1:
                front_cache = np.copy(self.state[0][mi])
                self.state[0][mi] = self.state[3][mi]
                self.state[3][mi] = self.state[2][mi]
                self.state[2][mi] = self.state[1][mi]
                self.state[1][mi] = front_cache
                if mi == 0:
                    self.state[4] = np.rot90(self.state[4], k=1)
                if mi == self.n - 1:
                    self.state[5] = np.rot90(self.state[5], k=3)
            elif md == 2:
                front_cache = np.copy(self.state[0][mi])
                self.state[0][mi] = self.state[2][mi]
                self.state[2][mi] = front_cache
                right_cache = np.copy(self.state[1][mi])
                self.state[1][mi] = self.state[3][mi]
                self.state[3][mi] = right_cache
                if mi == 0:
                    self.state[4] = np.rot90(self.state[4], k=2)
                if mi == self.n - 1:
                    self.state[5] = np.rot90(self.state[5], k=2)
            else:
                raise Exception("invalid move direction")
        elif ma == 1:
            if md == 0:
                front_cache = np.copy(self.state[0][:, mi])
                self.state[0][:, mi] = self.state[5][:, mi]
                self.state[5][:, mi] = np.flip(self.state[2][:, self.n - 1 - mi])
                self.state[2][:, self.n - 1 - mi] = np.flip(self.state[4][:, mi])
                self.state[4][:, mi] = front_cache
                if mi == 0:
                    self.state[3] = np.rot90(self.state[3], k=1)
                if mi == self.n - 1:
                    self.state[1] = np.rot90(self.state[1], k=3)
            elif md == 1:
                front_cache = np.copy(self.state[0][:, mi])
                self.state[0][:, mi] = self.state[4][:, mi]
                self.state[4][:, mi] = np.flip(self.state[2][:, self.n - 1 - mi])
                self.state[2][:, self.n - 1 - mi] = np.flip(self.state[5][:, mi])
                self.state[5][:, mi] = front_cache
                if mi == 0:
                    self.state[3] = np.rot90(self.state[3], k=3)
                if mi == self.n - 1:
                    self.state[1] = np.rot90(self.state[1], k=1)
            elif md == 2:
                front_cache = np.copy(self.state[0][:, mi])
                self.state[0][:, mi] = np.flip(self.state[2][:, self.n - 1 - mi])
                self.state[2][:, self.n - 1 - mi] = np.flip(front_cache)
                top_cache = np.copy(self.state[4][:, mi])
                self.state[4][:, mi] = self.state[5][:, mi]
                self.state[5][:, mi] = top_cache
                if mi == 0:
                    self.state[3] = np.rot90(self.state[3], k=2)
                if mi == self.n - 1:
                    self.state[1] = np.rot90(self.state[1], k=2)
            else:
                raise Exception("invalid move direction")
        elif ma == 2:
            if md == 0:
                right_cache = np.copy(self.state[1][:, mi])
                self.state[1][:, mi] = self.state[4][self.n - 1 - mi, :]
                self.state[4][self.n - 1 - mi, :] = np.flip(
                    self.state[3][:, self.n - 1 - mi]
                )
                self.state[3][:, self.n - 1 - mi] = self.state[5][mi, :]
                self.state[5][mi, :] = np.flip(right_cache)
                if mi == 0:
                    self.state[0] = np.rot90(self.state[0], k=3)
                if mi == self.n - 1:
                    self.state[2] = np.rot90(self.state[2], k=1)
            elif md == 1:
                right_cache = np.copy(self.state[1][:, mi])
                self.state[1][:, mi] = np.flip(self.state[5][mi, :])
                self.state[5][mi, :] = self.state[3][:, self.n - 1 - mi]
                self.state[3][:, self.n - 1 - mi] = np.flip(
                    self.state[4][self.n - 1 - mi, :]
                )
                self.state[4][self.n - 1 - mi, :] = right_cache
                if mi == 0:
                    self.state[0] = np.rot90(self.state[0], k=1)
                if mi == self.n - 1:
                    self.state[2] = np.rot90(self.state[2], k=3)
            elif md == 2:
                right_cache = np.copy(self.state[1][:, mi])
                self.state[1][:, mi] = np.flip(self.state[3][:, self.n - 1 - mi])
                self.state[3][:, self.n - 1 - mi] = np.flip(right_cache)
                top_cache = np.copy(self.state[4][self.n - 1 - mi, :])
                self.state[4][self.n - 1 - mi, :] = np.flip(self.state[5][mi, :])
                self.state[5][mi, :] = np.flip(top_cache)
                if mi == 0:
                    self.state[0] = np.rot90(self.state[0], k=2)
                if mi == self.n - 1:
                    self.state[2] = np.rot90(self.state[2], k=2)
            else:
                raise Exception("invalid move direction")
        else:
            raise Exception("invalid move axis")

    def print(self):
        square_size = 48
        image_size = (3 * self.n * square_size, 4 * self.n * square_size)
        im = Image.new(mode="RGB", size=image_size)
        draw = ImageDraw.Draw(im)

        def draw_face(start_x, start_y, face):
            for y in range(self.n):
                for x in range(self.n):
                    draw.rectangle(
                        (
                            start_x + (x * square_size),
                            start_y + (y * square_size),
                            start_x + ((x + 1) * square_size),
                            start_y + ((y + 1) * square_size),
                        ),
                        fill=color_name(face[y][x]),
                        outline="black",
                        width=4,
                    )

        draw_face(1 * self.n * square_size, 1 * self.n * square_size, self.state[0])
        draw_face(2 * self.n * square_size, 1 * self.n * square_size, self.state[1])
        draw_face(
            1 * self.n * square_size,
            3 * self.n * square_size,
            np.rot90(self.state[2], k=2),
        )
        draw_face(0 * self.n * square_size, 1 * self.n * square_size, self.state[3])
        draw_face(1 * self.n * square_size, 0 * self.n * square_size, self.state[4])
        draw_face(1 * self.n * square_size, 2 * self.n * square_size, self.state[5])
        im.show()
