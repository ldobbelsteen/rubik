import numpy as np
from PIL import Image, ImageDraw
import solve
import sys


def print_puzzle(puzzle: list[list[list[int]]]):
    n = len(puzzle[0])
    square_size = 48
    image_size = (3 * n * square_size, 4 * n * square_size)
    im = Image.new(mode="RGB", size=image_size)
    draw = ImageDraw.Draw(im)

    def draw_face(start_x, start_y, face):
        for y in range(n):
            for x in range(n):
                draw.rectangle(
                    (
                        start_x + (x * square_size),
                        start_y + (y * square_size),
                        start_x + ((x + 1) * square_size),
                        start_y + ((y + 1) * square_size),
                    ),
                    fill=solve.color_name(face[y][x]),
                    outline="black",
                    width=4,
                )

    draw_face(1 * n * square_size, 1 * n * square_size, puzzle[0])
    draw_face(2 * n * square_size, 1 * n * square_size, puzzle[1])
    draw_face(1 * n * square_size, 3 * n * square_size, np.rot90(puzzle[2], k=2))
    draw_face(0 * n * square_size, 1 * n * square_size, puzzle[3])
    draw_face(1 * n * square_size, 0 * n * square_size, puzzle[4])
    draw_face(1 * n * square_size, 2 * n * square_size, puzzle[5])
    im.show()


# python print_puzzle.py {puzzle.txt}
if __name__ == "__main__":
    file = sys.argv[1]
    puzzle = eval(open(file, "r").read())
    print_puzzle(puzzle)
