from datetime import datetime
import numpy as np
import os
import math


def print_with_stamp(s: str):
    print(f"[{datetime.now().isoformat(' ', 'seconds')}] {s}")


def create_parent_directory(file_path: str):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def reverse_direction(md: int):
    if md == 0:
        return 1
    elif md == 1:
        return 0
    elif md == 2:
        return 2
    else:
        raise Exception("invalid more direction")


def state_to_str(state: np.ndarray):
    return "".join([str(c) for c in state.flatten()])


def str_to_state(s: str) -> list[list[list[int]]]:
    n = math.sqrt(len(s) / 6)
    if not n.is_integer():
        raise Exception("invalid puzzle size")
    n = int(n)

    flat = np.array([int(c) for c in s])
    faces = flat.reshape(-1, len(s) // 6)
    return [face.reshape(-1, n).tolist() for face in faces]


def execute_move(n: int, state: np.ndarray, mi: int, ma: int, md: int):
    if ma == 0:
        if md == 0:
            front_cache = np.copy(state[0][mi])
            state[0][mi] = state[1][mi]
            state[1][mi] = state[2][mi]
            state[2][mi] = state[3][mi]
            state[3][mi] = front_cache
            if mi == 0:
                state[4] = np.rot90(state[4], k=3)
            if mi == n - 1:
                state[5] = np.rot90(state[5], k=1)
        elif md == 1:
            front_cache = np.copy(state[0][mi])
            state[0][mi] = state[3][mi]
            state[3][mi] = state[2][mi]
            state[2][mi] = state[1][mi]
            state[1][mi] = front_cache
            if mi == 0:
                state[4] = np.rot90(state[4], k=1)
            if mi == n - 1:
                state[5] = np.rot90(state[5], k=3)
        elif md == 2:
            front_cache = np.copy(state[0][mi])
            state[0][mi] = state[2][mi]
            state[2][mi] = front_cache
            right_cache = np.copy(state[1][mi])
            state[1][mi] = state[3][mi]
            state[3][mi] = right_cache
            if mi == 0:
                state[4] = np.rot90(state[4], k=2)
            if mi == n - 1:
                state[5] = np.rot90(state[5], k=2)
        else:
            raise Exception("invalid move direction")
    elif ma == 1:
        if md == 0:
            front_cache = np.copy(state[0][:, mi])
            state[0][:, mi] = state[5][:, mi]
            state[5][:, mi] = np.flip(state[2][:, n - 1 - mi])
            state[2][:, n - 1 - mi] = np.flip(state[4][:, mi])
            state[4][:, mi] = front_cache
            if mi == 0:
                state[3] = np.rot90(state[3], k=1)
            if mi == n - 1:
                state[1] = np.rot90(state[1], k=3)
        elif md == 1:
            front_cache = np.copy(state[0][:, mi])
            state[0][:, mi] = state[4][:, mi]
            state[4][:, mi] = np.flip(state[2][:, n - 1 - mi])
            state[2][:, n - 1 - mi] = np.flip(state[5][:, mi])
            state[5][:, mi] = front_cache
            if mi == 0:
                state[3] = np.rot90(state[3], k=3)
            if mi == n - 1:
                state[1] = np.rot90(state[1], k=1)
        elif md == 2:
            front_cache = np.copy(state[0][:, mi])
            state[0][:, mi] = np.flip(state[2][:, n - 1 - mi])
            state[2][:, n - 1 - mi] = np.flip(front_cache)
            top_cache = np.copy(state[4][:, mi])
            state[4][:, mi] = state[5][:, mi]
            state[5][:, mi] = top_cache
            if mi == 0:
                state[3] = np.rot90(state[3], k=2)
            if mi == n - 1:
                state[1] = np.rot90(state[1], k=2)
        else:
            raise Exception("invalid move direction")
    elif ma == 2:
        if md == 0:
            right_cache = np.copy(state[1][:, mi])
            state[1][:, mi] = state[4][n - 1 - mi, :]
            state[4][n - 1 - mi, :] = np.flip(state[3][:, n - 1 - mi])
            state[3][:, n - 1 - mi] = state[5][mi, :]
            state[5][mi, :] = np.flip(right_cache)
            if mi == 0:
                state[0] = np.rot90(state[0], k=3)
            if mi == n - 1:
                state[2] = np.rot90(state[2], k=1)
        elif md == 1:
            right_cache = np.copy(state[1][:, mi])
            state[1][:, mi] = np.flip(state[5][mi, :])
            state[5][mi, :] = state[3][:, n - 1 - mi]
            state[3][:, n - 1 - mi] = np.flip(state[4][n - 1 - mi, :])
            state[4][n - 1 - mi, :] = right_cache
            if mi == 0:
                state[0] = np.rot90(state[0], k=1)
            if mi == n - 1:
                state[2] = np.rot90(state[2], k=3)
        elif md == 2:
            right_cache = np.copy(state[1][:, mi])
            state[1][:, mi] = np.flip(state[3][:, n - 1 - mi])
            state[3][:, n - 1 - mi] = np.flip(right_cache)
            top_cache = np.copy(state[4][n - 1 - mi, :])
            state[4][n - 1 - mi, :] = np.flip(state[5][mi, :])
            state[5][mi, :] = np.flip(top_cache)
            if mi == 0:
                state[0] = np.rot90(state[0], k=2)
            if mi == n - 1:
                state[2] = np.rot90(state[2], k=2)
        else:
            raise Exception("invalid move direction")
    else:
        raise Exception("invalid move axis")
