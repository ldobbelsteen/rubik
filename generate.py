import numpy as np
import random
import sys

# python generate_puzzle.py {n} {randomizations}
if __name__ == "__main__":
    n = int(sys.argv[1])
    randomizations = int(sys.argv[2])

    result = np.array([np.array([np.array([f for _ in range(n)]) for _ in range(n)]) for f in range(6)])

    for _ in range(randomizations):
        i = random.randrange(n)
        a = random.choice([0, 1, 2])
        d = random.choice([0, 1, 2])
        if a == 0:
            if d == 0:
                front_cache = np.copy(result[0][i])
                result[0][i] = result[1][i]
                result[1][i] = result[2][i]
                result[2][i] = result[3][i]
                result[3][i] = front_cache
                if i == 0:
                    result[4] = np.rot90(result[4], k=3)
                if i == n - 1:
                    result[5] = np.rot90(result[5], k=1)
            elif d == 1:
                front_cache = np.copy(result[0][i])
                result[0][i] = result[3][i]
                result[3][i] = result[2][i]
                result[2][i] = result[1][i]
                result[1][i] = front_cache
                if i == 0:
                    result[4] = np.rot90(result[4], k=1)
                if i == n - 1:
                    result[5] = np.rot90(result[5], k=3)
            elif d == 2:
                front_cache = np.copy(result[0][i])
                result[0][i] = result[2][i]
                result[2][i] = front_cache
                right_cache = np.copy(result[1][i])
                result[1][i] = result[3][i]
                result[3][i] = right_cache
                if i == 0:
                    result[4] = np.rot90(result[4], k=2)
                if i == n - 1:
                    result[5] = np.rot90(result[5], k=2)
        elif a == 1:
            if d == 0:
                front_cache = np.copy(result[0][:, i])
                result[0][:, i] = result[5][:, i]
                result[5][:, i] = np.flip(result[2][:, n - 1 - i])
                result[2][:, n - 1 - i] = np.flip(result[4][:, i])
                result[4][:, i] = front_cache
                if i == 0:
                    result[3] = np.rot90(result[3], k=1)
                if i == n - 1:
                    result[1] = np.rot90(result[1], k=3)
            elif d == 1:
                front_cache = np.copy(result[0][:, i])
                result[0][:, i] = result[4][:, i]
                result[4][:, i] = np.flip(result[2][:, n - 1 - i])
                result[2][:, n - 1 - i] = np.flip(result[5][:, i])
                result[5][:, i] = front_cache
                if i == 0:
                    result[3] = np.rot90(result[3], k=3)
                if i == n - 1:
                    result[1] = np.rot90(result[1], k=1)
            elif d == 2:
                front_cache = np.copy(result[0][:, i])
                result[0][:, i] = np.flip(result[2][:, n - 1 - i])
                result[2][:, n - 1 - i] = np.flip(front_cache)
                top_cache = np.copy(result[4][:, i])
                result[4][:, i] = result[5][:, i]
                result[5][:, i] = top_cache
                if i == 0:
                    result[3] = np.rot90(result[3], k=2)
                if i == n - 1:
                    result[1] = np.rot90(result[1], k=2)
        elif a == 2:
            if d == 0:
                right_cache = np.copy(result[1][:, i])
                result[1][:, i] = result[4][n - 1 - i, :]
                result[4][n - 1 - i, :] = np.flip(result[3][:, n - 1 - i])
                result[3][:, n - 1 - i] = result[5][i, :]
                result[5][i, :] = np.flip(right_cache)
                if i == 0:
                    result[0] = np.rot90(result[0], k=3)
                if i == n - 1:
                    result[2] = np.rot90(result[2], k=1)
            elif d == 1:
                right_cache = np.copy(result[1][:, i])
                result[1][:, i] = np.flip(result[5][i, :])
                result[5][i, :] = result[3][:, n - 1 - i]
                result[3][:, n - 1 - i] = np.flip(result[4][n - 1 - i, :])
                result[4][n - 1 - i, :] = right_cache
                if i == 0:
                    result[0] = np.rot90(result[0], k=1)
                if i == n - 1:
                    result[2] = np.rot90(result[2], k=3)
            elif d == 2:
                right_cache = np.copy(result[1][:, i])
                result[1][:, i] = np.flip(result[3][:, n - 1 - i])
                result[3][:, n - 1 - i] = np.flip(right_cache)
                top_cache = np.copy(result[4][n - 1 - i, :])
                result[4][n - 1 - i, :] = np.flip(result[5][i, :])
                result[5][i, :] = np.flip(top_cache)
                if i == 0:
                    result[0] = np.rot90(result[0], k=2)
                if i == n - 1:
                    result[2] = np.rot90(result[2], k=2)

    file = open(f"puzzles/dim{n}/random{randomizations}.txt", "w")
    file.write(str(result.tolist()))
    file.close()
