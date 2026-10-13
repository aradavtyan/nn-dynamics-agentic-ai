import numpy as np


def absolute(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0) + np.maximum(-x, 0)


def xor(x):
    x = np.asarray(x)
    return absolute(x[..., 0] - x[..., 1])


if __name__ == "__main__":
    for x in [-2, -0.5, 0, 0.5, 3]:
        print(
            x,
            "preactivations",
            [x, -x],
            "activations",
            [max(x, 0), max(-x, 0)],
            "output",
            absolute(x),
        )
    for x in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        d = x[0] - x[1]
        print(x, "hidden", [max(d, 0), max(-d, 0)], "output", xor(x))
