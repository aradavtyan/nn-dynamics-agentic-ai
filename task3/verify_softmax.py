import numpy as np


def softmax(z):
    z = np.asarray(z, dtype=float)
    if z.ndim == 0 or z.shape[-1] == 0 or not np.isfinite(z).all():
        raise ValueError("Softmax requires a nonempty last axis of finite scores")
    
    with np.errstate(over="ignore", under="ignore"):
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    p = softmax([1, 2, 3])
    assert (p > 0).all() and np.allclose(p.sum(), 1)
    print(p, "sum", p.sum())
