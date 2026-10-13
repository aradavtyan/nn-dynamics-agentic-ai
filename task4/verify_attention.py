import numpy as np

from task3.verify_softmax import softmax


def attention(x, wq, wk, wv, scale=True):
    q, k, v = x @ wq, x @ wk, x @ wv
    return softmax(q @ k.T / (np.sqrt(k.shape[1]) if scale else 1)) @ v


def verify():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 3))
    w = [rng.normal(size=(3, 4)) for _ in range(3)]
    p = np.eye(5)[[2, 4, 0, 1, 3]]
    assert np.allclose(attention(p @ x, *w), p @ attention(x, *w))
    x = np.eye(2)
    p = x[::-1]
    e = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert not np.allclose(p @ x + e, p @ (x + e))
    return attention(x, x, x, x, False), attention(p @ x, x, x, x, False)


if __name__ == "__main__":
    print(verify())
