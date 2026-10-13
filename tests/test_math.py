import numpy as np

from task1.verify_networks import absolute, xor
from task2.parameter_count import calculate
from task3.verify_softmax import softmax
from task4.verify_attention import verify


def test_absolute():
    x = np.array([-2, -0.5, 0, 0.5, 3, -1e8, 1e8])
    np.testing.assert_array_equal(absolute(x), np.abs(x))


def test_xor():
    np.testing.assert_array_equal(xor([[0, 0], [0, 1], [1, 0], [1, 1]]), [0, 1, 1, 0])


def test_parameters():
    rows = calculate()
    assert [r["shape"] for r in rows] == [
        [112, 112, 64],
        [55, 55, 64],
        [55, 55, 192],
        [27, 27, 192],
        [27, 27, 384],
        [27, 27, 256],
        [27, 27, 256],
        [13, 13, 256],
        [512],
        [256],
        [10],
    ]
    assert sum(r["parameters"] for r in rows) == 24741450
    assert [r["parameters"] for r in rows] == [
        9472,
        0,
        307392,
        0,
        663936,
        884992,
        590080,
        0,
        22151680,
        131328,
        2570,
    ]
    assert rows[8]["weights"] == 13 * 13 * 256 * 512


def test_softmax():
    p = softmax([[1, 2, 3], [-100, 0, 100]])
    assert (p > 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_attention():
    a, b = verify()
    e = np.exp(1)
    expected = np.array([[e, 1], [1, e]]) / (e + 1)
    np.testing.assert_allclose(a, expected)
    np.testing.assert_allclose(b, expected[::-1])
