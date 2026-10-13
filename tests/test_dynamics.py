import numpy as np
import torch
from scipy.io import loadmat

from task5.data import DT, FS, SPLIT, load
from task5.evaluate import predict, rmse
from task5.model import Acceleration, rollout, step


def test_real_dataset():
    u, y = load()
    raw = loadmat("data/SNLS80mV.mat")
    for signal, key in [(u, "V1"), (y, "V2")]:
        expected = raw[key].ravel()[42650:46746]
        np.testing.assert_allclose(signal, expected - expected.mean())
        assert abs(signal.mean()) < 1e-14
    assert (
        len(u) == len(y) == 4096 and len(y[:SPLIT]) == 3072 and len(y[SPLIT:]) == 1024
    )
    assert DT == 1 / FS == 1 / 610.35
    assert 0.055 < rmse(np.zeros(1024), y[SPLIT:]) < 0.065


class InputAcceleration(torch.nn.Module):
    def forward(self, state, u):
        return u


def test_interpolation_and_alignment():
    
    state = torch.tensor([0.0, 0.0], dtype=torch.float64)
    final = step(
        InputAcceleration(), state, torch.tensor(0.0), torch.tensor(0.1), dt=0.1
    )
    np.testing.assert_allclose(final.numpy(), [0.1**3 / 6, 0.1**2 / 2], rtol=1e-6)
    out = rollout(InputAcceleration(), torch.zeros(10), torch.tensor([2.0, 3.0]))
    np.testing.assert_allclose(out.numpy(), 2 + 3 * np.arange(10) * DT, rtol=1e-6)


def test_gradient_and_shapes():
    model = Acceleration([8])
    u = torch.zeros(12, 4)
    pred = rollout(model, u, torch.zeros(4, 2))
    assert pred.shape == (12, 4) and torch.isfinite(pred).all()
    pred.square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert sum(float(p.grad.abs().sum()) for p in model.parameters()) > 0


def test_no_test_leakage_or_boundary_reset():
    
    u, y = load()
    changed = y.copy()
    changed[SPLIT:] += 100
    model = InputAcceleration()
    p = predict(model, u, y[0], y[1])
    q = predict(model, u, changed[0], changed[1])
    np.testing.assert_array_equal(p, q)
    direct = (
        rollout(
            model,
            torch.tensor(u, dtype=torch.float32),
            torch.tensor([y[0], (y[1] - y[0]) / DT], dtype=torch.float32),
        )
        .detach()
        .numpy()
    )
    np.testing.assert_array_equal(p, direct)
    
    constant = predict(model, np.zeros(4096), 0.0, DT)
    np.testing.assert_allclose(
        constant[[SPLIT - 1, SPLIT, SPLIT + 1]],
        np.array([SPLIT - 1, SPLIT, SPLIT + 1]) * DT,
        rtol=1e-4,
    )


def test_rmse():
    assert rmse([1, 3], [0, 0]) == np.sqrt(5)
