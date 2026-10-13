import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from task5.data import load
from task5.evaluate import rmse
from task5.experiments import cached_record
from task5.model import Acceleration, step
from task5.train import train

@pytest.mark.parametrize(
    "pred,target", [([1, 2], [[1], [2]]), ([], []), ([np.nan], [0]), ([np.inf], [0])]
)
def test_rmse_rejects_invalid_arrays(pred, target):
    with pytest.raises(ValueError):
        rmse(pred, target)

class Harmonic(torch.nn.Module):
    def forward(self, state, u):
        return -4 * state[..., 0]

def test_rk4_harmonic_exact_solution_and_order():
    errors = []
    exact = np.array([np.cos(2), -2 * np.sin(2)])
    for dt in [0.02, 0.01]:
        state = torch.tensor([1.0, 0.0], dtype=torch.float64)
        for _ in range(round(1 / dt)):
            state = step(Harmonic(), state, torch.tensor(0.0), torch.tensor(0.0), dt=dt)
        errors.append(np.linalg.norm(state.numpy() - exact))
    assert errors[1] < 1e-8 and 14 < errors[0] / errors[1] < 18

def test_actual_optimizer_changes_network():
    torch.manual_seed(42)
    model = Acceleration([8])
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    state = torch.tensor([[0.1, 2.0], [-0.05, -3.0]])
    prediction = step(model, state, torch.zeros(2), torch.ones(2) * 0.01)
    loss = prediction[:, 0].square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

def test_evaluation_does_not_read_test_targets_for_predictions(monkeypatch, tmp_path):
    evaluation = importlib.import_module("task5.evaluate")
    best = json.loads(Path("results/best.json").read_text())
    checkpoint = tmp_path / "model.pt"
    shutil.copyfile(Path("results") / (best["name"] + ".pt"), checkpoint)
    u, y = load()
    evaluation.evaluate(checkpoint, save_predictions=True)
    original = np.load(checkpoint.with_suffix(".npz"))["prediction"].copy()
    altered = y.copy()
    altered[3072:] = np.random.default_rng(3).normal(10, 5, 1024)
    monkeypatch.setattr(evaluation, "load", lambda: (u, altered))
    changed_metrics = evaluation.evaluate(checkpoint, save_predictions=True)
    changed = np.load(checkpoint.with_suffix(".npz"))["prediction"]
    np.testing.assert_array_equal(changed, original)
    assert changed_metrics["test_rmse"] > 1

def test_cached_record_requires_checkpoint_and_matching_metrics(tmp_path):
    record = json.loads(Path("results/best.json").read_text())
    path = tmp_path / (record["name"] + ".json")
    path.write_text(json.dumps(record))
    config = {
        k: record[k]
        for k in [
            "name",
            "widths",
            "optimizer",
            "learning_rate",
            "training_steps",
            "rollout_length",
            "seed",
            "batch_size",
            "pretrain_steps",
        ]
    }
    assert cached_record(path, config) is None
    shutil.copyfile(Path("results") / (record["name"] + ".pt"), path.with_suffix(".pt"))
    assert cached_record(path, config) is not None
    record["test_rmse"] = 0
    path.write_text(json.dumps(record))
    assert cached_record(path, config) is None
    assert cached_record(path, config | {"learning_rate": 0.1}) is None

@pytest.mark.parametrize("horizon", [0, 1, 3072])
def test_invalid_training_horizon(horizon):
    with pytest.raises(ValueError):
        train({"rollout_length": horizon})
