"""Additional independent regressions added during the final technical audit."""

import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from task1.verify_networks import absolute
from task3.verify_softmax import softmax
from task4.verify_attention import attention
from task5.data import load
from task5.evaluate import rmse
from task5.experiments import cached_record
from task5.model import Acceleration, step
from task5.train import train
from task6.agents import Decision
from task6.coding_tool import CodingTool, verified_execution
from task6.graph import build


@pytest.mark.parametrize(
    "scores",
    [
        [1000, 1001, 999],
        [-1000, -999],
        [1e308, -1e308],
        [1e308, 1e308],
        [0],
        [-1e308, -1e308],
        [0.0] * 100 + [-744.0],
    ],
)
def test_extreme_softmax(scores):
    with np.errstate(all="raise"):
        p = softmax(scores)
    assert np.isfinite(p).all() and (p >= 0).all()
    np.testing.assert_allclose(p.sum(), 1, atol=1e-15)


@pytest.mark.parametrize("scores", [[], [np.inf, 0], [np.nan, 0], 1.0])
def test_invalid_softmax(scores):
    with pytest.raises(ValueError):
        softmax(scores)


def test_absolute_extremes():
    limit = np.finfo(float).max
    np.testing.assert_array_equal(absolute([-limit, 0, limit]), [limit, 0, limit])
    assert absolute(np.iinfo(np.int64).min) == float(2**63)


@pytest.mark.parametrize("n,dk", [(1, 1), (2, 3), (7, 1), (11, 9)])
def test_random_attention_scores_softmax_and_equivariance(n, dk):
    rng = np.random.default_rng(n + dk)
    x = rng.normal(size=(n, 4))
    wq, wk = (rng.normal(size=(4, dk)) for _ in range(2))
    wv = rng.normal(size=(4, 5))
    permutation = rng.permutation(n)
    p = np.eye(n)[permutation]
    q, k = x @ wq, x @ wk
    score = q @ k.T
    np.testing.assert_allclose(
        score, np.array([[qi @ kj for kj in k] for qi in q]), atol=1e-12
    )
    np.testing.assert_allclose((p @ q) @ (p @ k).T, p @ score @ p.T, atol=1e-12)
    np.testing.assert_allclose(
        softmax(p @ score @ p.T), p @ softmax(score) @ p.T, atol=1e-12
    )
    np.testing.assert_allclose(
        attention(p @ x, wq, wk, wv), p @ attention(x, wq, wk, wv), atol=1e-12
    )


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


def test_decision_rejects_blank_or_extra_fields():
    with pytest.raises(ValidationError):
        Decision(next_role="finish", instruction="   ", reason="x")
    with pytest.raises(ValidationError):
        Decision(next_role="finish", instruction="x", reason="x", code="print(1)")


def test_false_verified_flag_cannot_finish(tmp_path):
    class Agents:
        def call(self, role, state):
            return {
                "next_role": "finish",
                "instruction": "Finish",
                "reason": "Test forged state",
            }

    result = build(Agents(), CodingTool(tmp_path)).invoke(
        {"dialogue": [{"role": "user", "text": "Test"}], "verified": True}
    )
    assert result["status"].startswith("unsuccessful")


@pytest.mark.parametrize(
    "failure",
    [ValueError("invalid model output"), RuntimeError("model request failed")],
)
def test_model_errors_never_produce_success(failure, tmp_path):
    class Agents:
        def call(self, role, state):
            raise failure

    with pytest.raises(type(failure), match=str(failure)):
        build(Agents(), CodingTool(tmp_path)).invoke({"dialogue": []})


def test_verified_metric_requires_realistic_process_evidence():
    metrics = {"train_rmse": 0.1, "test_rmse": 0.2, "zero_test_rmse": 0.06}
    result = {
        "verified": True,
        "exit_code": 0,
        "timed_out": False,
        "test_rmse": 0.2,
        "evaluation": {"exit_code": 0, "stdout": json.dumps(metrics)},
    }
    assert verified_execution(result)
    for change in [
        {"exit_code": 1},
        {"test_rmse": 0.3},
        {"timed_out": True},
        {"evaluation": {"exit_code": 0, "stdout": '{"test_rmse":NaN}'}},
    ]:
        assert not verified_execution(result | change)


def test_invalid_checkpoint_execution_is_not_success(tmp_path):
    result = CodingTool(tmp_path).execute(
        "open('candidate.pt','w').write('not a checkpoint')"
    )
    assert result["exit_code"] == 0
    assert result["evaluation"]["exit_code"] != 0
    assert result["evaluation"]["stderr"] and not result["verified"]


def test_executor_does_not_inherit_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-dummy-value")
    result = CodingTool(tmp_path).execute(
        "import os\nprint(bool(os.getenv('OPENAI_API_KEY')))"
    )
    assert result["exit_code"] == 0 and result["stdout"].strip() == "False"


@pytest.mark.parametrize(
    "source",
    [
        "import socket\nsocket.socket()",
        "import subprocess\nsubprocess.run(['true'])",
        "open('/etc/passwd').read()",
    ],
)
def test_executor_denies_network_subprocesses_and_unrelated_reads(source, tmp_path):
    result = CodingTool(tmp_path).execute(source)
    assert result["exit_code"] != 0 and "PermissionError" in result["stderr"]


def test_exact_agent_nodes_and_actual_successful_integration(tmp_path):
    class Agents:
        def __init__(self):
            self.roles = iter(["scientist", "executor", "finish"])

        def call(self, role, state):
            if role == "orchestrator":
                return {
                    "next_role": next(self.roles),
                    "instruction": "Infrastructure test",
                    "reason": "Mock routing only",
                }
            if role == "scientist":
                return "Mock plan: neural acceleration; continuous free-run evaluation."
            return {
                "source": "import torch\nfrom task5.model import Acceleration\nm=Acceleration([4])\nfor p in m.parameters(): p.data.zero_()\ntorch.save({'config':{'widths':[4]},'state_dict':m.state_dict()},'candidate.pt')",
                "explanation": "Infrastructure only, untrained checkpoint",
            }

    graph = build(Agents(), CodingTool(tmp_path))
    assert set(graph.get_graph().nodes) - {"__start__", "__end__"} == {
        "orchestrator",
        "scientist",
        "general",
        "executor",
    }
    result = graph.invoke(
        {
            "dialogue": [{"role": "user", "text": "Infrastructure test"}],
            "decisions": 0,
            "verified": False,
        }
    )
    assert result["status"] == "success" and result["decisions"] == 3
    assert verified_execution(result["executions"][-1])
