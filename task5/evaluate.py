import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from task5.data import DT, SPLIT, load
from task5.model import Acceleration, rollout


def rmse(pred, target):
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if pred.shape != target.shape or pred.size == 0:
        raise ValueError("RMSE requires nonempty arrays of identical shape")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("RMSE requires finite values")
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def predict(model, u, y0, y1):
    with torch.no_grad():
        return rollout(
            model,
            torch.tensor(u, dtype=torch.float32),
            torch.tensor([y0, (y1 - y0) / DT], dtype=torch.float32),
        ).numpy()


def evaluate(checkpoint, plot=True, save_predictions=False):
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    widths = saved["config"]["widths"]
    if (
        not isinstance(widths, (list, tuple))
        or not 1 <= len(widths) <= 4
        or any(type(w) is not int or not 1 <= w <= 512 for w in widths)
    ):
        raise ValueError("Invalid checkpoint architecture")
    model = Acceleration(widths)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    u, y = load()
    pred = predict(model, u, y[0], y[1])
    if not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite full rollout")
    metrics = {
        "train_rmse": rmse(pred[:SPLIT], y[:SPLIT]),
        "test_rmse": rmse(pred[SPLIT:], y[SPLIT:]),
        "zero_test_rmse": rmse(np.zeros(1024), y[SPLIT:]),
    }
    if plot:
        os.environ.setdefault(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "homework1-mpl")
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.arange(SPLIT, 4096) * DT
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, y[SPLIT:], label="Measured", lw=1)
        ax.plot(
            t, pred[SPLIT:], label="Predicted continuous free run", lw=0.8, alpha=0.8
        )
        ax.set(xlabel="Time from excerpt start (s)", ylabel="Centered output (V)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(checkpoint).with_suffix(".png"), dpi=160)
        plt.close(fig)
    if save_predictions:
        np.savez(
            Path(checkpoint).with_suffix(".npz"), prediction=pred, measured=y, input=u
        )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.checkpoint), indent=2))
