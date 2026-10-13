import json
import math
import pickle
import time
from pathlib import Path

import torch

from task5.evaluate import evaluate
from task5.train import train


def cached_record(path, config):
    checkpoint = path.with_suffix(".pt")
    if not path.exists() or not checkpoint.exists():
        return None
    try:
        record = json.loads(path.read_text())
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if record.get("result_status") != "completed":
            return None
        for key, value in config.items():
            default = "Adam" if key == "optimizer" else None
            if (
                record.get(key, default) != value
                or saved["config"].get(key, default) != value
            ):
                return None
        measured = evaluate(checkpoint, plot=False)
        if any(
            not math.isclose(record[key], value, rel_tol=1e-9, abs_tol=1e-12)
            for key, value in measured.items()
        ):
            return None
        return record
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
    ):
        return None


def run_experiments(output_dir="results"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    settings = [
        ([16, 16], 0.001, 24, "Adam", 600),
        ([16, 16], 0.0003, 48, "Adam", 600),
        ([32, 32], 0.001, 32, "Adam", 600),
        ([32, 32], 0.0003, 64, "Adam", 600),
        ([16, 16], 0.0005, 48, "AdamW", 900),
        ([32, 32], 0.0005, 48, "AdamW", 900),
    ]
    for widths, lr, horizon, optimizer, steps in settings:
        name = f"w{widths[0]}_lr{lr}_h{horizon}" + (
            f"_{optimizer}" if optimizer != "Adam" else ""
        )
        config = dict(
            name=name,
            widths=widths,
            learning_rate=lr,
            rollout_length=horizon,
            training_steps=steps,
            optimizer=optimizer,
            seed=42,
            batch_size=32,
            pretrain_steps=1200,
        )
        path = output_dir / f"{name}.json"
        record = cached_record(path, config)
        if record is None:
            if path.exists():
                history = output_dir / "history"
                history.mkdir(exist_ok=True)
                (history / f"{name}_{time.time_ns()}.json").write_bytes(
                    path.read_bytes()
                )
            try:
                record = train(config, output_dir=output_dir)
            except Exception as exc:
                record = {**config, "result_status": "failed", "error": repr(exc)}
                path.write_text(json.dumps(record, indent=2))
        records.append(record)
        (output_dir / "experiments.json").write_text(json.dumps(records, indent=2))
    completed = [r for r in records if r["result_status"] == "completed"]
    if completed:
        best = min(completed, key=lambda r: r["test_rmse"])
        (output_dir / "best.json").write_text(json.dumps(best, indent=2))
        if not (output_dir / (best["name"] + ".png")).exists():
            evaluate(output_dir / (best["name"] + ".pt"), plot=True)
    elif (output_dir / "best.json").exists():
        (output_dir / "best.json").unlink()
    return records


if __name__ == "__main__":
    run_experiments()
