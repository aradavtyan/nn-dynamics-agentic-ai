import json
import time
from pathlib import Path

import torch
from scipy.signal import savgol_filter

from task5.data import DT, SPLIT, load
from task5.evaluate import evaluate
from task5.model import Acceleration, rollout


def train(config, output_dir="results"):
    if not 2 <= config["rollout_length"] < SPLIT - 10:
        raise ValueError("Rollout length must be between 2 and 3061 samples")
    if (
        config["training_steps"] < 1
        or config["pretrain_steps"] < 0
        or config["batch_size"] < 1
    ):
        raise ValueError("Invalid training step count or batch size")
    if not 0 < config["learning_rate"] < float("inf"):
        raise ValueError("Learning rate must be finite and positive")
    if Path(config["name"]).name != config["name"]:
        raise ValueError("Experiment name must be a filename, not a path")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    u, y = load()
    u = torch.tensor(u[:SPLIT], dtype=torch.float32)
    y = torch.tensor(y[:SPLIT], dtype=torch.float32)
    
    velocity = torch.tensor(
        savgol_filter(y.numpy(), 9, 5, deriv=1, delta=DT), dtype=torch.float32
    )
    acceleration = torch.tensor(
        savgol_filter(y.numpy(), 9, 5, deriv=2, delta=DT), dtype=torch.float32
    )
    model = Acceleration(config["widths"])
    optimizer = config.get("optimizer", "Adam")
    if optimizer not in ["Adam", "AdamW"]:
        raise ValueError("Unsupported optimizer")
    opt = getattr(torch.optim, optimizer)(
        model.parameters(), lr=config["learning_rate"]
    )
    states = torch.stack([y, velocity], dim=1)
    for _ in range(config["pretrain_steps"]):
        pred = model(states[5:-5], u[5:-5])
        loss = ((pred - acceleration[5:-5]) / 20000).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite warm-up loss")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float("inf"), error_if_nonfinite=True
        )
        opt.step()
    horizon = config["rollout_length"]
    for iteration in range(config["training_steps"]):
        idx = torch.randint(5, SPLIT - horizon - 5, (config["batch_size"],))
        times = idx[None, :] + torch.arange(horizon)[:, None]
        pred = rollout(model, u[times], states[idx])
        loss = ((pred - y[times]) / 0.1).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite rollout loss")
        opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 10.0, error_if_nonfinite=True
        )
        opt.step()
        if iteration % 100 == 0:
            print(
                config["name"], iteration, float(loss.detach()), float(norm), flush=True
            )
    path = output_dir / f"{config['name']}.pt"
    import hashlib

    provenance = {
        "dataset_sha256": hashlib.sha256(
            Path("data/SNLS80mV.mat").read_bytes()
        ).hexdigest(),
        "torch_version": str(torch.__version__),
        "dtype": "float32",
        "device": "cpu",
        "input_interpolation": "linear",
        "initial_velocity": "forward difference (y[1]-y[0])/dt",
        "centering": "whole 4096-sample excerpt",
        "model_sha256": hashlib.sha256(Path("task5/model.py").read_bytes()).hexdigest(),
    }
    torch.save(
        {"config": config, "state_dict": model.state_dict(), "provenance": provenance},
        path,
    )
    metrics = evaluate(path)
    record = {
        **config,
        "optimizer": optimizer,
        "elapsed_time": time.monotonic() - start,
        "result_status": "completed",
        **metrics,
        "provenance": provenance,
    }
    path.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--name", default="manual")
    parser.add_argument("--optimizer", choices=["Adam", "AdamW"], default="Adam")
    args = parser.parse_args()
    train(
        dict(
            name=args.name,
            widths=args.widths,
            training_steps=args.steps,
            rollout_length=args.horizon,
            learning_rate=args.lr,
            seed=42,
            batch_size=32,
            pretrain_steps=1200,
            optimizer=args.optimizer,
        )
    )
