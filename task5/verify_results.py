

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from scipy.io import loadmat
from torch.nn.functional import linear


def run():
    torch.set_num_threads(1)
    mat_path = Path("data/SNLS80mV.mat")
    raw = loadmat(mat_path)
    digest = hashlib.sha256(mat_path.read_bytes()).hexdigest()
    source = json.loads(Path("results/dataset.json").read_text())
    assert digest == source["sha256"]
    archive_path = Path("data/Silverbox/SilverboxFiles.zip")
    archive_matches = None
    if archive_path.exists():
        with ZipFile(archive_path) as archive:
            entry = next(
                name for name in archive.namelist() if name.endswith("/SNLS80mV.mat")
            )
            assert hashlib.sha256(archive.read(entry)).hexdigest() == digest
            archive_matches = True
    signals = []
    for name in ["V1", "V2"]:
        assert raw[name].shape == (1, 131072)
        excerpt = raw[name].ravel()[42650:46746].astype(float)
        signals.append(excerpt - np.mean(excerpt))
    u, y = signals
    assert u.size == y.size == 4096
    dt = 1 / 610.35
    records = json.loads(Path("results/experiments.json").read_text())
    results = []
    for record in records:
        path = Path("results") / (record["name"] + ".pt")
        if not path.exists():
            continue
        saved = torch.load(path, map_location="cpu", weights_only=True)
        weights = saved["state_dict"]
        layers = sorted(
            int(key.split(".")[1]) for key in weights if key.endswith(".weight")
        )

        def rhs(state, forcing):
            inputs = torch.cat((state, forcing.reshape(1))) / weights["scales"]
            for index in layers:
                inputs = linear(
                    inputs, weights[f"net.{index}.weight"], weights[f"net.{index}.bias"]
                )
                if index != layers[-1]:
                    inputs = torch.tanh(inputs)
            return torch.stack((state[1], inputs.squeeze() * 20000))

        forcing = torch.tensor(u, dtype=torch.float32)
        state = torch.tensor([y[0], (y[1] - y[0]) / dt], dtype=torch.float32)
        output = [float(state[0])]
        with torch.no_grad():
            for index in range(4095):
                left, right = forcing[index], forcing[index + 1]
                middle = (left + right) / 2
                first = rhs(state, left)
                second = rhs(state + dt * first / 2, middle)
                third = rhs(state + dt * second / 2, middle)
                fourth = rhs(state + dt * third, right)
                state = state + dt * (first + 2 * second + 2 * third + fourth) / 6
                output.append(float(state[0]))
        prediction = np.asarray(output)
        assert np.isfinite(prediction).all()
        measured = {
            "train_rmse": float(np.sqrt(np.mean((prediction[:3072] - y[:3072]) ** 2))),
            "test_rmse": float(np.sqrt(np.mean((prediction[3072:] - y[3072:]) ** 2))),
            "zero_test_rmse": float(np.sqrt(np.mean(y[3072:] ** 2))),
        }
        if path.with_suffix(".npz").exists():
            with np.load(path.with_suffix(".npz")) as stored:
                np.testing.assert_array_equal(stored["input"], u)
                np.testing.assert_array_equal(stored["measured"], y)
                np.testing.assert_array_equal(stored["prediction"], prediction)
        for key, value in measured.items():
            assert value == record[key], (record["name"], key, value, record[key])
        results.append(
            {
                "name": record["name"],
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "status": "passed",
                **measured,
            }
        )
    best = json.loads(Path("results/best.json").read_text())
    assert any(r["name"] == best["name"] for r in results), "Selected checkpoint is required"
    repeat_status = "not retained; rerun the documented training command to repeat"
    if Path("results/audit_repeat_best.pt").exists():
        original = torch.load(Path("results") / (best["name"] + ".pt"), weights_only=True)
        repeated = torch.load("results/audit_repeat_best.pt", weights_only=True)
        assert all(torch.equal(value, repeated["state_dict"][key]) for key, value in original["state_dict"].items())
        repeat_status = "all weight tensors are bit-identical"
    result = {
        "status": "passed",
        "dataset_sha256": digest,
        "downloaded_archive_matches_mat_file": archive_matches,
        "integrator": "independent four-stage RK4 loop; no task5 imports",
        "records": results,
        "same_seed_repeat": repeat_status,
        "verified_checkpoint_count": len(results),
        "comparison_records_without_retained_checkpoints": [r["name"] for r in records if not (Path("results") / (r["name"] + ".pt")).exists()],
        "architecture_count": len({tuple(record["widths"]) for record in records}),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
