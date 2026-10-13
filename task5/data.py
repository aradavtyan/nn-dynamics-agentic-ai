import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

FS = 610.35
DT = 1 / FS
SPLIT = 3072


def load(path="data/SNLS80mV.mat"):
    mat = loadmat(path)
    channels = []
    for key in ["V1", "V2"]:
        signal = np.asarray(mat[key], dtype=float).squeeze()
        if signal.ndim != 1 or len(signal) != 131072 or not np.isfinite(signal).all():
            raise ValueError(f"Unexpected {key} shape or values: {signal.shape}")
        excerpt = signal[42650:46746].copy()
        channels.append(excerpt - excerpt.mean())
    u, y = channels
    assert u.shape == y.shape == (4096,)
    assert len(y[:SPLIT]) == 3072 and len(y[SPLIT:]) == 1024
    return u, y


def acquire():
    Path("data").mkdir(exist_ok=True)
    path = Path("data/SNLS80mV.mat")
    if not path.exists():
        import shutil

        import nonlinear_benchmarks

        paths = nonlinear_benchmarks.Silverbox(
            data_file_locations=True, dir_placement="data"
        )

        def flatten(x):
            if isinstance(x, (str, Path)):
                return [Path(x)]
            return [p for item in x for p in flatten(item)]

        matches = [p for p in flatten(paths) if p.name == "SNLS80mV.mat"]
        if not matches:
            raise FileNotFoundError(f"SNLS80mV.mat not in downloaded files: {paths}")
        shutil.copyfile(matches[0], path)
    _, y = load(path)
    metadata = {
        "source": "https://www.nonlinearbenchmark.org/benchmarks/silverbox",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mat_variables": {
            k: list(v.shape) for k, v in loadmat(path).items() if not k.startswith("__")
        },
        "zero_test_rmse": float(np.sqrt(np.mean(y[SPLIT:] ** 2))),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/dataset.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    acquire()
