import json

import torch
from torch import nn


def calculate():
    h, c, rows = 224, 3, []
    specs = [
        ("C1", 64, 7, 2, 3),
        ("P1", None, 3, 2, 0),
        ("C2", 192, 5, 1, 2),
        ("P2", None, 3, 2, 0),
        ("C3", 384, 3, 1, 1),
        ("C4", 256, 3, 1, 1),
        ("C5", 256, 3, 1, 1),
        ("P3", None, 3, 2, 0),
    ]
    modules = []
    for name, out, k, s, p in specs:
        weights = k * k * c * out if out else 0
        biases = out or 0
        modules.append(nn.Conv2d(c, out, k, s, p) if out else nn.MaxPool2d(k, s, p))
        if out:
            modules.append(nn.ReLU())
        h = (h + 2 * p - k) // s + 1
        c = out or c
        rows.append(
            dict(
                layer=name,
                shape=[h, h, c],
                weights=weights,
                biases=biases,
                parameters=weights + biases,
            )
        )
    n = h * h * c
    modules.append(nn.Flatten())
    for name, out in [("F6", 512), ("F7", 256), ("F8", 10)]:
        modules.append(nn.Linear(n, out))
        if name != "F8":
            modules.append(nn.ReLU())
        rows.append(
            dict(
                layer=name,
                shape=[out],
                weights=n * out,
                biases=out,
                parameters=n * out + out,
            )
        )
        n = out
    x = torch.zeros(1, 3, 224, 224)
    index = 0
    with torch.no_grad():
        for m in modules:
            x = m(x)
            if isinstance(m, (nn.Flatten, nn.ReLU)):
                continue
            row = rows[index]
            shape = (
                [x.shape[2], x.shape[3], x.shape[1]] if x.ndim == 4 else [x.shape[1]]
            )
            assert shape == row["shape"]
            assert sum(p.numel() for p in m.parameters()) == row["parameters"]
            index += 1
    assert sum(r["parameters"] for r in rows) == sum(
        p.numel() for m in modules for p in m.parameters()
    )
    return rows


if __name__ == "__main__":
    rows = calculate()
    print(json.dumps(rows, indent=2))
    print("Total:", sum(r["parameters"] for r in rows))
