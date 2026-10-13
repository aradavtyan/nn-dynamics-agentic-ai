import torch
from torch import nn

from task5.data import DT


class Acceleration(nn.Module):
    def __init__(self, widths=(32, 32)):
        super().__init__()
        self.register_buffer("scales", torch.tensor([0.1, 45.0, 0.1]))
        layers = []
        previous = 3
        for width in widths:
            layers.extend([nn.Linear(previous, width), nn.Tanh()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, u):
        inputs = torch.cat([state, u.unsqueeze(-1)], dim=-1) / self.scales
        return self.net(inputs).squeeze(-1) * 20000


def step(model, state, u0, u1, dt=DT):
    def rhs(s, u):
        return torch.stack([s[..., 1], model(s, u)], dim=-1)

    mid = (u0 + u1) / 2
    k1 = rhs(state, u0)
    k2 = rhs(state + dt * k1 / 2, mid)
    k3 = rhs(state + dt * k2 / 2, mid)
    k4 = rhs(state + dt * k3, u1)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rollout(model, u, initial):
    state = initial
    outputs = [state[..., 0]]
    for i in range(len(u) - 1):
        state = step(model, state, u[i], u[i + 1])
        outputs.append(state[..., 0])
    return torch.stack(outputs)
