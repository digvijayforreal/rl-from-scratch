"""The small multi-game network: one brain for every game."""

import torch
import torch.nn as nn

from envs.base import GRID, NUM_ACTIONS, NUM_CHANNELS

LAYER_NAMES = ["conv1", "conv2", "fc", "policy", "value"]


class SmallMultiGameNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 16, kernel_size=3, padding=1)  # 8x10x10 -> 16x10x10
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # -> 32x5x5
        self.fc = nn.Linear(32 * (GRID // 2) ** 2, 128)                      # 800 -> 128
        self.policy = nn.Linear(128, NUM_ACTIONS)  # one score per action
        self.value = nn.Linear(128, 1)             # how good this state is

        self.reinit(LAYER_NAMES)

    def reinit(self, names):
        """Orthogonal init (standard for PPO). A tiny policy head makes every allowed
        action almost equally likely at the start, so the agent explores.
        Also used by experiments to reset chosen layers."""
        gains = {"conv1": 2 ** 0.5, "conv2": 2 ** 0.5, "fc": 2 ** 0.5, "policy": 0.01, "value": 1.0}
        for name in names:
            layer = getattr(self, name)
            nn.init.orthogonal_(layer.weight, gain=gains[name])
            nn.init.zeros_(layer.bias)

    def forward(self, obs, mask, return_activations=False):
        a1 = torch.relu(self.conv1(obs))
        a2 = torch.relu(self.conv2(a1))
        h = torch.relu(self.fc(a2.flatten(1)))
        # Actions this game doesn't use get a huge negative score, so they're never picked.
        logits = self.policy(h).masked_fill(~mask, -1e9)
        value = self.value(h).squeeze(-1)
        if return_activations:
            return logits, value, {"conv1": a1, "conv2": a2, "fc": h}
        return logits, value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
