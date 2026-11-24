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

        # Orthogonal init (standard for PPO). A tiny policy head makes every
        # allowed action almost equally likely at the start, so the agent explores.
        for layer in (self.conv1, self.conv2, self.fc):
            nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

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
