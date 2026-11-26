"""Load a trained model and compare it with a random agent on every game.

    uv run python -m experiments.evaluate results\\some_model.pt
"""

import sys

import numpy as np
import torch

from agents.network import SmallMultiGameNet
from agents.ppo import to_tensor
from experiments.train import GAMES


def load_model(path):
    saved = torch.load(path, weights_only=True)
    net = SmallMultiGameNet()
    net.load_state_dict(saved["model"])
    net.eval()
    return net, saved


def evaluate_policy(net, game, episodes=200, seed=999, mode="greedy", policy="model"):
    """Play `episodes` games side by side (one batched forward pass per step).

    Episode i always uses seed + i, so the model and the random agent face the same games.
    mode: "greedy" (best move) or "sample" (roll the dice like training).
    policy: "model" or "random".
    """
    from experiments.baselines import SUCCESS_EVENTS

    envs = [GAMES[game](seed=seed + i) for i in range(episodes)]
    obs = np.stack([e.reset() for e in envs])
    mask = to_tensor(np.stack([e.action_mask() for e in envs]), torch.bool)
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    success = np.zeros(episodes, dtype=bool)
    active = np.ones(episodes, dtype=bool)
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)

    while active.any():
        idx = np.flatnonzero(active)
        if policy == "random":
            actions = [int(rng.choice(envs[i].valid_actions)) for i in idx]
        else:
            with torch.no_grad():
                logits, _ = net(to_tensor(obs[idx]), mask[idx])
            if mode == "greedy":
                actions = logits.argmax(-1).tolist()
            else:
                probs = torch.softmax(logits, -1)
                actions = torch.multinomial(probs, 1, generator=generator).squeeze(-1).tolist()
        for i, a in zip(idx, actions):
            o, r, terminated, truncated = envs[i].step(int(a))
            returns[i] += r
            lengths[i] += 1
            obs[i] = o
            if terminated or truncated:
                active[i] = False
                success[i] = envs[i].last_event in SUCCESS_EVENTS
    return {"returns": returns, "lengths": lengths, "success": success}


def main(path="results/multigame_small.pt", episodes=200):
    net, saved = load_model(path)
    games = saved.get("config", {}).get("games") or list(GAMES)  # the games it was trained on
    print(f"{'game':<8}{'random':>10}{'trained':>10}")
    for name in games:
        random_score = evaluate_policy(net, name, episodes, policy="random")["returns"].mean()
        trained_score = evaluate_policy(net, name, episodes)["returns"].mean()
        print(f"{name:<8}{random_score:>10.2f}{trained_score:>10.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
