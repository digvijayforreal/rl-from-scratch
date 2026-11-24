"""Train one small network on all games at the same time (no dashboard)."""

import os
import time

import numpy as np
import torch

from agents.network import SmallMultiGameNet, count_parameters
from agents.ppo import RolloutCollector, compute_gae, ppo_update
from envs.catch import Catch
from envs.reach import Reach

GAMES = {"catch": Catch, "reach": Reach}

CONFIG = {
    "games": ["catch", "reach"],
    "envs_per_game": 4,       # 4 Catch + 4 Reach running together
    "total_steps": 600_000,   # game steps across all copies
    "rollout_steps": 128,     # steps per copy before each update
    "lr": 5e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "epochs": 4,
    "minibatches": 4,
    "clip": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "seed": 0,
    "save_path": "results/multigame_small.pt",
}


def make_envs(cfg):
    names = [g for g in cfg["games"] for _ in range(cfg["envs_per_game"])]
    return [GAMES[name](seed=cfg["seed"] * 1000 + i) for i, name in enumerate(names)]


def main(cfg=CONFIG):
    torch.manual_seed(cfg["seed"])
    envs = make_envs(cfg)

    net = SmallMultiGameNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    print(f"Parameters: {count_parameters(net):,}")

    collector = RolloutCollector(envs, cfg["gamma"])
    num_updates = cfg["total_steps"] // (len(envs) * cfg["rollout_steps"])
    start = time.time()

    for update in range(1, num_updates + 1):
        # Lower the learning rate slowly to 0 by the end.
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * (1 - (update - 1) / num_updates)

        batch, last_value = collector.collect(net, cfg["rollout_steps"])
        advantages, returns = compute_gae(batch["reward"], batch["value"], batch["done"],
                                          last_value, cfg["gamma"], cfg["lam"])
        ppo_update(net, optimizer, batch, advantages, returns, cfg)

        if update % 20 == 0 or update == num_updates:
            parts = []
            for name in cfg["games"]:
                recent = [r for g, r in collector.finished if g == name][-100:]
                parts.append(f"{name} {np.mean(recent):+.2f}")
            print(f"update {update:3d}/{num_updates} | " + " | ".join(parts)
                  + f" | {time.time() - start:.0f}s")

    os.makedirs(os.path.dirname(cfg["save_path"]), exist_ok=True)
    torch.save({"model": net.state_dict(), "config": cfg}, cfg["save_path"])
    print(f"Saved model to {cfg['save_path']}")


if __name__ == "__main__":
    main()
