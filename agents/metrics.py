"""Plasticity measurements for a network."""

import torch

from envs.base import NUM_ACTIONS


def plasticity_metrics(net, obs, max_samples=512, tau=0.025):
    """Dormant-unit fractions (Sokar et al., 2023) and effective feature rank (Kumar et al., 2021).

    obs: a batch of observations [N, 8, 10, 10]. A unit is dormant when its average
    |activation| is at most `tau` times the layer's average.
    """
    if obs.shape[0] > max_samples:
        obs = obs[torch.randperm(obs.shape[0])[:max_samples]]
    mask = torch.ones(obs.shape[0], NUM_ACTIONS, dtype=torch.bool)
    with torch.no_grad():
        _, _, acts = net(obs, mask, return_activations=True)
    out = {}
    for name, a in acts.items():
        score = a.abs().mean(dim=(0, 2, 3)) if a.dim() == 4 else a.abs().mean(dim=0)
        normalized = score / (score.mean() + 1e-8)
        out[f"dormant_{name}"] = float((normalized <= tau).float().mean())
    # Effective rank: how many singular values are needed to cover 99% of the total.
    s = torch.linalg.svdvals(acts["fc"])
    if float(s.sum()) > 0:
        out["eff_rank"] = int((torch.cumsum(s, 0) / s.sum() < 0.99).sum()) + 1
    else:
        out["eff_rank"] = 0
    return out
