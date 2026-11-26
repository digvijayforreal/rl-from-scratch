"""Test-tab helpers: evaluate any brain on every game, and record one episode for replay."""

import numpy as np
import torch

from agents.network import SmallMultiGameNet
from agents.ppo import to_tensor
from dashboard import storage
from dashboard.trainer import build_frame, clean, rounded
from experiments.baselines import SUCCESS_EVENTS, optimal_score
from experiments.evaluate import evaluate_policy
from experiments.train import GAMES

LIVE = "__live__"
HIST_RANGE = (-1.0, 1.0)
HIST_BINS = 20


def _int(value, lo, hi, name):
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number") from None
    return min(max(value, lo), hi)


def _mode(value):
    if value not in ("greedy", "sample"):
        raise ValueError("mode must be 'greedy' or 'sample'")
    return value


def load_brain(trainer, name):
    net = SmallMultiGameNet()
    if name == LIVE:
        _, weights = trainer.weights
        net.load_state_dict(weights)
        label = "Live brain (current training)"
        trained = list(trainer.games)
    else:
        saved = torch.load(storage.resolve_checkpoint(name), weights_only=True)
        net.load_state_dict(saved["model"])
        label = str(name)
        trained = saved.get("config", {}).get("games") or list(GAMES)
    net.eval()
    return net, label, trained


def _stats(res):
    r = res["returns"]
    counts, _ = np.histogram(np.clip(r, *HIST_RANGE), bins=HIST_BINS, range=HIST_RANGE)
    return {"mean": float(r.mean()), "std": float(r.std()), "min": float(r.min()), "max": float(r.max()),
            "success": float(res["success"].mean()), "length": float(res["lengths"].mean()),
            "hist": counts.tolist(), "episodes": int(r.size)}


def evaluate_brain(trainer, name, episodes=200, mode="greedy", seed=999):
    episodes = _int(episodes, 10, 1000, "episodes")
    seed = _int(seed, 0, 1_000_000, "seed")
    mode = _mode(mode)
    net, label, trained = load_brain(trainer, name)
    games = {}
    for g in GAMES:
        model = evaluate_policy(net, g, episodes, seed, mode)
        rand = evaluate_policy(net, g, episodes, seed, policy="random")  # same games, random moves
        games[g] = {"model": _stats(model), "random": _stats(rand), "optimal": optimal_score(g),
                    "trained": g in trained}
    return clean({"label": label, "episodes": episodes, "mode": mode, "seed": seed,
                  "hist_range": HIST_RANGE, "bins": HIST_BINS, "games": games})


def record_episode(trainer, name, game, seed=0, mode="greedy"):
    if game not in GAMES:
        raise ValueError(f"Unknown game {game!r}")
    seed = _int(seed, 0, 1_000_000, "seed")
    mode = _mode(mode)
    net, label, _ = load_brain(trainer, name)
    env = GAMES[game](seed=seed)
    obs = env.reset()
    mask = to_tensor(env.action_mask()[None], torch.bool)
    generator = torch.Generator().manual_seed(seed)
    start = env.state()
    frames = [build_frame(seq=0, game=game, episode=1, step=0, before=start, after=start, action=None,
                          probs=None, value=None, reward=0.0, ep_return=0.0, event=None,
                          valid=env.valid_actions, version=0, reset=True)]
    total = 0.0
    while True:
        with torch.no_grad():
            logits, value, acts = net(to_tensor(obs[None]), mask, return_activations=True)
        probs = torch.softmax(logits, -1)[0]
        action = int(probs.argmax()) if mode == "greedy" else int(torch.multinomial(probs, 1, generator=generator))
        before = env.state()
        next_obs, reward, terminated, truncated = env.step(action)
        total += reward
        done = terminated or truncated
        frames.append(build_frame(seq=len(frames), game=game, episode=1, step=env.steps, before=before,
                                  after=env.state(), action=action, probs=probs, value=value[0],
                                  reward=reward, ep_return=total, event=env.last_event,
                                  valid=env.valid_actions, version=0, acts=acts, obs=obs, done=done))
        obs = next_obs
        if done:
            break
    return clean({"label": label, "game": game, "seed": seed, "mode": mode, "frames": frames,
                  "total": total, "success": env.last_event in SUCCESS_EVENTS,
                  "brain": {"policy_w": rounded(net.policy.weight)}})
