"""Plasticity experiment: does a brain that already learned one game learn a new one
slower than a fresh brain? Does it forget the old game?

For every seed:
  1. Phase A  - train a new brain on the phase-A games.
  2. Phase B  - copy that brain, optionally apply a fix ("intervention"), and train it
                on the phase-B games. One branch per intervention, all starting from
                the same phase-A brain.
  3. Fresh    - a new brain with the SAME starting weights, trained only on phase B.

Every few updates each brain is tested (greedy, fixed test games) on all games, and
its plasticity metrics are measured on a fixed probe set of boards.

Run without the dashboard:
    uv run python -m experiments.plasticity --seeds 3
"""

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from agents.metrics import plasticity_metrics
from agents.network import SmallMultiGameNet
from agents.ppo import RolloutCollector, compute_gae, ppo_update
from experiments.baselines import optimal_score, random_baseline
from experiments.evaluate import evaluate_policy
from experiments.train import CONFIG, GAMES

INTERVENTIONS = {
    "none": "Keep training the same brain (no fix).",
    "reset_heads": "Re-initialize the policy and value output layers at the switch.",
    "shrink_perturb": "Shrink all weights to 40% and add 10% of a fresh random init (Ash & Adams, 2020).",
    "reset_optimizer": "Keep the weights but clear Adam's running averages.",
}
SHRINK, PERTURB = 0.4, 0.1
EVAL_SEED = 999_000  # the same test games at every evaluation

DEFAULT_SPEC = {
    "name": "Catch then Reach",
    "phase_a": ["catch"],
    "phase_b": ["reach"],
    "steps_a": 150_000,
    "steps_b": 300_000,
    "seeds": 3,
    "interventions": ["none"],
    "eval_every": 10,
    "eval_episodes": 30,
    "envs_per_game": 4,
    "lr": 5e-4,
    "ent_coef": 0.01,
}
QUICK_SPEC = dict(DEFAULT_SPEC, name="Quick test", steps_a=50_000, steps_b=100_000, seeds=2,
                  eval_every=5, eval_episodes=20)

SPEC_LIMITS = {
    "steps_a": (int, 10_000, 3_000_000),
    "steps_b": (int, 10_000, 3_000_000),
    "seeds": (int, 1, 10),
    "eval_every": (int, 1, 100),
    "eval_episodes": (int, 5, 200),
    "envs_per_game": (int, 1, 8),
    "lr": (float, 1e-5, 1e-2),
    "ent_coef": (float, 0.0, 0.2),
}

BASE_FIELDS = ["condition", "seed", "phase", "update", "phase_step", "run_step", "entropy",
               "dormant_conv1", "dormant_conv2", "dormant_fc", "eff_rank"]


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- spec

def validate_spec(spec):
    """Check a user-supplied experiment spec and fill in defaults. Raises ValueError."""
    spec = spec or {}
    out = dict(DEFAULT_SPEC)
    out["name"] = str(spec.get("name", out["name"])).strip()[:60] or DEFAULT_SPEC["name"]
    order = list(GAMES)
    for key in ("phase_a", "phase_b"):
        games = spec.get(key, out[key])
        if not isinstance(games, list):
            raise ValueError(f"{key} must be a list of games")
        games = sorted({g for g in games if g in GAMES}, key=order.index)
        if not games:
            raise ValueError(f"{key} needs at least one game")
        out[key] = games
    ivs = spec.get("interventions", out["interventions"])
    if not isinstance(ivs, list):
        raise ValueError("interventions must be a list")
    ivs = [iv for iv in INTERVENTIONS if iv in ivs]  # keep a fixed order
    if not ivs:
        raise ValueError("choose at least one intervention")
    out["interventions"] = ivs
    for key, (typ, lo, hi) in SPEC_LIMITS.items():
        if key in spec:
            try:
                value = typ(spec[key])
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number") from None
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            out[key] = min(max(value, lo), hi)
    return out


def conditions_for(spec):
    return ["fresh"] + [f"pretrained_{iv}" for iv in spec["interventions"]]


def all_games_of(spec):
    order = list(GAMES)
    return sorted(set(spec["phase_a"]) | set(spec["phase_b"]), key=order.index)


def total_steps(spec):
    return spec["seeds"] * (spec["steps_a"] + spec["steps_b"] * (len(spec["interventions"]) + 1))


# ---------------------------------------------------------------- running

def make_probe(games, size=512, seed=123):
    """A fixed set of boards (from random play) used to measure plasticity the same way every time."""
    rng = np.random.default_rng(seed)
    per = size // len(games)
    boards = []
    for i, g in enumerate(games):
        env = GAMES[g](seed=seed + i)
        obs = env.reset()
        count = per if i < len(games) - 1 else size - per * (len(games) - 1)
        for _ in range(count):
            boards.append(obs)
            obs, _, term, trunc = env.step(int(rng.choice(env.valid_actions)))
            if term or trunc:
                obs = env.reset()
    return torch.as_tensor(np.stack(boards))


class Progress:
    def __init__(self, total, on_progress, should_stop):
        self.total, self.on_progress, self.should_stop = total, on_progress, should_stop
        self.done = 0
        self.t0 = time.time()

    def check(self):
        if self.should_stop is not None and self.should_stop():
            raise Cancelled

    def advance(self, n, detail):
        self.done += n
        if self.on_progress is not None:
            sps = self.done / max(time.time() - self.t0, 1e-6)
            self.on_progress(min(1.0, self.done / self.total), detail, sps)


def apply_intervention(name, net, optimizer, seed, lr):
    """Change the pretrained brain at the switch. Returns the optimizer to use next."""
    if name == "reset_heads":
        torch.manual_seed(2_000_000 + seed)
        net.reinit(["policy", "value"])
    elif name == "shrink_perturb":
        torch.manual_seed(2_000_000 + seed)
        fresh = SmallMultiGameNet()
        with torch.no_grad():
            for p, q in zip(net.parameters(), fresh.parameters()):
                p.mul_(SHRINK).add_(q, alpha=PERTURB)
    elif name == "reset_optimizer":
        return torch.optim.Adam(net.parameters(), lr=lr)
    return optimizer


def train_phase(net, optimizer, *, phase, games, steps, env_seed, spec, all_games, probe,
                emit, progress, condition, seed, run_offset):
    """PPO on `games` for about `steps` moves, logging an evaluation row every few updates.
    The learning rate is kept constant, as is usual in plasticity studies."""
    cfg = dict(CONFIG, lr=spec["lr"], ent_coef=spec["ent_coef"])
    names = [g for g in games for _ in range(spec["envs_per_game"])]
    envs = [GAMES[g](seed=env_seed + i) for i, g in enumerate(names)]
    collector = RolloutCollector(envs, cfg["gamma"])
    batch_size = len(envs) * cfg["rollout_steps"]
    num_updates = max(1, steps // batch_size)
    for group in optimizer.param_groups:
        group["lr"] = cfg["lr"]

    def log(update, stats):
        row = {"condition": condition, "seed": seed, "phase": phase, "update": update,
               "phase_step": update * batch_size, "run_step": run_offset + update * batch_size,
               "entropy": stats["entropy"] if stats else None}
        row.update(plasticity_metrics(net, probe))
        for g in all_games:
            recent = [r for name, r in collector.finished if name == g][-100:]
            row[f"train_{g}"] = float(np.mean(recent)) if recent else None
            row[f"eval_{g}"] = float(evaluate_policy(net, g, spec["eval_episodes"], EVAL_SEED)["returns"].mean())
        emit(row)

    log(0, None)  # where the brain starts this phase
    for u in range(1, num_updates + 1):
        progress.check()
        batch, last_value = collector.collect(net, cfg["rollout_steps"])
        advantages, returns = compute_gae(batch["reward"], batch["value"], batch["done"],
                                          last_value, cfg["gamma"], cfg["lam"])
        stats = ppo_update(net, optimizer, batch, advantages, returns, cfg)
        progress.advance(batch_size, f"{condition} · seed {seed + 1}/{spec['seeds']} · "
                                     f"phase {phase} · update {u}/{num_updates}")
        if u % spec["eval_every"] == 0 or u == num_updates:
            log(u, stats)
    return num_updates * batch_size


def run_experiment(spec, out_dir, on_progress=None, should_stop=None):
    spec = validate_spec(spec)
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    all_games = all_games_of(spec)
    probe = make_probe(all_games)
    fields = BASE_FIELDS + [f"train_{g}" for g in all_games] + [f"eval_{g}" for g in all_games]
    progress = Progress(total_steps(spec), on_progress, should_stop)
    common = dict(spec=spec, all_games=all_games, probe=probe, progress=progress)

    def run_to_file(filename, **kwargs):
        with open(runs_dir / f"{filename}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            def emit(row):
                writer.writerow(row)
                f.flush()  # so the dashboard can show curves while it runs

            return train_phase(emit=emit, **common, **kwargs)

    for s in range(spec["seeds"]):
        # Phase A: learn the first game(s). Every intervention branches from this brain.
        torch.manual_seed(s)
        net = SmallMultiGameNet()
        opt = torch.optim.Adam(net.parameters(), lr=spec["lr"])
        steps_a = run_to_file(f"phaseA_s{s}", net=net, optimizer=opt, phase="A", games=spec["phase_a"],
                              steps=spec["steps_a"], env_seed=20_000 + 100 * s,
                              condition="pretrained", seed=s, run_offset=0)
        torch.save({"model": net.state_dict(), "config": {"games": spec["phase_a"], "seed": s}},
                   out_dir / f"phase_a_s{s}.pt")
        snap_net = copy.deepcopy(net.state_dict())
        snap_opt = copy.deepcopy(opt.state_dict())

        for iv in spec["interventions"]:
            net.load_state_dict(snap_net)
            opt = torch.optim.Adam(net.parameters(), lr=spec["lr"])
            opt.load_state_dict(copy.deepcopy(snap_opt))
            opt = apply_intervention(iv, net, opt, s, spec["lr"])
            torch.manual_seed(1_000_000 + s)  # same move sampling for every condition
            condition = f"pretrained_{iv}"
            run_to_file(f"{condition}_s{s}", net=net, optimizer=opt, phase="B", games=spec["phase_b"],
                        steps=spec["steps_b"], env_seed=10_000 + 100 * s,
                        condition=condition, seed=s, run_offset=steps_a)

        # Fresh brain: identical starting weights to the pretrained one, but no phase A.
        torch.manual_seed(s)
        net = SmallMultiGameNet()
        opt = torch.optim.Adam(net.parameters(), lr=spec["lr"])
        torch.manual_seed(1_000_000 + s)
        run_to_file(f"fresh_s{s}", net=net, optimizer=opt, phase="B", games=spec["phase_b"],
                    steps=spec["steps_b"], env_seed=10_000 + 100 * s,
                    condition="fresh", seed=s, run_offset=steps_a)

    summary = summarize(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------- analysis

def _number(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for raw in csv.DictReader(f):
            row = {k: (v if k in ("condition", "phase") else _number(v)) for k, v in raw.items()}
            rows.append(row)
        return rows


def bootstrap(values, n=2000, seed=0):
    """Mean and 95% bootstrap confidence interval across seeds."""
    vals = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if vals.size == 0:
        return None
    mean = float(vals.mean())
    if vals.size == 1:
        lo = hi = mean
    else:
        rng = np.random.default_rng(seed)
        means = rng.choice(vals, size=(n, vals.size)).mean(axis=1)
        lo, hi = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    return {"mean": mean, "lo": lo, "hi": hi, "n": int(vals.size), "values": vals.tolist()}


def aggregate(series):
    """series: list of (xs, ys), one per seed -> mean and ~95% band (mean ± 1.96 SE) at each x."""
    by_x = {}
    for xs, ys in series:
        for x, y in zip(xs, ys):
            if y is not None:
                by_x.setdefault(x, []).append(y)
    out = {"x": [], "mean": [], "lo": [], "hi": [], "n": []}
    for x in sorted(by_x):
        v = np.array(by_x[x])
        mean = float(v.mean())
        half = 1.96 * float(v.std(ddof=1)) / math.sqrt(v.size) if v.size > 1 else 0.0
        for key, val in (("x", x), ("mean", mean), ("lo", mean - half), ("hi", mean + half), ("n", int(v.size))):
            out[key].append(val)
    return out


def summarize(out_dir):
    out_dir = Path(out_dir)
    spec = json.loads((out_dir / "spec.json").read_text(encoding="utf-8"))
    games_a, games_b = spec["phase_a"], spec["phase_b"]
    all_games = all_games_of(spec)
    conditions = conditions_for(spec)
    baselines = {g: {"random": random_baseline(g), "optimal": optimal_score(g)} for g in all_games}

    phase_a, runs = {}, {}
    runs_dir = out_dir / "runs"
    for path in sorted(runs_dir.glob("*.csv")) if runs_dir.exists() else []:
        rows = read_rows(path)
        if not rows:
            continue
        seed = int(rows[0]["seed"])
        if path.stem.startswith("phaseA_"):
            phase_a[seed] = rows
        else:
            runs[(rows[0]["condition"], seed)] = rows

    batch_b = len(games_b) * spec["envs_per_game"] * CONFIG["rollout_steps"]
    expected_b = max(1, spec["steps_b"] // batch_b) * batch_b
    switch_step = max((rows[-1]["run_step"] for rows in phase_a.values()), default=None)

    def progress_of(g, v):
        b = baselines[g]
        return (v - b["random"]) / (b["optimal"] - b["random"])

    def last3(rows, key):
        vals = [r[key] for r in rows[-3:] if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    # ---- per-seed metrics (completed runs only)
    per_seed = {c: {} for c in conditions}
    for (c, s), rows in runs.items():
        if c not in per_seed or rows[-1]["phase_step"] < expected_b:
            continue
        games = {}
        xs = np.array([r["phase_step"] for r in rows], dtype=float)
        for g in games_b:
            ps = np.array([progress_of(g, r[f"eval_{g}"]) for r in rows])
            hit = np.flatnonzero(ps >= 0.9)
            games[g] = {
                "auc": float(np.trapezoid(ps, xs) / (xs[-1] - xs[0])) if xs[-1] > xs[0] else float(ps[-1]),
                "steps90": float(xs[hit[0]]) if hit.size else float(xs[-1]),
                "reached90": bool(hit.size),
                "final": last3(rows, f"eval_{g}"),
            }
        forgetting = {}
        has_a = c != "fresh" and s in phase_a
        if has_a:
            end_a = phase_a[s][-1]
            for g in games_a:
                if g not in games_b:
                    forgetting[g] = end_a[f"eval_{g}"] - last3(rows, f"eval_{g}")
        switch_row = phase_a[s][-1] if has_a else rows[0]
        # Exploration when phase B begins: end of phase A, or the fresh brain's first update.
        entropy = switch_row["entropy"] if has_a else next((r["entropy"] for r in rows if r["entropy"] is not None), None)
        per_seed[c][s] = {"games": games, "forgetting": forgetting,
                          "switch": {"dormant_fc": switch_row["dormant_fc"], "eff_rank": switch_row["eff_rank"],
                                     "entropy": entropy}}

    # ---- across seeds
    metrics = {}
    fresh = per_seed["fresh"]
    for c in conditions:
        seeds = per_seed[c]
        entry = {"seeds": sorted(seeds), "games": {}, "forgetting": {}, "switch": {}}
        for g in games_b:
            pick = lambda key: [seeds[s]["games"][g][key] for s in seeds]
            e = {k: bootstrap(pick(k)) for k in ("auc", "steps90", "final")}
            e["reached90"] = sum(seeds[s]["games"][g]["reached90"] for s in seeds)
            if c != "fresh":
                ratios = [seeds[s]["games"][g]["auc"] / fresh[s]["games"][g]["auc"]
                          for s in seeds if s in fresh and fresh[s]["games"][g]["auc"] > 0.05]
                e["auc_ratio"] = bootstrap(ratios)
            entry["games"][g] = e
        for g in games_a:
            if g not in games_b and c != "fresh":
                entry["forgetting"][g] = bootstrap([seeds[s]["forgetting"].get(g) for s in seeds])
        for key in ("dormant_fc", "eff_rank", "entropy"):
            entry["switch"][key] = bootstrap([seeds[s]["switch"][key] for s in seeds])
        metrics[c] = entry

    # ---- curves (include runs still in progress)
    def timeline(c, s, rows):
        if c != "fresh" and s in phase_a:
            return phase_a[s] + rows[1:]  # phase B's first row repeats the end of phase A (or the reset)
        return rows

    curves = {"learn": {}, "remember": {}, "timeline": {}}
    for g in games_b:
        curves["learn"][g] = {c: aggregate([([r["phase_step"] for r in rows], [r[f"eval_{g}"] for r in rows])
                                            for (cc, s), rows in runs.items() if cc == c]) for c in conditions}
    for g in games_a:
        if g in games_b:
            continue
        curves["remember"][g] = {}
        for c in conditions:
            series = [([r["run_step"] for r in timeline(c, s, rows)], [r[f"eval_{g}"] for r in timeline(c, s, rows)])
                      for (cc, s), rows in runs.items() if cc == c]
            if c != "fresh":  # show phase A even before phase B has started
                series += [([r["run_step"] for r in rows], [r[f"eval_{g}"] for r in rows])
                           for s, rows in phase_a.items() if (c, s) not in runs]
            curves["remember"][g][c] = aggregate(series)
    for key in ("dormant_fc", "eff_rank", "entropy"):
        curves["timeline"][key] = {}
        for c in conditions:
            series = [([r["run_step"] for r in timeline(c, s, rows)], [r[key] for r in timeline(c, s, rows)])
                      for (cc, s), rows in runs.items() if cc == c]
            if c != "fresh":
                series += [([r["run_step"] for r in rows], [r[key] for r in rows])
                           for s, rows in phase_a.items() if (c, s) not in runs]
            curves["timeline"][key][c] = aggregate(series)

    done = sum(len(v) for v in per_seed.values())
    return {
        "spec": spec, "games_a": games_a, "games_b": games_b, "all_games": all_games,
        "conditions": conditions, "baselines": baselines, "switch_step": switch_step,
        "metrics": metrics, "curves": curves,
        "runs_done": done, "runs_total": len(conditions) * spec["seeds"],
    }


# ---------------------------------------------------------------- command line

def main():
    parser = argparse.ArgumentParser(description="Run a plasticity experiment")
    parser.add_argument("--phase-a", nargs="+", default=DEFAULT_SPEC["phase_a"])
    parser.add_argument("--phase-b", nargs="+", default=DEFAULT_SPEC["phase_b"])
    parser.add_argument("--steps-a", type=int, default=DEFAULT_SPEC["steps_a"])
    parser.add_argument("--steps-b", type=int, default=DEFAULT_SPEC["steps_b"])
    parser.add_argument("--seeds", type=int, default=DEFAULT_SPEC["seeds"])
    parser.add_argument("--interventions", nargs="+", default=["none"], choices=list(INTERVENTIONS))
    parser.add_argument("--out", default=f"results/experiments/cli_{time.strftime('%Y%m%d-%H%M%S')}")
    args = parser.parse_args()
    spec = validate_spec({"name": "CLI run", "phase_a": args.phase_a, "phase_b": args.phase_b,
                          "steps_a": args.steps_a, "steps_b": args.steps_b, "seeds": args.seeds,
                          "interventions": args.interventions})
    print(f"Running {len(conditions_for(spec)) * spec['seeds']} runs, {total_steps(spec):,} moves in total.")
    summary = run_experiment(spec, args.out, on_progress=lambda f, d, sps: print(
        f"\r{f:6.1%} {d} ({sps:,.0f} moves/s)   ", end="", flush=True))
    print()
    for g in summary["games_b"]:
        print(f"\nLearning {g} (phase B):")
        for c in summary["conditions"]:
            m = summary["metrics"][c]["games"][g]
            ratio = m.get("auc_ratio")
            ratio_text = f"{ratio['mean']:.2f} [{ratio['lo']:.2f}, {ratio['hi']:.2f}]" if ratio else "—"
            print(f"  {c:<28} AUC ratio vs fresh {ratio_text:<20} final {m['final']['mean']:+.2f}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
