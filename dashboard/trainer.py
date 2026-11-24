"""Background PPO trainer and live 'showcase' player for the dashboard.

Trainer  : runs the same PPO loop as experiments/train.py in a thread, records everything
           the dashboard shows, and saves each run to results/runs/<run_id>/.
Showcase : plays one copy of every game at human speed with the latest weights, so you
           can watch the agent improve while it trains.
"""

import csv
import math
import threading
import time
from collections import deque

import numpy as np
import torch

from agents.metrics import plasticity_metrics
from agents.network import LAYER_NAMES, SmallMultiGameNet, count_parameters
from agents.ppo import RolloutCollector, compute_gae, ppo_update, to_tensor
from dashboard import storage
from envs.base import ACTION_NAMES, AGENT, CHANNEL_NAMES, GOAL, GRID, ITEM
from experiments.baselines import SUCCESS_EVENTS, optimal_score, random_baseline
from experiments.train import CONFIG, GAMES, make_envs

# Settings the dashboard may change, with (type, min, max).
CONFIG_LIMITS = {
    "lr": (float, 1e-5, 1e-2),
    "ent_coef": (float, 0.0, 0.2),
    "clip": (float, 0.05, 0.5),
    "gamma": (float, 0.8, 0.999),
    "lam": (float, 0.5, 1.0),
    "total_steps": (int, 10_000, 5_000_000),
    "envs_per_game": (int, 1, 8),
    "rollout_steps": (int, 16, 512),
    "epochs": (int, 1, 10),
    "minibatches": (int, 1, 16),
    "seed": (int, 0, 10_000),
}


class StopTraining(Exception):
    pass


def clean(x):
    """Make a value JSON-safe (no NaN or infinity, numpy numbers become Python numbers)."""
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def rounded(t, digits=3):
    return np.round(t.detach().cpu().numpy().astype(np.float64), digits).ravel().tolist()


def sanitize_config(updates, base):
    cfg = dict(base)
    for key, value in (updates or {}).items():
        if key not in CONFIG_LIMITS:
            continue
        typ, lo, hi = CONFIG_LIMITS[key]
        try:
            value = typ(value)
        except (TypeError, ValueError):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        cfg[key] = min(max(value, lo), hi)
    return cfg


def build_frame(*, seq, game, episode, step, before, after, action, probs, value, reward,
                ep_return, event, valid, version, acts=None, obs=None, done=False, reset=False,
                recent_return=None, success_rate=None):
    """One moment of a game, in the format the dashboard draws (live player and replays)."""
    frame = {
        "seq": seq, "game": game, "episode": episode, "step": step,
        "prev": before, "state": after, "reset": reset, "done": done, "event": event,
        "action": action, "action_name": None if action is None else ACTION_NAMES[action],
        "valid": valid,
        "probs": None if probs is None else rounded(probs, 4),
        "value": None if value is None else float(value),
        "reward": reward, "ep_return": round(ep_return, 3),
        "recent_return": recent_return, "success_rate": success_rate,
        "weights_version": version,
    }
    if acts is not None:
        frame["obs"] = obs.astype(np.int8).ravel().tolist()
        frame["acts"] = {name: rounded(a, 3) for name, a in acts.items()}
    return clean(frame)


class Trainer:
    def __init__(self):
        self.lock = threading.Lock()        # held while the training thread uses the network
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.thread = None
        self.stop_flag = False
        self.delay_ms = 0                   # slow-motion: pause this long after every move
        self.version = 0                    # increases whenever the weights change
        self.run_id = 0
        self.events = []                    # log shown on the dashboard
        self.games = list(CONFIG["games"])
        self.baselines = {g: {"random": random_baseline(g), "optimal": optimal_score(g)}
                          for g in self.games}
        self.run_dir = None
        self.build(dict(CONFIG))

    # ---------- setup ----------

    def build(self, cfg):
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        self.cfg = cfg
        self.envs = make_envs(cfg)
        self.net = SmallMultiGameNet()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])
        self.collector = RolloutCollector(self.envs, cfg["gamma"])
        self.batch_size = len(self.envs) * cfg["rollout_steps"]
        self.num_updates = max(1, cfg["total_steps"] // self.batch_size)
        self.num_params = count_parameters(self.net)

        self.run_id += 1
        self.run_dir = None  # created when training starts
        self.status = "idle"
        self.phase = "idle"
        self.phase_progress = 0.0
        self.update = 0
        self.env_steps = 0
        self.active_time = 0.0
        self.update_started = None
        self.sps = 0.0
        self.lr = cfg["lr"]
        self.history = []
        self.milestones = set()
        self._grad_sums = {}
        self._mb_count = 0
        self.arena = [self._arena_entry(env, None, None) for env in self.envs]
        self._publish(grad_norm=None, weight_change=None)
        self.log("info", f"New network built: {self.num_params:,} parameters, "
                         f"{len(self.envs)} games in parallel, {self.num_updates} updates planned.")

    def log(self, kind, text):
        self.events.append({"id": len(self.events), "kind": kind, "text": text,
                            "time": time.strftime("%H:%M:%S")})

    # ---------- run folder ----------

    def _ensure_run_dir(self):
        if self.run_dir is not None:
            return
        run_name = f"train_{time.strftime('%Y%m%d-%H%M%S')}_s{self.cfg['seed']}"
        self.run_dir = storage.RUNS_DIR / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        storage.write_json(self.run_dir / "config.json", self.cfg)
        storage.write_json(self.run_dir / "meta.json", {
            "id": run_name, "kind": "run", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": "", "games": self.games, "config": self.cfg, "params": self.num_params})
        self.log("info", f"Saving this run to results/runs/{run_name}/")

    def _write_meta(self):
        if self.run_dir is None:
            return
        path = self.run_dir / "meta.json"
        meta = storage.read_json(path, default={}) or {}  # keep notes the user wrote
        last = self.history[-1] if self.history else {}
        meta.update(status=self.status, update=self.update, num_updates=self.num_updates,
                    env_steps=self.env_steps, minutes=round(self.active_time / 60, 2),
                    scores={g: last.get(f"score_{g}") for g in self.games})
        storage.write_json(path, clean(meta))

    # ---------- controls ----------

    def start(self):
        if self.status == "paused":
            return self.resume()
        if self.status != "idle":
            return
        self._ensure_run_dir()
        self.stop_flag = False
        self.resume_event.set()
        self.status = "running"
        self._write_meta()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log("info", "Training started.")

    def pause(self):
        if self.status == "running":
            self.resume_event.clear()
            self.status = "paused"
            self._write_meta()
            self.log("info", f"Paused at update {self.update}.")

    def resume(self):
        if self.status == "paused":
            self.status = "running"
            self.resume_event.set()
            self._write_meta()
            self.log("info", "Resumed.")

    def reset(self, updates=None):
        self.stop_flag = True
        self.resume_event.set()
        if self.thread is not None:
            self.thread.join(timeout=30)
        self.thread = None
        if self.run_dir is not None and self.status != "finished":
            self.status = "stopped"
            self._write_meta()
        self.build(sanitize_config(updates, self.cfg))

    def save(self):
        self._ensure_run_dir()
        path = self.run_dir / f"u{self.update:04d}.pt"
        with self.lock:
            torch.save({"model": self.net.state_dict(), "config": self.cfg,
                        "update": self.update, "env_steps": self.env_steps}, path)
        name = path.relative_to(storage.RESULTS_DIR).as_posix()
        self.log("success", f"Saved checkpoint {name}")
        return name

    def load(self, name):
        """Load weights into the current network (training must not be running)."""
        if self.status == "running":
            raise ValueError("Pause training before loading a checkpoint.")
        path = storage.resolve_checkpoint(name)
        saved = torch.load(path, weights_only=True)
        with self.lock:
            self.net.load_state_dict(saved["model"])
            self._publish(grad_norm=None, weight_change=None)
        self.log("success", f"Loaded {name}. The live players now use these weights.")

    # ---------- training loop ----------

    def _run(self):
        cfg = self.cfg
        try:
            while self.update < self.num_updates and not self.stop_flag:
                u = self.update + 1
                self.lr = cfg["lr"] * (1 - (u - 1) / self.num_updates)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.lr
                self.update_started = time.time()

                with self.lock:
                    self.phase, self.phase_progress = "collect", 0.0
                    batch, last_value = self.collector.collect(self.net, cfg["rollout_steps"],
                                                               on_step=self._on_step)
                    self.phase, self.phase_progress = "advantage", 1.0
                    advantages, returns = compute_gae(batch["reward"], batch["value"], batch["done"],
                                                      last_value, cfg["gamma"], cfg["lam"])
                    self.phase, self.phase_progress = "update", 0.0
                    before = {n: getattr(self.net, n).weight.detach().clone() for n in LAYER_NAMES}
                    self._grad_sums = {n: 0.0 for n in LAYER_NAMES}
                    self._mb_count = 0
                    stats = ppo_update(self.net, self.optimizer, batch, advantages, returns, cfg,
                                       on_minibatch=self._on_minibatch)
                    grad_norm = {n: self._grad_sums[n] / max(1, self._mb_count) for n in LAYER_NAMES}
                    change = {}
                    for n in LAYER_NAMES:
                        w = getattr(self.net, n).weight.detach()
                        change[n] = float((w - before[n]).norm() / (before[n].norm() + 1e-12))
                    obs = batch["obs"].reshape(-1, *batch["obs"].shape[2:])
                    plasticity = plasticity_metrics(self.net, obs)
                    self._publish(grad_norm=grad_norm, weight_change=change)

                duration = time.time() - self.update_started
                self.active_time += duration
                self.update_started = None
                self.sps = self.batch_size / max(duration, 1e-6)
                self.update = u
                self._record(stats, plasticity)

            if not self.stop_flag:
                self.status, self.phase = "finished", "idle"
                self.log("success", "Training finished.")
                self.save()
                self._write_meta()
        except StopTraining:
            pass
        except Exception as exc:  # show the error on the dashboard instead of dying silently
            self.status, self.phase = "error", "idle"
            self.log("warn", f"Training crashed: {exc!r}")
            self._write_meta()
            raise

    def _on_step(self, t, actions, probs):
        self.env_steps += len(self.envs)
        self.phase_progress = (t + 1) / self.cfg["rollout_steps"]
        self.arena = [self._arena_entry(env, int(actions[i]), probs[i])
                      for i, env in enumerate(self.envs)]
        if self.stop_flag:
            raise StopTraining
        if self.delay_ms > 0 or not self.resume_event.is_set():
            # Let go of the network while waiting, so a checkpoint can be loaded during a pause.
            self.lock.release()
            try:
                if self.delay_ms > 0:
                    time.sleep(self.delay_ms / 1000)
                self.resume_event.wait()
            finally:
                self.lock.acquire()
            if self.stop_flag:
                raise StopTraining

    def _on_minibatch(self, epoch):
        for n in LAYER_NAMES:
            grad = getattr(self.net, n).weight.grad
            self._grad_sums[n] += float(grad.norm()) if grad is not None else 0.0
        self._mb_count += 1
        self.phase_progress = self._mb_count / (self.cfg["epochs"] * self.cfg["minibatches"])

    def _arena_entry(self, env, action, probs):
        return {"game": env.name, "state": env.state(), "action": action,
                "prob": None if probs is None or action is None else round(float(probs[action]), 3)}

    # ---------- brain statistics ----------

    def _publish(self, grad_norm, weight_change):
        """Share a copy of the weights with the live players, plus brain stats for the dashboard."""
        net = self.net
        histograms = {}
        for n in LAYER_NAMES:
            w = getattr(net, n).weight.detach().cpu().numpy().ravel()
            lim = float(np.abs(w).max()) or 1.0
            counts, _ = np.histogram(w, bins=31, range=(-lim, lim))
            histograms[n] = {"lim": lim, "counts": counts.tolist(),
                             "norm": float(np.linalg.norm(w)), "count": int(w.size)}
        conv1 = net.conv1.weight.detach()[:, [AGENT, GOAL, ITEM]]  # 16 filters x 3 channels x 3 x 3
        self.version += 1
        self.brain = clean({
            "version": self.version,
            "update": self.update,
            "policy_w": rounded(net.policy.weight),  # 6 x 128
            "value_w": rounded(net.value.weight),    # 128
            "conv1": rounded(conv1),
            "conv1_channels": ["agent", "goal", "item"],
            "hist": histograms,
            "grad_norm": grad_norm,
            "weight_change": weight_change,
        })
        self.weights = (self.version, {k: v.detach().clone() for k, v in net.state_dict().items()})

    def _record(self, stats, plasticity):
        row = {"update": self.update, "step": self.env_steps, "lr": self.lr,
               "time": round(self.active_time, 2), "sps": self.sps, **stats, **plasticity}
        for g in self.games:
            recent = [r for name, r in self.collector.finished if name == g][-100:]
            row[f"score_{g}"] = float(np.mean(recent)) if recent else None
        row = clean(row)
        self.history.append(row)
        if self.run_dir is not None:
            path = self.run_dir / "log.csv"
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    writer.writeheader()
                writer.writerow(row)
            if self.update % 10 == 0:
                self._write_meta()
        self._check_milestones(row)

    def _check_milestones(self, row):
        for g in self.games:
            score = row.get(f"score_{g}")
            base = self.baselines[g]
            if score is None or base["optimal"] is None:
                continue
            progress = (score - base["random"]) / (base["optimal"] - base["random"])
            for level in (0.5, 0.9):
                key = (g, level)
                if progress >= level and key not in self.milestones:
                    self.milestones.add(key)
                    self.log("success", f"{g.title()} reached {int(level * 100)}% of the way from "
                                        f"random to perfect at {self.env_steps:,} steps "
                                        f"(score {score:+.2f}).")
        if row["entropy"] < 0.05 and "low_entropy" not in self.milestones:
            self.milestones.add("low_entropy")
            self.log("warn", "Entropy is very low: the agent has almost stopped exploring.")
        if row["approx_kl"] > 0.05 and row["update"] - getattr(self, "_last_kl_warn", -99) >= 20:
            self._last_kl_warn = row["update"]  # warn at most once every 20 updates
            self.log("warn", f"Update {row['update']}: large policy jump (KL {row['approx_kl']:.3f}).")
        if row["update"] % 10 == 0:
            scores = " · ".join(f"{g} {row[f'score_{g}']:+.2f}" for g in self.games
                                if row.get(f"score_{g}") is not None)
            self.log("info", f"Update {row['update']}/{self.num_updates} · {scores}")

    def elapsed(self):
        partial = time.time() - self.update_started if self.update_started else 0.0
        return self.active_time + (partial if self.status == "running" else 0.0)


class Showcase:
    """Plays every game at watchable speed using the newest weights."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.fps = 6
        self.mode = "greedy"  # "greedy" = always the best move, "sample" = like during training
        self.net = SmallMultiGameNet()
        self.net.eval()
        self.version = -1
        self.generator = torch.Generator().manual_seed(0)
        self.envs = {g: GAMES[g](seed=7000 + i) for i, g in enumerate(trainer.games)}
        self.obs = {g: env.reset() for g, env in self.envs.items()}
        self.ep_return = {g: 0.0 for g in self.envs}
        self.episode = {g: 1 for g in self.envs}
        self.needs_reset = {g: False for g in self.envs}
        self.results = {g: deque(maxlen=50) for g in self.envs}
        self.frames = {}
        self.seq = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _sync(self):
        version, weights = self.trainer.weights
        if version != self.version:
            self.net.load_state_dict(weights)
            self.version = version

    def _run(self):
        while True:
            started = time.time()
            try:
                self._sync()
                for g, env in self.envs.items():
                    self._tick(g, env)
            except Exception as exc:
                self.trainer.log("warn", f"Live player error: {exc!r}")
                time.sleep(1)
            time.sleep(max(0.0, 1 / self.fps - (time.time() - started)))

    def _stats(self, g):
        results = self.results[g]
        if not results:
            return None, None
        return float(np.mean([r for r, _ in results])), float(np.mean([s for _, s in results]))

    def _tick(self, g, env):
        self.seq += 1
        if self.needs_reset[g]:
            self.obs[g] = env.reset()
            self.ep_return[g] = 0.0
            self.episode[g] += 1
            self.needs_reset[g] = False
            state = env.state()
            recent, rate = self._stats(g)
            self.frames[g] = build_frame(
                seq=self.seq, game=g, episode=self.episode[g], step=0, before=state, after=state,
                action=None, probs=None, value=None, reward=0.0, ep_return=0.0, event=None,
                valid=env.valid_actions, version=self.version, reset=True,
                recent_return=recent, success_rate=rate)
            return

        obs = self.obs[g]
        mask = to_tensor(env.action_mask()[None], torch.bool)
        with torch.no_grad():
            logits, value, acts = self.net(to_tensor(obs[None]), mask, return_activations=True)
        probs = torch.softmax(logits, -1)[0]
        if self.mode == "greedy":
            action = int(probs.argmax())
        else:
            action = int(torch.multinomial(probs, 1, generator=self.generator))

        before = env.state()
        self.obs[g], reward, terminated, truncated = env.step(action)
        self.ep_return[g] += reward
        done = terminated or truncated
        if done:
            self.needs_reset[g] = True
            self.results[g].append((self.ep_return[g], env.last_event in SUCCESS_EVENTS))
        recent, rate = self._stats(g)
        self.frames[g] = build_frame(
            seq=self.seq, game=g, episode=self.episode[g], step=env.steps, before=before,
            after=env.state(), action=action, probs=probs, value=value[0], reward=reward,
            ep_return=self.ep_return[g], event=env.last_event, valid=env.valid_actions,
            version=self.version, acts=acts, obs=obs, done=done,
            recent_return=recent, success_rate=rate)


def init_payload(trainer):
    return clean({
        "config": trainer.cfg,
        "limits": {k: [lo, hi] for k, (_, lo, hi) in CONFIG_LIMITS.items()},
        "games": trainer.games,
        "num_envs": len(trainer.envs),
        "num_updates": trainer.num_updates,
        "params": trainer.num_params,
        "baselines": trainer.baselines,
        "actions": ACTION_NAMES,
        "channels": CHANNEL_NAMES,
        "layers": LAYER_NAMES,
        "grid": GRID,
    })
