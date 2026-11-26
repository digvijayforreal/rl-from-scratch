"""Train one small network on several games at once.

    uv run python -m experiments.train --games catch reach maze snake --steps-per-game 1000000 --name four_games

Every run gets its own folder, results/runs/<name>/, which the dashboard's Run history shows too:

    config.json      the settings used
    meta.json        status, progress and latest scores (updated as it trains)
    log.csv          one row per 20 updates: training score of every game
    latest.pt        the newest checkpoint (saved every --save-every updates)
    model.pt         the finished model (latest.pt is removed when training completes)
    evaluation.json  the final test of every game against random play

Press Ctrl+C at any time: the run saves latest.pt and tells you how to carry on from it.
To start from an existing model instead of a new one, add --init path/to/model.pt (this
keeps the weights; the optimizer and the learning-rate schedule start fresh).
"""

import argparse
import csv
import dataclasses
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from agents.network import SmallMultiGameNet, count_parameters
from agents.ppo import RolloutCollector, compute_gae, ppo_update
from dashboard import storage
from envs.catch import Catch
from envs.maze import Maze
from envs.reach import Reach
from envs.snake import Snake

GAMES = {"catch": Catch, "reach": Reach, "maze": Maze, "snake": Snake}

CONFIG = {
    "games": ["catch", "reach"],
    "envs_per_game": 4,       # 4 copies of every game running together
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

LOG_EVERY = 20  # updates between log lines


def make_envs(cfg):
    names = [g for g in cfg["games"] for _ in range(cfg["envs_per_game"])]
    return [GAMES[name](seed=cfg["seed"] * 1000 + i) for i, name in enumerate(names)]


@dataclasses.dataclass
class TrainResult:
    net: torch.nn.Module
    history: list          # one row per log line: update, step, time, score_<game>
    update: int            # updates completed
    num_updates: int       # updates planned
    interrupted: bool      # stopped early with Ctrl+C


def train(cfg, init=None, log=print, on_row=None, on_checkpoint=None, save_every=0):
    """Train a new network (or continue from the weights in `init`).

    on_row(row)               is called at every log line
    on_checkpoint(net, update) is called every `save_every` updates
    Ctrl+C stops training cleanly and returns what was learned so far.
    """
    torch.manual_seed(cfg["seed"])
    envs = make_envs(cfg)

    net = SmallMultiGameNet()
    if init is not None:
        net.load_state_dict(torch.load(init, weights_only=True)["model"])
        log(f"Continuing from {init}")
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    collector = RolloutCollector(envs, cfg["gamma"])
    moves_per_update = len(envs) * cfg["rollout_steps"]
    num_updates = max(1, cfg["total_steps"] // moves_per_update)
    log(f"Games: {', '.join(cfg['games'])} | parameters: {count_parameters(net):,} | "
        f"{num_updates} updates of {moves_per_update:,} moves = {num_updates * moves_per_update:,} moves")

    start = time.time()
    history, done_updates, interrupted = [], 0, False
    try:
        for update in range(1, num_updates + 1):
            # Lower the learning rate slowly to 0 by the end.
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * (1 - (update - 1) / num_updates)

            batch, last_value = collector.collect(net, cfg["rollout_steps"])
            advantages, returns = compute_gae(batch["reward"], batch["value"], batch["done"],
                                              last_value, cfg["gamma"], cfg["lam"])
            ppo_update(net, optimizer, batch, advantages, returns, cfg)
            done_updates = update

            if save_every and on_checkpoint and update % save_every == 0 and update < num_updates:
                on_checkpoint(net, update)

            if update % LOG_EVERY == 0 or update == num_updates:
                elapsed = time.time() - start
                row = {"update": update, "step": update * moves_per_update, "time": round(elapsed, 1)}
                for name in cfg["games"]:
                    recent = [r for g, r in collector.finished if g == name][-100:]
                    row[f"score_{name}"] = float(np.mean(recent)) if recent else float("nan")
                history.append(row)
                scores = " | ".join(f"{g} {row[f'score_{g}']:+.2f}" for g in cfg["games"])
                eta = elapsed / update * (num_updates - update) / 60
                log(f"update {update:4d}/{num_updates} | {scores} | {elapsed / 60:.1f} min, "
                    f"~{eta:.0f} min left")
                if on_row:
                    on_row(row)
    except KeyboardInterrupt:
        interrupted = True
    return TrainResult(net, history, done_updates, num_updates, interrupted)


def save_model(net, cfg, path, update, env_steps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "config": cfg, "update": update, "env_steps": env_steps}, path)


def verdict(skill_percent):
    if skill_percent >= 90:
        return "learned"
    if skill_percent >= 50:
        return "partly learned"
    return "not learned yet"


def run(cfg, init=None, name=None, save_every=50, eval_episodes=200, copy_to=None, log=print):
    """Train, save everything into results/runs/<name>/, then test every game.

    Returns the run folder.
    """
    run_id = storage.check_id(name or f"train_{time.strftime('%Y%m%d-%H%M%S')}_s{cfg['seed']}")
    run_dir = storage.RUNS_DIR / run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} already exists. Pick another --name so nothing is overwritten.")
    run_dir.mkdir(parents=True)

    moves_per_update = len(cfg["games"]) * cfg["envs_per_game"] * cfg["rollout_steps"]
    meta = {"id": run_id, "kind": "run", "created": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": "",
            "games": list(cfg["games"]), "config": cfg, "params": count_parameters(SmallMultiGameNet()),
            "status": "running", "update": 0, "num_updates": max(1, cfg["total_steps"] // moves_per_update),
            "env_steps": 0, "minutes": 0.0, "scores": {}, "started_from": init}
    storage.write_json(run_dir / "config.json", cfg)
    storage.write_json(run_dir / "meta.json", meta)
    log(f"Run folder: {run_dir}")

    def on_row(row):
        path = run_dir / "log.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()})
        meta.update(update=row["update"], env_steps=row["step"], minutes=round(row["time"] / 60, 2),
                    scores={g: (None if np.isnan(row[f"score_{g}"]) else row[f"score_{g}"])
                            for g in cfg["games"]})
        storage.write_json(run_dir / "meta.json", meta)

    def on_checkpoint(net, update):
        save_model(net, cfg, run_dir / "latest.pt", update, update * moves_per_update)

    result = train(cfg, init=init, log=log, on_row=on_row, on_checkpoint=on_checkpoint,
                   save_every=save_every)
    steps = result.update * moves_per_update

    if result.interrupted:
        save_model(result.net, cfg, run_dir / "latest.pt", result.update, steps)
        meta.update(status="stopped", update=result.update, env_steps=steps)
        storage.write_json(run_dir / "meta.json", meta)
        log(f"\nStopped at update {result.update}/{result.num_updates}. Saved {run_dir / 'latest.pt'}")
        log("To carry on from there, run the same command with: "
            f"--init \"{run_dir / 'latest.pt'}\" --name <a new name>")
        return run_dir

    save_model(result.net, cfg, run_dir / "model.pt", result.update, steps)
    (run_dir / "latest.pt").unlink(missing_ok=True)
    meta.update(status="finished", update=result.update, env_steps=steps)
    storage.write_json(run_dir / "meta.json", meta)
    if copy_to:
        Path(copy_to).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run_dir / "model.pt", copy_to)
        log(f"Copied the model to {copy_to}")
    log(f"\nTraining finished. Model: {run_dir / 'model.pt'}")

    if eval_episodes > 0:
        from experiments.capacity import MODES, format_table, score_net  # imported here: it imports this file
        log(f"\nTesting every game on {eval_episodes} fresh games each...\n")
        res = score_net(result.net, cfg["games"], "this run", eval_episodes)
        for mode in MODES:
            log(format_table([res], mode) + "\n")
        log("each cell: average score and skill (0% = random play, 100% = best possible)")
        log("* = a game this model was not trained on\n")
        for game in cfg["games"]:
            sampled = res["modes"]["sample"][game]["skill"] * 100
            best_move = res["modes"]["greedy"][game]["skill"] * 100
            hint = "  -> train this game for more moves" if sampled < 50 else ""
            log(f"  {game:<6} {verdict(sampled):<16} (sampled play {sampled:.0f}%, best-move play "
                f"{best_move:.0f}%){hint}")
        storage.write_json(run_dir / "evaluation.json", res)
    return run_dir


def build_parser():
    p = argparse.ArgumentParser(
        description="Train one network on several games at once.",
        epilog="Example: python -m experiments.train --games catch reach maze snake "
               "--steps-per-game 1000000 --name four_games")
    p.add_argument("--games", nargs="+", choices=list(GAMES), default=CONFIG["games"])
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--steps-per-game", type=int, help="game moves each game gets (recommended)")
    budget.add_argument("--steps", type=int, help="total game moves across all games")
    p.add_argument("--envs-per-game", type=int, default=CONFIG["envs_per_game"])
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--name", default=None, help="run folder name (default: train_<date>_s<seed>)")
    p.add_argument("--init", default=None, help="start from this saved model instead of a new one")
    p.add_argument("--save-every", type=int, default=50, help="updates between checkpoints (0 = never)")
    p.add_argument("--eval-episodes", type=int, default=200, help="test games per game at the end (0 = skip)")
    p.add_argument("--save", default=None, help="also copy the finished model to this file")
    p.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    return p


def config_from_args(args):
    games = list(dict.fromkeys(args.games))  # in order, no repeats
    if args.steps_per_game:
        total = args.steps_per_game * len(games)
    else:
        total = args.steps or CONFIG["total_steps"]
    return {**CONFIG, "games": games, "total_steps": total, "seed": args.seed,
            "envs_per_game": args.envs_per_game}


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    run(config_from_args(args), init=args.init, name=args.name, save_every=args.save_every,
        eval_episodes=args.eval_episodes, copy_to=args.save)


if __name__ == "__main__":
    main()
