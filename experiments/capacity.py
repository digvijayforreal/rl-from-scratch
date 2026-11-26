"""Score saved models on every game, to see how many games one small network can hold.

    uv run python -m experiments.capacity results/capacity/*.pt
    uv run python -m experiments.capacity models/multigame_small.pt --episodes 500 --mode sample

For each model and game it plays fresh test games (seeds the model never trained on) and
reports the average score and "skill":

    skill = (score - random score) / (best possible score - random score)

so 0% means no better than random and 100% means perfect. Games the model was not
trained on are marked with a * (they show what it does on a game it has never seen).

Two ways of playing are scored, because they can differ a lot:
    greedy  always take the most likely move. The network has no memory, so if its best
            move bumps into a wall the board does not change and it repeats the bump forever.
    sample  pick moves at random with the network's probabilities, as during training.
            The randomness lets it escape those loops.
"""

import argparse
import glob
import json
import os

import numpy as np

from experiments.baselines import optimal_score, random_baseline
from experiments.evaluate import evaluate_policy, load_model
from experiments.train import GAMES

TEST_SEED = 999
MODES = ("greedy", "sample")


def skill(score, game):
    rand, best = random_baseline(game), optimal_score(game)
    return (score - rand) / (best - rand)


def score_model(path, episodes, modes=MODES):
    net, saved = load_model(path)
    return score_net(net, saved.get("config", {}).get("games", []), os.path.basename(path),
                     episodes, modes)


def score_net(net, trained_on, name, episodes, modes=MODES):
    """Score a network that is already in memory (used right after training)."""
    out = {"model": name, "trained_on": list(trained_on), "modes": {}}
    for mode in modes:
        games = {}
        for game in GAMES:
            res = evaluate_policy(net, game, episodes, TEST_SEED, mode=mode)
            r = res["returns"]
            games[game] = {"score": float(r.mean()),
                           "ci95": float(1.96 * r.std() / np.sqrt(len(r))),
                           "win_rate": float(res["success"].mean()),
                           "skill": float(skill(r.mean(), game)),
                           "trained": game in trained_on}
        out["modes"][mode] = games
    return out


def format_table(results, mode):
    names = list(GAMES)
    head = f"{'model (' + mode + ')':<26}" + "".join(f"{g:>16}" for g in names) + f"{'avg skill*':>12}"
    lines = [head, "-" * len(head)]
    lines.append(f"{'random / best':<26}" + "".join(
        f"{random_baseline(g):>+7.2f}/{optimal_score(g):<+8.2f}" for g in names))
    for res in results:
        cells, trained_skills = [], []
        for g in names:
            d = res["modes"][mode][g]
            mark = " " if d["trained"] else "*"
            cells.append(f"{d['score']:>+7.2f} {d['skill'] * 100:>4.0f}%{mark}")
            if d["trained"]:
                trained_skills.append(d["skill"])
        avg = np.mean(trained_skills) * 100 if trained_skills else float("nan")
        lines.append(f"{res['model']:<26}" + "".join(f"{c:>16}" for c in cells) + f"{avg:>11.0f}%")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Score saved models on every game.")
    parser.add_argument("models", nargs="+", help="model files (wildcards are fine)")
    parser.add_argument("--episodes", type=int, default=500, help="test games per game")
    parser.add_argument("--mode", choices=(*MODES, "both"), default="both")
    parser.add_argument("--json", default=None, help="also write the results to this file")
    args = parser.parse_args()

    paths = sorted({p for pattern in args.models for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit("No model files found.")
    modes = MODES if args.mode == "both" else (args.mode,)
    results = [score_model(p, args.episodes, modes) for p in paths]
    for mode in modes:
        print(format_table(results, mode))
        print()
    print("each cell: average score and skill (0% = random, 100% = best possible)")
    print("* = the model was not trained on this game;  avg skill* = mean over trained games only")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"episodes": args.episodes, "test_seed": TEST_SEED, "results": results}, f, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
