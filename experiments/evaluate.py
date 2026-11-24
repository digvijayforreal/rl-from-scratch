"""Load the trained model and compare it with a random agent on every game."""

import sys

import numpy as np
import torch

from agents.network import SmallMultiGameNet
from agents.ppo import to_tensor
from experiments.train import GAMES


def play(env, choose, episodes):
    total = 0.0
    for _ in range(episodes):
        obs, done = env.reset(), False
        while not done:
            obs, reward, terminated, truncated = env.step(choose(obs))
            total += reward
            done = terminated or truncated
    return total / episodes


def main(path="results/multigame_small.pt", episodes=200):
    saved = torch.load(path)
    net = SmallMultiGameNet()
    net.load_state_dict(saved["model"])
    net.eval()
    rng = np.random.default_rng(123)

    print(f"{'game':<8}{'random':>10}{'trained':>10}")
    for name in saved["config"]["games"]:
        env = GAMES[name](seed=999)  # a seed not used in training
        mask = to_tensor(env.action_mask()[None], torch.bool)

        def model_move(obs):
            with torch.no_grad():
                logits, _ = net(to_tensor(obs[None]), mask)
            return int(logits.argmax())  # best move, no randomness

        random_score = play(env, lambda obs: int(rng.choice(env.valid_actions)), episodes)
        trained_score = play(env, model_move, episodes)
        print(f"{name:<8}{random_score:>10.2f}{trained_score:>10.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
