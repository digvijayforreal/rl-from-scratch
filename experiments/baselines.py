"""Reference scores: random play (zero skill) and perfect play."""

from functools import lru_cache

import numpy as np

from envs.base import GRID
from experiments.train import GAMES

# Episode endings that count as a win.
SUCCESS_EVENTS = {"catch", "goal"}


@lru_cache(maxsize=None)
def random_baseline(name, episodes=400, seed=4242):
    """Average return of random valid moves."""
    env = GAMES[name](seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, r, term, trunc = env.step(int(rng.choice(env.valid_actions)))
            total += r
            done = term or trunc
    return total / episodes


@lru_cache(maxsize=None)
def optimal_score(name):
    """Best possible average return, worked out exactly."""
    if name == "catch":
        return 1.0  # the paddle can always reach the ball in time
    if name == "reach":
        # Shortest path = Manhattan distance d; return = 1 - 0.01 * (d - 1), averaged over all start/goal pairs.
        cells = [(r, c) for r in range(GRID) for c in range(GRID)]
        scores = [1 - 0.01 * (abs(a[0] - b[0]) + abs(a[1] - b[1]) - 1)
                  for a in cells for b in cells if a != b]
        return float(np.mean(scores))
    return None
