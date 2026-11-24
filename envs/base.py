"""Shared format for every game: an 8 x 10 x 10 board and the same 6 actions."""

import numpy as np

GRID = 10

# Each channel is a 10x10 layer of 0s and 1s showing one kind of object.
NUM_CHANNELS = 8
AGENT, WALL, GOAL, DANGER, ITEM, TRAIL, AGENT_SHOT, ENEMY_SHOT = range(NUM_CHANNELS)
CHANNEL_NAMES = ["agent", "wall", "goal", "danger", "item", "trail", "agent_shot", "enemy_shot"]

# One action set shared by all games.
NUM_ACTIONS = 6
NOOP, UP, RIGHT, DOWN, LEFT, FIRE = range(NUM_ACTIONS)
ACTION_NAMES = ["noop", "up", "right", "down", "left", "fire"]
MOVES = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

SYMBOLS = {AGENT: "A", WALL: "#", GOAL: "G", DANGER: "X",
           ITEM: "o", TRAIL: "=", AGENT_SHOT: "|", ENEMY_SHOT: ":"}


class MultiGameEnv:
    """Base class. Each game sets name, valid_actions, max_steps and
    writes _reset, _step, _draw and state()."""

    name = "base"
    valid_actions = list(range(NUM_ACTIONS))
    max_steps = 100

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.done = True  # must call reset() first
        self.last_event = None  # e.g. "catch", "miss", "goal" (used by the dashboard)

    def reset(self):
        self.steps = 0
        self.done = False
        self.last_event = None
        self._reset()
        return self.observe()

    def step(self, action):
        """Returns (obs, reward, terminated, truncated)."""
        if self.done:
            raise RuntimeError("Episode is over. Call reset() first.")
        if action not in self.valid_actions:
            raise ValueError(f"{self.name}: action {action} is not allowed")
        self.last_event = None
        reward, terminated = self._step(action)
        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated:
            self.last_event = "timeout"
        self.done = terminated or truncated
        return self.observe(), reward, terminated, truncated

    def action_mask(self):
        """True for the actions this game allows."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[self.valid_actions] = True
        return mask

    def observe(self):
        obs = np.zeros((NUM_CHANNELS, GRID, GRID), dtype=np.float32)
        self._draw(obs)
        return obs

    def render(self):
        obs = self.observe()
        for r in range(GRID):
            row = []
            for c in range(GRID):
                on = np.flatnonzero(obs[:, r, c])
                row.append(SYMBOLS[int(on[0])] if len(on) else ".")
            print(" ".join(row))
        print()

    # Each game fills these in.
    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        """Update the game and return (reward, terminated)."""
        raise NotImplementedError

    def _draw(self, obs):
        """Put 1s on the board for this game's objects."""
        raise NotImplementedError

    def state(self):
        """Object positions as a dict, so the dashboard can draw nice graphics."""
        raise NotImplementedError
