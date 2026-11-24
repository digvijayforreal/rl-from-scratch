"""Catch: a ball falls from the top. Move the paddle to catch it."""

from envs.base import AGENT, GRID, ITEM, LEFT, NOOP, RIGHT, MultiGameEnv


class Catch(MultiGameEnv):
    name = "catch"
    valid_actions = [NOOP, RIGHT, LEFT]
    max_steps = GRID  # the ball always lands before this

    def _reset(self):
        self.ball = [0, int(self.rng.integers(GRID))]  # [row, col]
        self.paddle = int(self.rng.integers(GRID))      # column on the bottom row

    def _step(self, action):
        if action == LEFT:
            self.paddle = max(0, self.paddle - 1)
        elif action == RIGHT:
            self.paddle = min(GRID - 1, self.paddle + 1)
        self.ball[0] += 1  # ball falls one row
        if self.ball[0] == GRID - 1:  # ball reached the bottom
            caught = self.paddle == self.ball[1]
            self.last_event = "catch" if caught else "miss"
            return (1.0 if caught else -1.0), True
        return 0.0, False

    def _draw(self, obs):
        obs[AGENT, GRID - 1, self.paddle] = 1
        obs[ITEM, self.ball[0], self.ball[1]] = 1

    def state(self):
        return {"ball": list(self.ball), "paddle": self.paddle}
