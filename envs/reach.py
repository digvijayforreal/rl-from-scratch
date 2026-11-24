"""Reach: walk to the goal. Start and goal are random every episode."""

from envs.base import AGENT, DOWN, GOAL, GRID, LEFT, MOVES, RIGHT, UP, MultiGameEnv


class Reach(MultiGameEnv):
    name = "reach"
    valid_actions = [UP, RIGHT, DOWN, LEFT]
    max_steps = 50

    def _reset(self):
        # Pick two different cells: one for the agent, one for the goal.
        a, g = self.rng.choice(GRID * GRID, size=2, replace=False)
        self.agent = divmod(int(a), GRID)
        self.goal = divmod(int(g), GRID)

    def _step(self, action):
        dr, dc = MOVES[action]
        row = min(max(self.agent[0] + dr, 0), GRID - 1)  # stay on the board
        col = min(max(self.agent[1] + dc, 0), GRID - 1)
        self.agent = (row, col)
        if self.agent == self.goal:
            self.last_event = "goal"
            return 1.0, True
        return -0.01, False

    def _draw(self, obs):
        obs[AGENT, self.agent[0], self.agent[1]] = 1
        obs[GOAL, self.goal[0], self.goal[1]] = 1

    def state(self):
        return {"agent": list(self.agent), "goal": list(self.goal)}
