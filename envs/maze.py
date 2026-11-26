"""Maze: walk to the goal around random walls. The layout changes every episode.

Like Reach, but with obstacles, so the agent has to go around them instead of
walking in a straight line. Every layout is checked so the goal can be reached.
"""

from collections import deque

import numpy as np

from envs.base import AGENT, DOWN, GOAL, GRID, LEFT, MOVES, RIGHT, UP, WALL, MultiGameEnv

WALL_DENSITY = 0.18   # share of cells that are walls
MAX_TRIES = 200       # layouts to try before falling back to an empty board


def shortest_path_length(walls, start, goal):
    """Fewest moves from start to goal around the walls, or None if unreachable (BFS)."""
    if start == goal:
        return 0
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), dist = queue.popleft()
        for dr, dc in MOVES.values():
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
                continue
            if walls[nxt] or nxt in seen:
                continue
            if nxt == goal:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


class Maze(MultiGameEnv):
    name = "maze"
    valid_actions = [UP, RIGHT, DOWN, LEFT]
    max_steps = 60

    def _reset(self):
        for _ in range(MAX_TRIES):
            walls = self.rng.random((GRID, GRID)) < WALL_DENSITY
            free = np.flatnonzero(~walls.ravel())
            if len(free) < 2:
                continue
            a, g = self.rng.choice(free, size=2, replace=False)
            agent, goal = divmod(int(a), GRID), divmod(int(g), GRID)
            if shortest_path_length(walls, agent, goal) is not None:
                break
        else:  # extremely unlikely: use an empty board so the episode is still playable
            walls = np.zeros((GRID, GRID), dtype=bool)
            a, g = self.rng.choice(GRID * GRID, size=2, replace=False)
            agent, goal = divmod(int(a), GRID), divmod(int(g), GRID)
        self.walls, self.agent, self.goal = walls, agent, goal

    def _step(self, action):
        dr, dc = MOVES[action]
        new = (self.agent[0] + dr, self.agent[1] + dc)
        inside = 0 <= new[0] < GRID and 0 <= new[1] < GRID
        if inside and not self.walls[new]:  # bumping a wall or the edge leaves you in place
            self.agent = new
        if self.agent == self.goal:
            self.last_event = "goal"
            return 1.0, True
        return -0.01, False

    def _draw(self, obs):
        obs[WALL][self.walls] = 1
        obs[AGENT, self.agent[0], self.agent[1]] = 1
        obs[GOAL, self.goal[0], self.goal[1]] = 1

    def state(self):
        return {"agent": list(self.agent), "goal": list(self.goal),
                "walls": [[int(r), int(c)] for r, c in np.argwhere(self.walls)]}
