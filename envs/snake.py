"""Snake: eat the food without hitting a wall or yourself. Eat 5 to win.

The snake grows by one cell for every food it eats. Hitting the edge or your own
body ends the game with -1. Each food is worth +0.2 and every other step costs
-0.01, like Reach and Maze. The step cost matters: without it, doing nothing
risky scores 0, which beats chasing food at the risk of a -1 crash, and the agent
learns to stall instead of play.
"""

from collections import deque

from envs.base import AGENT, DOWN, GRID, ITEM, LEFT, MOVES, RIGHT, TRAIL, UP, MultiGameEnv

FOODS_TO_WIN = 5
FOOD_REWARD = 1.0 / FOODS_TO_WIN
CRASH_REWARD = -1.0
STEP_PENALTY = -0.01


class Snake(MultiGameEnv):
    name = "snake"
    valid_actions = [UP, RIGHT, DOWN, LEFT]
    max_steps = 100

    def _reset(self):
        cell = int(self.rng.integers(GRID * GRID))
        self.body = [divmod(cell, GRID)]  # head first
        self.eaten = 0
        self.food = self._place_food()

    def _place_food(self):
        taken = {r * GRID + c for r, c in self.body}
        free = [i for i in range(GRID * GRID) if i not in taken]
        return divmod(int(self.rng.choice(free)), GRID)

    def _step(self, action):
        dr, dc = MOVES[action]
        head = self.body[0]
        new = (head[0] + dr, head[1] + dc)

        if not (0 <= new[0] < GRID and 0 <= new[1] < GRID):
            self.last_event = "crash"
            return CRASH_REWARD, True

        eating = new == self.food
        # The tail cell moves away this turn, so stepping onto it is safe (unless we just ate).
        blocking = self.body if eating else self.body[:-1]
        if new in blocking:
            self.last_event = "crash"
            return CRASH_REWARD, True

        self.body.insert(0, new)
        if not eating:
            self.body.pop()
            return STEP_PENALTY, False

        self.eaten += 1
        if self.eaten >= FOODS_TO_WIN:
            self.last_event = "win"
            return FOOD_REWARD, True
        self.food = self._place_food()
        self.last_event = "food"
        return FOOD_REWARD, False

    def _draw(self, obs):
        obs[AGENT, self.body[0][0], self.body[0][1]] = 1
        for r, c in self.body[1:]:
            obs[TRAIL, r, c] = 1
        obs[ITEM, self.food[0], self.food[1]] = 1

    def state(self):
        return {"body": [list(p) for p in self.body], "food": list(self.food), "eaten": self.eaten}


def expert_action(env):
    """A scripted player used to estimate the best possible score: walk the shortest safe
    path to the food (BFS around the body). If the food is walled in by the body, take any
    safe step instead."""
    head, food = env.body[0], env.food
    blocked = set(env.body[:-1])  # the tail moves away, so it is not an obstacle
    queue = deque([(head, None)])
    seen = {head}
    while queue:
        cell, first = queue.popleft()
        for action in env.valid_actions:
            dr, dc = MOVES[action]
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID) or nxt in seen or nxt in blocked:
                continue
            step = first if first is not None else action
            if nxt == food:
                return step
            seen.add(nxt)
            queue.append((nxt, step))
    for action in env.valid_actions:  # no path to the food: just survive
        dr, dc = MOVES[action]
        nxt = (head[0] + dr, head[1] + dc)
        if 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID and nxt not in blocked:
            return action
    return env.valid_actions[0]
