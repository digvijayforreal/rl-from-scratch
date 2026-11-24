"""A small, configurable gridworld for reinforcement learning.

Default 5x5 layout (S = start, G = goal, # = wall):

    S . . . .
    . # # . .
    . . # . .
    . # . . #
    . . . . G

Actions: 0 = up, 1 = right, 2 = down, 3 = left
Observation: the agent's (row, col) position as a tuple.
Reward: +1 on reaching the goal, -0.01 for every other step.
"""

import random

# Action index -> (row change, col change)
ACTIONS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}
ACTION_KEYS = {"w": 0, "d": 1, "s": 2, "a": 3}

DEFAULT_WALLS = {(1, 1), (1, 2), (2, 2), (3, 1), (3, 4)}


class GridWorld:
    def __init__(
        self,
        size=5,
        start=(0, 0),
        goal=(4, 4),
        walls=None,
        max_steps=100,
        goal_reward=1.0,
        step_penalty=-0.01,
    ):
        self.size = size
        self.start = start
        self.goal = goal
        self.walls = set(DEFAULT_WALLS if walls is None else walls)
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty

        # Catch bad layouts early instead of training on a broken game.
        for name, cell in (("start", start), ("goal", goal)):
            if not self._in_bounds(cell):
                raise ValueError(f"{name} {cell} is outside the grid")
            if cell in self.walls:
                raise ValueError(f"{name} {cell} is inside a wall")

        self.num_actions = len(ACTIONS)
        self.agent = None
        self.steps = 0
        self.done = True  # must call reset() before step()
        self.rng = random.Random()

    def _in_bounds(self, cell):
        row, col = cell
        return 0 <= row < self.size and 0 <= col < self.size

    def reset(self, seed=None):
        """Put the agent back at the start. Returns the first observation."""
        if seed is not None:
            # Not used yet (the game is deterministic), but kept so later
            # versions with random layouts are reproducible from a seed.
            self.rng.seed(seed)
        self.agent = self.start
        self.steps = 0
        self.done = False
        return self.agent

    def step(self, action):
        """Apply one action. Returns (obs, reward, terminated, truncated, info)."""
        if self.done:
            raise RuntimeError("Episode is over. Call reset() first.")
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}, expected 0-3")

        # Work out the new cell first, then only move if it is legal.
        drow, dcol = ACTIONS[action]
        new_cell = (self.agent[0] + drow, self.agent[1] + dcol)
        if self._in_bounds(new_cell) and new_cell not in self.walls:
            self.agent = new_cell

        self.steps += 1

        terminated = self.agent == self.goal  # reached the goal: true end
        truncated = (not terminated) and self.steps >= self.max_steps  # time limit
        reward = self.goal_reward if terminated else self.step_penalty

        self.done = terminated or truncated
        return self.agent, reward, terminated, truncated, {"steps": self.steps}

    def render(self):
        """Print the grid in ASCII."""
        for row in range(self.size):
            line = []
            for col in range(self.size):
                cell = (row, col)
                if cell == self.agent:
                    line.append("A")
                elif cell == self.goal:
                    line.append("G")
                elif cell in self.walls:
                    line.append("#")
                else:
                    line.append(".")
            print(" ".join(line))
        print()


def run_random_agent(env, episodes=1000, seed=0):
    """Baseline: average return and goal rate of an agent acting randomly."""
    rng = random.Random(seed)
    total_return, reached = 0.0, 0
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, reward, terminated, truncated, _ = env.step(rng.randrange(env.num_actions))
            total_return += reward
            done = terminated or truncated
        reached += terminated
    return total_return / episodes, reached / episodes


def play():
    """Play the game with w/a/s/d. Type q to quit."""
    env = GridWorld()
    env.reset()
    total = 0.0
    while True:
        env.render()
        key = input("Move (w/a/s/d, q to quit): ").strip().lower()
        if key == "q":
            break
        if key not in ACTION_KEYS:
            print("Use w, a, s or d.")
            continue
        _, reward, terminated, truncated, _ = env.step(ACTION_KEYS[key])
        total += reward
        if terminated or truncated:
            env.render()
            print("Reached the goal!" if terminated else "Out of time.")
            print(f"Total reward: {total:.2f}")
            break


if __name__ == "__main__":
    import sys

    if "--play" in sys.argv:
        play()
    else:
        env = GridWorld()
        env.reset()
        env.render()
        avg_return, goal_rate = run_random_agent(env)
        print(f"Random agent over 1000 episodes: "
              f"avg return {avg_return:.3f}, reached goal {goal_rate:.1%}")
