import json

import numpy as np
import pytest

from envs.base import DOWN, GRID, LEFT, NOOP, NUM_ACTIONS, NUM_CHANNELS, RIGHT, UP
from envs.maze import Maze, shortest_path_length
from envs.snake import FOOD_REWARD, FOODS_TO_WIN, STEP_PENALTY, Snake, expert_action
from experiments.train import GAMES

ALL = list(GAMES)


def play_random(env, seed):
    """Play one full episode with random valid moves. Returns (steps, total reward, last event)."""
    rng = np.random.default_rng(seed)
    env.reset()
    total, done = 0.0, False
    while not done:
        _, r, terminated, truncated = env.step(int(rng.choice(env.valid_actions)))
        total += r
        done = terminated or truncated
    return env.steps, total, env.last_event


# ---------- every game follows the shared format ----------

@pytest.mark.parametrize("name", ALL)
def test_observation_format(name):
    env = GAMES[name](seed=0)
    obs = env.reset()
    assert obs.shape == (NUM_CHANNELS, GRID, GRID)
    assert obs.dtype == np.float32
    assert set(np.unique(obs)) <= {0.0, 1.0}
    assert obs.sum() > 0


@pytest.mark.parametrize("name", ALL)
def test_action_mask_matches_valid_actions(name):
    env = GAMES[name](seed=0)
    mask = env.action_mask()
    assert mask.shape == (NUM_ACTIONS,)
    assert set(np.flatnonzero(mask)) == set(env.valid_actions)


@pytest.mark.parametrize("name", ALL)
def test_invalid_action_raises(name):
    env = GAMES[name](seed=0)
    env.reset()
    invalid = next(a for a in range(NUM_ACTIONS) if a not in env.valid_actions)
    with pytest.raises(ValueError):
        env.step(invalid)


@pytest.mark.parametrize("name", ALL)
def test_same_seed_gives_same_game(name):
    runs = []
    for _ in range(2):
        env = GAMES[name](seed=7)
        obs = [env.reset().copy()]
        rng = np.random.default_rng(3)
        done = False
        while not done:
            o, r, t, tr = env.step(int(rng.choice(env.valid_actions)))
            obs.append(o.copy())
            done = t or tr
        runs.append(np.stack(obs))
    assert runs[0].shape == runs[1].shape
    assert np.array_equal(runs[0], runs[1])


@pytest.mark.parametrize("name", ALL)
def test_state_can_be_sent_to_the_dashboard(name):
    env = GAMES[name](seed=1)
    env.reset()
    rng = np.random.default_rng(1)
    done = False
    while not done:
        json.dumps(env.state())  # must be plain numbers and lists
        _, _, t, tr = env.step(int(rng.choice(env.valid_actions)))
        done = t or tr


@pytest.mark.parametrize("name", ALL)
def test_episodes_always_end_within_the_time_limit(name):
    env = GAMES[name](seed=2)
    for i in range(20):
        steps, _, event = play_random(env, i)
        assert steps <= env.max_steps
        assert event is not None


@pytest.mark.parametrize("name", ALL)
def test_step_after_done_raises(name):
    env = GAMES[name](seed=0)
    play_random(env, 0)
    with pytest.raises(RuntimeError):
        env.step(env.valid_actions[0])


# ---------- Maze ----------

def test_maze_goal_is_always_reachable():
    env = Maze(seed=0)
    for _ in range(300):
        env.reset()
        assert not env.walls[env.agent] and not env.walls[env.goal]
        assert env.agent != env.goal
        assert shortest_path_length(env.walls, env.agent, env.goal) is not None


def test_maze_layout_changes_between_episodes():
    env = Maze(seed=0)
    layouts = set()
    for _ in range(10):
        env.reset()
        layouts.add(env.walls.tobytes())
    assert len(layouts) > 1


def test_maze_observation_matches_state():
    env = Maze(seed=3)
    obs = env.reset()
    from envs.base import AGENT, GOAL, WALL
    assert obs[WALL].sum() == len(env.state()["walls"])
    assert obs[AGENT][env.agent] == 1 and obs[AGENT].sum() == 1
    assert obs[GOAL][env.goal] == 1 and obs[GOAL].sum() == 1


def test_maze_walls_and_edges_block_movement():
    env = Maze(seed=0)
    env.reset()
    env.walls = np.zeros((GRID, GRID), dtype=bool)
    env.walls[5, 6] = True
    env.agent, env.goal = (5, 5), (0, 0)
    _, reward, terminated, _ = env.step(RIGHT)
    assert env.agent == (5, 5)               # the wall keeps the agent in place
    assert reward == pytest.approx(-0.01) and not terminated
    env.agent = (0, 3)
    env.step(UP)
    assert env.agent == (0, 3)               # the edge blocks too


def test_maze_reaching_goal_ends_the_episode_with_plus_one():
    env = Maze(seed=0)
    env.reset()
    env.walls = np.zeros((GRID, GRID), dtype=bool)
    env.agent, env.goal = (4, 4), (4, 5)
    _, reward, terminated, truncated = env.step(RIGHT)
    assert reward == 1.0 and terminated and not truncated
    assert env.last_event == "goal"


def test_maze_normal_step_costs_a_little():
    env = Maze(seed=0)
    env.reset()
    env.walls = np.zeros((GRID, GRID), dtype=bool)
    env.agent, env.goal = (4, 4), (9, 9)
    _, reward, terminated, _ = env.step(RIGHT)
    assert reward == pytest.approx(-0.01) and not terminated


# ---------- Snake ----------

def make_snake(body, food, eaten=0):
    env = Snake(seed=0)
    env.reset()
    env.body, env.food, env.eaten = list(body), food, eaten
    return env


def test_snake_eating_food_grows_and_pays():
    env = make_snake([(5, 5)], (5, 6))
    _, reward, terminated, _ = env.step(RIGHT)
    assert reward == pytest.approx(FOOD_REWARD) and not terminated
    assert env.body[0] == (5, 6) and len(env.body) == 2
    assert env.eaten == 1 and env.last_event == "food"
    assert env.food not in env.body  # new food is not placed inside the snake


def test_snake_moving_without_eating_keeps_its_length():
    env = make_snake([(5, 5), (5, 4), (5, 3)], (0, 0))
    _, reward, terminated, _ = env.step(RIGHT)
    assert reward == pytest.approx(STEP_PENALTY) and not terminated
    assert env.body == [(5, 6), (5, 5), (5, 4)]


def test_snake_hitting_the_wall_loses():
    env = make_snake([(0, 3)], (9, 9))
    _, reward, terminated, _ = env.step(UP)
    assert reward == -1.0 and terminated and env.last_event == "crash"


def test_snake_hitting_itself_loses():
    env = make_snake([(5, 5), (5, 6), (6, 6), (6, 5), (6, 4)], (0, 0))
    _, reward, terminated, _ = env.step(RIGHT)  # into its own neck
    assert reward == -1.0 and terminated and env.last_event == "crash"


def test_snake_can_step_onto_the_tail_cell():
    env = make_snake([(5, 5), (5, 6), (6, 6), (6, 5)], (0, 0))  # tail at (6, 5)
    _, reward, terminated, _ = env.step(DOWN)
    assert reward == pytest.approx(STEP_PENALTY) and not terminated


def test_snake_wins_after_enough_food():
    env = make_snake([(5, 5)], (5, 6), eaten=FOODS_TO_WIN - 1)
    _, reward, terminated, _ = env.step(RIGHT)
    assert reward == pytest.approx(FOOD_REWARD) and terminated and env.last_event == "win"


def test_snake_observation_has_head_body_and_food():
    from envs.base import AGENT, ITEM, TRAIL
    env = make_snake([(5, 5), (5, 4), (5, 3)], (2, 2))
    obs = env.observe()
    assert obs[AGENT].sum() == 1 and obs[AGENT][5, 5] == 1
    assert obs[TRAIL].sum() == 2 and obs[TRAIL][5, 4] == 1 and obs[TRAIL][5, 3] == 1
    assert obs[ITEM].sum() == 1 and obs[ITEM][2, 2] == 1


def test_snake_expert_wins_almost_every_game():
    env = Snake(seed=11)
    wins, total = 0, 100
    for _ in range(total):
        env.reset()
        done = False
        while not done:
            _, _, t, tr = env.step(expert_action(env))
            done = t or tr
        wins += env.last_event == "win"
    assert wins >= 95
