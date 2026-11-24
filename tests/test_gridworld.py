import pytest

from envs.gridworld import GridWorld

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

# A shortest path from (0, 0) to (4, 4) on the default layout: 8 moves.
SHORTEST_PATH = [DOWN, DOWN, DOWN, DOWN, RIGHT, RIGHT, RIGHT, RIGHT]


def test_reset_puts_agent_at_start():
    env = GridWorld()
    assert env.reset() == (0, 0)


def test_wall_blocks_movement():
    env = GridWorld()
    env.reset()
    env.step(DOWN)                    # (0,0) -> (1,0)
    obs, *_ = env.step(RIGHT)         # (1,1) is a wall
    assert obs == (1, 0)


def test_edge_blocks_movement():
    env = GridWorld()
    env.reset()
    assert env.step(UP)[0] == (0, 0)
    assert env.step(LEFT)[0] == (0, 0)


def test_reaching_goal_terminates_with_reward():
    env = GridWorld()
    env.reset()
    for action in SHORTEST_PATH[:-1]:
        _, reward, terminated, truncated, _ = env.step(action)
        assert reward == -0.01 and not terminated and not truncated
    obs, reward, terminated, truncated, _ = env.step(SHORTEST_PATH[-1])
    assert obs == (4, 4)
    assert reward == 1.0
    assert terminated and not truncated


def test_shortest_path_total_reward():
    env = GridWorld()
    env.reset()
    total = sum(env.step(a)[1] for a in SHORTEST_PATH)
    assert total == pytest.approx(7 * -0.01 + 1.0)  # 0.93


def test_time_limit_truncates():
    env = GridWorld(max_steps=10)
    env.reset()
    for _ in range(9):
        _, _, terminated, truncated, _ = env.step(UP)
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(UP)
    assert truncated and not terminated


def test_step_after_done_raises():
    env = GridWorld(max_steps=1)
    env.reset()
    env.step(UP)
    with pytest.raises(RuntimeError):
        env.step(UP)


def test_invalid_action_raises():
    env = GridWorld()
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)


def test_bad_layout_raises():
    with pytest.raises(ValueError):
        GridWorld(goal=(1, 1))        # goal inside a wall
    with pytest.raises(ValueError):
        GridWorld(start=(9, 9))       # start off the grid
