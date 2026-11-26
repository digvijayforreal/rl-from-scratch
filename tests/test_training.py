import csv
import json

import pytest

from dashboard import storage
from experiments import train as train_module
from experiments.evaluate import load_model
from experiments.train import CONFIG, build_parser, config_from_args, run, train

ALL_GAMES = ["catch", "reach", "maze", "snake"]


def tiny_config(games=ALL_GAMES, updates=3):
    """Small enough to finish in a couple of seconds."""
    cfg = {**CONFIG, "games": games, "envs_per_game": 1, "rollout_steps": 16, "epochs": 1,
           "minibatches": 2}
    cfg["total_steps"] = updates * len(games) * cfg["rollout_steps"]
    return cfg


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Send run folders to a temporary directory instead of the real results/."""
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(storage, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def quiet(*_):
    pass


def test_train_plays_every_game_and_logs_scores():
    result = train(tiny_config(), log=quiet)
    assert not result.interrupted
    assert result.update == result.num_updates == 3
    row = result.history[-1]
    assert row["update"] == 3
    assert all(f"score_{g}" in row for g in ALL_GAMES)


def test_steps_per_game_scales_with_the_number_of_games():
    args = build_parser().parse_args(["--games", "catch", "snake", "--steps-per-game", "250000"])
    assert config_from_args(args)["total_steps"] == 500_000
    args = build_parser().parse_args(["--games", "catch", "catch", "maze", "--steps-per-game", "1000"])
    cfg = config_from_args(args)
    assert cfg["games"] == ["catch", "maze"] and cfg["total_steps"] == 2000  # repeats are dropped


def test_giving_both_budgets_is_an_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--steps", "1000", "--steps-per-game", "1000"])


def test_run_writes_a_dashboard_compatible_folder(results_dir):
    run_dir = run(tiny_config(), name="smoke", save_every=1, eval_episodes=5, log=quiet)
    assert run_dir == results_dir / "runs" / "smoke"
    for name in ("config.json", "meta.json", "log.csv", "model.pt", "evaluation.json"):
        assert (run_dir / name).is_file(), name
    assert not (run_dir / "latest.pt").exists()  # only kept for unfinished runs

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["id"] == "smoke" and meta["kind"] == "run" and meta["status"] == "finished"
    assert meta["games"] == ALL_GAMES and meta["update"] == meta["num_updates"] == 3
    assert set(meta["scores"]) == set(ALL_GAMES)

    rows = list(csv.DictReader(open(run_dir / "log.csv", newline="")))
    assert rows and "score_snake" in rows[0]

    # the dashboard's Run history and checkpoint list can see it
    assert [r["id"] for r in storage.list_runs()] == ["smoke"]
    assert "runs/smoke/model.pt" in [c["name"] for c in storage.list_checkpoints()]

    net, saved = load_model(run_dir / "model.pt")  # the same loader the tests and dashboard use
    assert saved["config"]["games"] == ALL_GAMES
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert set(evaluation["modes"]) == {"greedy", "sample"}


def test_run_never_overwrites_an_existing_run(results_dir):
    run(tiny_config(["catch"], 1), name="same", eval_episodes=0, log=quiet)
    with pytest.raises(SystemExit):
        run(tiny_config(["catch"], 1), name="same", eval_episodes=0, log=quiet)


def test_run_rejects_unsafe_names(results_dir):
    with pytest.raises(ValueError):
        run(tiny_config(["catch"], 1), name="../escape", eval_episodes=0, log=quiet)


def test_ctrl_c_saves_a_checkpoint_that_can_be_continued(results_dir, monkeypatch):
    real_update = train_module.ppo_update
    calls = {"n": 0}

    def interrupt_on_second_update(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real_update(*args, **kwargs)

    monkeypatch.setattr(train_module, "ppo_update", interrupt_on_second_update)
    run_dir = run(tiny_config(["catch", "reach"], updates=5), name="first", eval_episodes=0, log=quiet)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "stopped" and meta["update"] == 1
    assert (run_dir / "latest.pt").is_file() and not (run_dir / "model.pt").exists()

    monkeypatch.setattr(train_module, "ppo_update", real_update)
    second = run(tiny_config(["catch", "reach", "maze"], updates=2), init=str(run_dir / "latest.pt"),
                 name="second", eval_episodes=0, log=quiet)
    meta = json.loads((second / "meta.json").read_text())
    assert meta["status"] == "finished" and meta["started_from"] == str(run_dir / "latest.pt")


def test_copy_to_writes_the_model_elsewhere(results_dir, tmp_path):
    target = tmp_path / "elsewhere" / "my_model.pt"
    run(tiny_config(["catch"], 1), name="copy", eval_episodes=0, copy_to=str(target), log=quiet)
    assert target.is_file()
    load_model(target)
