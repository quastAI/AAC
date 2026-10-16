"""Tests for scripts/time_expert.py (tiny network on CPU)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("craftax")
pytest.importorskip("wandb")

SCRIPT = Path(__file__).parents[1] / "scripts" / "time_expert.py"
TINY = ["NUM_ENVS=4", "NUM_STEPS=8", "NUM_MINIBATCHES=2", "NUM_EPOCHS=1", "HIDDEN_SIZE=16",
        "USE_OPTIMISTIC_RESETS=False", "TEST_NUM_ENVS=2", "TEST_NUM_STEPS=40"]  # fmt: skip


@pytest.fixture(scope="module")
def timing():
    spec = importlib.util.spec_from_file_location("time_expert", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_configuration_is_the_one_the_training_script_runs(timing):
    config = timing.load_config(["NUM_ENVS=64", "GAMMA=0.995"])
    assert config["NUM_ENVS"] == 64 and config["GAMMA"] == 0.995
    assert config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1" and config["NUM_STEPS"] == 128  # the file's values
    assert config["SAVE_PATH"] is None and config["WANDB_MODE"] == "disabled" and config["NUM_SEEDS"] == 1


def test_the_projection_adds_the_tests_to_the_updates(timing):
    (row,) = timing.project(t_update=2.0, t_test=60.0, steps_per_update=131072, totals=(1e9,), test_interval=0.05)
    assert row["updates"] == 7629
    assert row["test_hours"] == pytest.approx(60 * 20 / 3600)
    assert row["hours"] == pytest.approx((7629 * 2.0 + 60 * 20) / 3600)


def test_a_tiny_run_gives_positive_times_and_a_table(timing, capsys):
    timing.main([*TINY, "--updates", "2"])
    out = capsys.readouterr().out
    assert "per update" in out and "one greedy test" in out and "| 1e+09 |" in out
    assert "WARNING: no GPU" in out  # these tests run on the CPU
    line = next(x for x in out.splitlines() if x.startswith("per update"))
    assert float(line.split()[2]) > 0
