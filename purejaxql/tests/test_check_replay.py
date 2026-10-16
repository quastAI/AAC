"""Tests for scripts/check_replay.py: a collected file replays, and a changed column is found."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("craftax")
h5py = pytest.importorskip("h5py")

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def collect():
    return load("collect_craftax_episodes")


@pytest.fixture(scope="module")
def check_replay():
    return load("check_replay")


@pytest.fixture()
def episodes(collect, monkeypatch, tmp_path):
    path = tmp_path / "a.h5"
    args = ["--out", str(path), "--eps", "1.0", "--seed-start", "0", "--max-steps", "300", "--num-episodes", "3",
            "--num-envs", "3", "--no-require-diamond", "--no-stop-at-diamond"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["collect"] + args)
    collect.main()
    return path


def test_a_collected_file_replays(check_replay, collect, episodes):
    report = check_replay.check(episodes, num_episodes=3, collector=collect)
    assert report["replayable"] and len(report["episodes"]) == 3
    assert all(not e["failed"] for e in report["episodes"])
    assert {"full_state", "achievements", "position", "pixels", "reward"} <= set(report["columns"])
    assert all(c["episodes_ok"] == 3 for c in report["columns"].values())
    assert "REPLAYABLE" in check_replay.render(report)


def test_a_changed_state_byte_is_found_at_its_frame(check_replay, collect, episodes):
    with h5py.File(episodes, "r+") as f:
        o = int(f["ep_offset"][1])
        f["full_state"][o + 5, 100] ^= 1
    report = check_replay.check(episodes, num_episodes=3, collector=collect)
    assert not report["replayable"]
    bad = [e for e in report["episodes"] if e["failed"]]
    assert len(bad) == 1 and set(bad[0]["failed"]) == {"full_state"}
    assert bad[0]["failed"]["full_state"]["first_frame"] == 5
    assert "NOT REPLAYABLE" in check_replay.render(report)
    assert check_replay.main(["--data", str(episodes), "--num-episodes", "3"]) == 1


def test_a_float_difference_within_the_tolerance_is_no_failure(check_replay, collect, episodes):
    with h5py.File(episodes, "r+") as f:
        f["daylight"][3, 0] += 1e-8
        f["reward"][4] += 1e-8
    report = check_replay.check(episodes, num_episodes=3, collector=collect)
    assert report["replayable"] and report["columns"]["daylight"]["episodes_ok"] == 3


def test_the_sample_can_be_restricted_to_a_source(check_replay, collect, episodes):
    report = check_replay.check(episodes, num_episodes=5, source=2, collector=collect)  # random episodes
    assert len(report["episodes"]) == 3
    none = check_replay.check(episodes, num_episodes=5, source=0, collector=collect)
    assert none["episodes"] == [] and not none["replayable"]  # nothing replayed is no pass
    assert "No episode" in check_replay.render(none)
