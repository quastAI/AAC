"""Tests for scripts/check_replay.py: a collected file replays, and a changed column is found."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
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


def test_the_actions_must_follow_the_rule(check_replay, collect, episodes):
    report = check_replay.check(episodes, num_episodes=3, collector=collect)
    assert report["actions"] == {"relabeled": True, "episodes_ok": 3}
    with h5py.File(episodes, "r+") as f:  # a policy action that `action` no longer follows (an awake step)
        o = int(f["ep_offset"][0])
        t = int(np.flatnonzero(f["state"][o : o + 50, collect.SLEEPING] < 0.5)[0])
        f["action_policy"][o + t, 0] = (f["action"][o + t, 0] + 1) % 17
    report = check_replay.check(episodes, num_episodes=3, collector=collect)
    assert report["actions"]["episodes_ok"] == 2 and not report["replayable"]
    assert all(c["episodes_ok"] == 3 for c in report["columns"].values())  # the replay itself is fine
    assert "in 2 of 3 episodes" in check_replay.render(report)


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
    assert bad[0]["failed"]["full_state"]["leaves"]  # and it names the leaf that differs


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


def leaf(check_replay, collect, suffix):
    """(name, dtype, byte offset, bytes) of the packed-state leaf whose name ends with `suffix`."""
    (found,) = [x for x in check_replay.state_layout(collect) if x[0].endswith(suffix)]
    return found


def shift_leaf(path, episode, frame, spec, delta):
    """Add `delta` to a scalar leaf of one stored frame of the full state."""
    _, dtype, offset, size = spec
    with h5py.File(path, "r+") as f:
        o = int(f["ep_offset"][episode])
        value = np.frombuffer(f["full_state"][o + frame, offset : offset + size].tobytes(), dtype)[0]
        f["full_state"][o + frame, offset : offset + size] = np.frombuffer(
            np.array(value + delta, dtype).tobytes(), np.uint8
        )


def test_the_layout_covers_the_whole_packed_state(check_replay, collect, episodes):
    layout = check_replay.state_layout(collect)
    with h5py.File(episodes, "r") as f:
        assert layout[-1][2] + layout[-1][3] == f["full_state"].shape[1]
    names = [x[0] for x in layout]
    assert any(n.endswith("light_level") for n in names) and any(n.endswith("timestep") for n in names)


def test_a_last_bits_difference_of_the_light_level_is_no_failure(check_replay, collect, episodes):
    # A GPU and a CPU differ in the last bits of the light level (about 2e-7), which changes the state bytes from the
    # second frame on without changing the game: seen in the first collection on an A100.
    shift_leaf(episodes, 0, 5, leaf(check_replay, collect, "light_level"), 2e-7)
    report = check_replay.check(episodes, num_episodes=3, collector=collect)
    assert report["replayable"]
    light = next(v for k, v in report["leaves"].items() if k.endswith("light_level"))
    assert light["failed"] == 0 and 1e-7 < light["max_abs_diff"] < 1e-6
    assert "no failure" in check_replay.render(report) and "REPLAYABLE" in check_replay.render(report)


def test_a_larger_light_level_difference_fails_and_names_the_leaf(check_replay, collect, episodes):
    shift_leaf(episodes, 0, 5, leaf(check_replay, collect, "light_level"), 1e-3)
    report = check_replay.check(episodes, num_episodes=3, collector=collect)
    assert not report["replayable"]
    assert "light_level" in check_replay.render(report) and "NOT REPLAYABLE" in check_replay.render(report)


def test_an_integer_leaf_must_be_identical(check_replay, collect, episodes):
    shift_leaf(episodes, 1, 7, leaf(check_replay, collect, "timestep"), 1)
    report = check_replay.check(episodes, num_episodes=3, collector=collect)
    assert not report["replayable"]
    failing = [e for e in report["episodes"] if e["failed"]]
    assert len(failing) == 1 and failing[0]["failed"]["full_state"]["first_frame"] == 7
    assert "must be identical" in check_replay.render(report) and "timestep" in check_replay.render(report)
