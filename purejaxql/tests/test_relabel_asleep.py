"""Tests for scripts/relabel_asleep.py: an old file (the policy's actions in `action`) gets the executed actions."""

import importlib.util
import json
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
def relabel():
    return load("relabel_asleep")


def collect_into(collect, monkeypatch, path, episodes=2, seed_start=0):
    args = ["--out", str(path), "--eps", "1.0", "--seed-start", str(seed_start), "--max-steps", "400",
            "--num-episodes", str(episodes), "--num-envs", "2", "--no-require-diamond", "--no-stop-at-diamond"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", ["collect"] + args)
    collect.main()


@pytest.fixture()
def old_file(collect, monkeypatch, tmp_path):
    """A file in the layout before the relabeling: `action` holds the policy's choice, no `action_policy`."""
    path = tmp_path / "old.h5"
    collect_into(collect, monkeypatch, path)
    with h5py.File(path, "r+") as f:
        executed, policy = f["action"][:], f["action_policy"][:]
        del f["action_policy"]
        f["action"][:] = policy
    return path, executed, policy


def test_an_old_file_gets_the_executed_actions(collect, relabel, old_file):
    path, executed, policy = old_file
    assert (executed != policy)[~np.isnan(policy)].any()  # random play slept and pressed keys while asleep
    record = relabel.relabel(path, collect)
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["action"][:], executed)  # what the collector now writes itself
        np.testing.assert_array_equal(f["action_policy"][:], policy)
        assert f["action_policy"].maxshape == f["action"].maxshape  # a later collection can append
        assert json.loads(f.attrs["relabel"]) == record
        asleep = f["state"][:, collect.SLEEPING] > 0.5
    valid = ~np.isnan(policy[:, 0])
    assert record["asleep_steps"] == int((asleep & valid).sum())
    assert record["changed"] == int(((executed != policy)[:, 0] & valid).sum()) > 0
    assert record["changed"] <= record["asleep_steps"] and record["steps"] == int(valid.sum())
    assert sum(record["changed_by_policy_action"].values()) == record["changed"]
    assert "0" not in record["changed_by_policy_action"]  # a noop pressed while asleep is no change


def test_relabeling_again_changes_nothing(collect, relabel, old_file):
    path, executed, _ = old_file
    first = relabel.relabel(path, collect)
    second = relabel.relabel(path, collect)
    assert {k: v for k, v in first.items() if k != "written"} == {k: v for k, v in second.items() if k != "written"}
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f["action"][:], executed)


def test_a_relabeled_file_replays_and_takes_new_episodes(collect, relabel, old_file, monkeypatch):
    path, *_ = old_file
    relabel.relabel(path, collect)
    report = load("check_replay").check(path, num_episodes=2, collector=collect)
    assert report["replayable"] and report["actions"] == {"relabeled": True, "episodes_ok": 2}
    collect_into(collect, monkeypatch, path, episodes=1, seed_start=100)  # the collector appends both columns
    with h5py.File(path, "r") as f:
        assert len(f["ep_len"]) == 3 and f["action_policy"].shape == f["action"].shape


def test_an_old_file_takes_no_new_episodes_before_the_relabeling(collect, old_file, monkeypatch):
    path, *_ = old_file
    with pytest.raises(ValueError, match="action_policy"):
        collect_into(collect, monkeypatch, path, episodes=1, seed_start=100)


def test_a_file_without_the_state_column_is_refused(relabel, collect, tmp_path):
    path = tmp_path / "bare.h5"
    with h5py.File(path, "w") as f:
        f["action"] = np.zeros((3, 1), np.float32)
        f["ep_len"] = np.array([3])
    with pytest.raises(SystemExit, match="state"):
        relabel.relabel(path, collect)
