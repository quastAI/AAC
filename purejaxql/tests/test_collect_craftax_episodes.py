"""Tests for scripts/collect_craftax_episodes.py (uniform-random policy, no checkpoint)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("craftax")
h5py = pytest.importorskip("h5py")

SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_craftax_episodes.py"


@pytest.fixture(scope="module")
def collect():
    spec = importlib.util.spec_from_file_location("collect_craftax_episodes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(collect, monkeypatch, out, **flags):
    args = ["--out", str(out), "--eps", "1.0", "--num-episodes", "4", "--seed-start", "0",
            "--max-steps", "500"]  # fmt: skip
    for key, value in flags.items():
        args += [f"--{key.replace('_', '-')}"] + ([] if value is True else [str(value)])
    monkeypatch.setattr(sys, "argv", ["collect"] + args)
    collect.main()
    return h5py.File(out, "r")


def test_layout_and_replayability(collect, monkeypatch, tmp_path):
    kwargs = dict(no_require_diamond=True, no_stop_at_diamond=True)
    a = run(collect, monkeypatch, tmp_path / "a.h5", num_envs=4, **kwargs)
    b = run(collect, monkeypatch, tmp_path / "b.h5", num_envs=2, **kwargs)

    n = int(a["ep_len"][:].sum())
    assert a["pixels"].shape == (n, 63, 63, 3) and a["pixels"].dtype == np.uint8
    assert a["state"].shape == (n, 1345)
    assert a["achievements"].shape == (n, 22)
    assert a["full_state"].shape == (n, 20952) and a["full_state"].dtype == np.uint8
    for name, shape in dict(inventory=(n, 12), vitals=(n, 4), facing=(n, 1), daylight=(n, 1),
                            position=(n, 2), mobs_in_view=(n, 4)).items():  # fmt: skip
        assert a[name].shape == shape, name
    # a new world starts with an empty inventory, full vitals, and the player on the map
    first = a["ep_offset"][:]
    assert (a["inventory"][first] == 0).all() and (a["vitals"][first] == 9).all()
    assert ((a["position"][:] >= 0) & (a["position"][:] < 64)).all()
    assert set(np.unique(a["facing"][:])) <= {1, 2, 3, 4}
    assert (a["mobs_in_view"][:] >= 0).all()
    assert list(a["ep_offset"][:]) == list(np.cumsum(np.r_[0, a["ep_len"][:-1]]))

    # One NaN action per episode, in its last frame; the others are valid indices.
    action = a["action"][:, 0]
    last = a["ep_offset"][:] + a["ep_len"][:] - 1
    assert np.isnan(action).sum() == len(last) and np.isnan(action[last]).all()
    assert set(np.unique(action[~np.isnan(action)])) <= set(range(17))

    # The same world seed gives the same episode regardless of the batch size.
    for key in a:
        np.testing.assert_array_equal(a[key][:], b[key][:], err_msg=key)


@pytest.mark.parametrize("tail", [0, 7])
def test_stop_at_achievement(collect, monkeypatch, tmp_path, tail):
    # Wood stands in for diamond: a random agent collects it within a few steps.
    monkeypatch.setattr(collect, "DIAMOND", collect.Achievement.COLLECT_WOOD.value)
    f = run(collect, monkeypatch, tmp_path / "c.h5", num_envs=8, tail_steps=tail)
    for offset, length in zip(f["ep_offset"][:], f["ep_len"][:]):
        flag = f["achievements"][offset : offset + length, collect.DIAMOND]
        assert flag.any(), "kept an episode without the achievement"
        assert length - 1 - int(np.argmax(flag)) == tail


def test_append_skips_used_seeds(collect, monkeypatch, tmp_path):
    kwargs = dict(num_envs=2, no_require_diamond=True, no_stop_at_diamond=True)
    run(collect, monkeypatch, tmp_path / "d.h5", **kwargs)
    f = run(collect, monkeypatch, tmp_path / "d.h5", **kwargs)  # same seed range again
    seeds = f["seed"][f["ep_offset"][:]]
    assert len(seeds) == 8 and len(set(seeds.tolist())) == 8
    assert sorted(set(f["ep_idx"][:].tolist())) == list(range(8))
