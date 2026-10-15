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
    flags = {"num_episodes": 4, **flags}
    args = ["--out", str(out), "--eps", "1.0", "--seed-start", "0", "--max-steps", "500"]  # fmt: skip
    for key, value in flags.items():
        if value is None:
            continue
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

    # `dead` marks the last frame of an episode that ended in death, and nothing else.
    dead = a["dead"][:]
    assert dead.shape == (n,) and dead.dtype == np.uint8
    assert dead[:].sum() == dead[last].sum()  # only ever on a last frame
    health = a["vitals"][:, 0]
    assert ((health[last] > 0) | (dead[last] == 1)).all()  # no health left means dead
    assert (dead[last] == 1).any()  # random play dies within 500 steps

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


def test_append_after_an_interrupted_run(collect, monkeypatch, tmp_path):
    # a kill between two column writes leaves rows behind that no finished
    # episode covers; the next run must not misalign the columns
    kwargs = dict(num_envs=2, no_require_diamond=True, no_stop_at_diamond=True)
    path = tmp_path / "e.h5"
    run(collect, monkeypatch, path, num_episodes=2, **kwargs).close()
    with h5py.File(path, "a") as f:
        rows = f["pixels"].shape[0]
        f["pixels"].resize(rows + 13, axis=0)  # orphans in some columns only
        f["action"].resize(rows + 13, axis=0)
    f = run(collect, monkeypatch, path, num_episodes=2, seed_start=100, **kwargs)
    total = int(f["ep_offset"][-1] + f["ep_len"][-1])
    assert all(f[key].shape[0] == total for key in f if not key.startswith("ep_o") and key != "ep_len")
    offsets = f["ep_offset"][:]
    assert f["seed"][offsets].tolist() == [0, 1, 100, 101]
    assert f["ep_idx"][offsets].tolist() == [0, 1, 2, 3]
    # the rows of every episode belong to its seed
    for offset, length in zip(offsets, f["ep_len"][:]):
        assert len(set(f["seed"][offset : offset + length].tolist())) == 1


def test_replay_check_sees_a_diverging_path_without_reward_change(collect):
    length = 5
    rendered = dict(
        reward=np.zeros(length + 1, np.float32),
        achievements=np.zeros((length + 1, 22), np.uint8),
        position=np.tile(np.arange(length + 1, dtype=np.int16)[:, None], (1, 2)),
        vitals=np.full((length + 1, 4), 9, np.int8),
    )
    rollout = dict(
        rewards=np.zeros((length, 1), np.float32),
        ach_counts=np.zeros((length, 1), np.int8),
        positions=rendered["position"][1:][:, None, :].copy(),
        health=np.full((length, 1), 9, np.int8),
    )
    collect.check_replay(7, length, rendered, rollout, 0)  # the same run: fine
    for key, change in (("positions", 1), ("health", 8)):
        broken = {k: v.copy() for k, v in rollout.items()}
        broken[key][3] = change  # rewards and achievements still agree
        with pytest.raises(RuntimeError, match="not deterministic"):
            collect.check_replay(7, length, rendered, broken, 0)


def test_seed_end_stops_at_the_end_of_the_range(collect, monkeypatch, tmp_path, capsys):
    # a low diamond rate must not run into the seeds of another split
    kwargs = dict(num_envs=2, no_require_diamond=True, no_stop_at_diamond=True)
    f = run(collect, monkeypatch, tmp_path / "r.h5", seed_end=3, **kwargs)  # asks for 4 episodes
    assert f["seed"][f["ep_offset"][:]].tolist() == [0, 1, 2]
    assert "used up" in capsys.readouterr().out


def test_target_frames_stop_after_the_episode_that_reaches_them(collect, monkeypatch, tmp_path):
    kwargs = dict(num_envs=2, no_require_diamond=True, no_stop_at_diamond=True)
    f = run(collect, monkeypatch, tmp_path / "t.h5", num_episodes=None, target_frames=150, **kwargs)
    lengths = f["ep_len"][:]
    assert lengths.sum() >= 150  # the target is reached ...
    assert lengths[:-1].sum() < 150  # ... by the last episode and not before
    # without a cap on episodes it keeps going until the frames are there
    assert len(lengths) >= 2


def test_collection_needs_a_stopping_rule(collect, monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="num-episodes or --target-frames"):
        run(collect, monkeypatch, tmp_path / "n.h5", num_episodes=None, no_require_diamond=True)


def test_dead_flag_covers_health_and_lava(collect):
    import jax

    env = collect.make_craftax_env_from_name(collect.ENV_NAME, auto_reset=False)
    _, state = env.reset(jax.random.PRNGKey(0), env.default_params)
    dead = lambda st: int(collect.derived_columns(st)[-1])  # noqa: E731
    assert dead(state) == 0
    assert dead(state.replace(player_health=0)) == 1
    row, col = (int(x) for x in state.player_position)
    lava = state.map.at[row, col].set(collect.BlockType.LAVA.value)
    # lava kills at once and leaves the health as it was
    assert dead(state.replace(map=lava)) == 1
