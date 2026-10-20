"""dataset_section on synthetic files with known answers."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from dataset_section import actions, change, data, episodes, figures, spectral  # noqa: E402
from dataset_section import compute as numbers  # noqa: E402

MOVES = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def make_world_episode(rng, n, size=24, flip=0.01, static=False):
    """Random walk of the player over a world whose blocks and mobs change; the view (7x9) is cut from the world."""
    world = rng.integers(0, 17, (size, size))
    mobs = rng.integers(1, 16, (size, size)) * (rng.random((size, size)) < 0.1)
    pos = np.array([size // 2, size // 2])
    worlds, mob_worlds, positions = [], [], []
    for t in range(n):
        if t:
            pos = np.clip(pos + MOVES[rng.integers(5)], [3, 4], [size - 4, size - 5])
            if not static:
                world = np.where(rng.random((size, size)) < flip, rng.integers(0, 17, (size, size)), world)
                mobs = np.where(rng.random((size, size)) < flip, rng.integers(0, 16, (size, size)), mobs)
        worlds.append(world.copy())
        mob_worlds.append(mobs.copy())
        positions.append(pos.copy())
    return np.array(worlds), np.array(mob_worlds), np.array(positions)


def cut_view(plane, pos):
    return plane[:, pos[0] - 3 : pos[0] + 4, pos[1] - 4 : pos[1] + 5]


def view_state(worlds, mob_worlds, positions):
    """The `state` column rows: block one-hot (17) and mob bits (4) per tile of the view, then zeros."""
    n = len(worlds)
    state = np.zeros((n, data.STATE_DIM), np.float32)
    view = np.zeros((n, data.VIEW_ROWS, data.VIEW_COLS, data.VIEW_CHANNELS), np.float32)
    for t in range(n):
        block = worlds[t][positions[t, 0] - 3 : positions[t, 0] + 4, positions[t, 1] - 4 : positions[t, 1] + 5]
        mob = mob_worlds[t][positions[t, 0] - 3 : positions[t, 0] + 4, positions[t, 1] - 4 : positions[t, 1] + 5]
        view[t, :, :, :17] = np.eye(17, dtype=np.float32)[block]
        for bit in range(4):
            view[t, :, :, 17 + bit] = (mob >> bit) & 1
    state[:, : data.VIEW_DIM] = view.reshape(n, -1)
    return state


def write(path, specs, state=True, seed_base=0):
    """`specs`: dicts with n and optional diamond (frame), static, source, columns to override."""
    rng = np.random.default_rng(0)
    cols = {k: [] for k in ("action", "achievements", "inventory", "vitals", "facing", "daylight", "position", "mobs_in_view",
                            "source", "seed", "reward", "policy_updates", "dead", "state")}  # fmt: skip
    lengths, offsets, offset = [], [], 0
    for e, spec in enumerate(specs):
        n = spec["n"]
        worlds, mob_worlds, positions = make_world_episode(rng, n, static=spec.get("static", False))
        action = rng.integers(0, 17, n).astype(np.float32)
        action[-1] = np.nan
        flags = np.zeros((n, 22), np.uint8)
        flags[min(10 + e, n - 1) :, 0] = 1
        if spec.get("diamond") is not None:
            flags[spec["diamond"] :, data.DIAMOND_INDEX] = 1
        inventory = np.zeros((n, 12), np.int16)
        inventory[:, 0] = np.arange(n) // 4
        vitals = (9 - np.cumsum(rng.random((n, 4)) < 0.1, axis=0)).astype(np.int8)
        cols["action"].append(action[:, None])
        cols["achievements"].append(flags)
        cols["inventory"].append(inventory)
        cols["vitals"].append(vitals)
        cols["facing"].append(rng.integers(1, 5, (n, 1)).astype(np.int8))
        cols["daylight"].append((0.02 * np.arange(n))[:, None].astype(np.float32))
        cols["position"].append(positions.astype(np.int16))
        cols["mobs_in_view"].append(rng.integers(0, 3, (n, 4)).astype(np.int8))
        cols["source"].append(np.full(n, spec.get("source", 0), np.int8))
        cols["seed"].append(np.full(n, seed_base + e, np.int64))
        cols["reward"].append(np.zeros(n, np.float32))
        cols["policy_updates"].append(np.full(n, -1, np.int32))
        cols["dead"].append(np.zeros(n, np.uint8))
        cols["state"].append(view_state(worlds, mob_worlds, positions) if state else np.zeros((n, 3), np.float32))
        lengths.append(n)
        offsets.append(offset)
        offset += n
    with h5py.File(path, "w") as f:
        for k, v in cols.items():
            f[k] = np.concatenate(v)
        f["ep_len"] = np.array(lengths, np.int32)
        f["ep_offset"] = np.array(offsets, np.int64)
        f.attrs["runs"] = json.dumps([{"finished": True}])
    return path


def brute_world(path, k):
    """Reference for the world-frame rows: compare the world maps of every pair on the cells that both views show."""
    with h5py.File(path, "r") as f:
        pos = f["position"][:].astype(int)
        lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
        state = f["state"][:, : data.VIEW_DIM].reshape(-1, data.VIEW_ROWS, data.VIEW_COLS, data.VIEW_CHANNELS)
    block = state[..., :17].argmax(-1)
    mob = (state[..., 17:] * (1 << np.arange(4))).sum(-1).astype(int)
    out = {n: [0, 0.0] for n in ("terrain", "mobs")}
    pairs = 0
    ego = [0, 0.0]
    for n, o in zip(lengths, offsets):
        for t in range(o, o + n - k):
            r0, c0, r1, c1 = *pos[t], *pos[t + k]
            pairs += 1
            tile_a, tile_b = block[t] * 16 + mob[t], block[t + k] * 16 + mob[t + k]
            ego[0] += int((tile_a != tile_b).any())
            ego[1] += float((tile_a != tile_b).mean())
            # world cells shown by both views: rows max(r0, r1) - 3 .. min(r0, r1) + 3, columns likewise with 4
            cells = [(r, c) for r in range(max(r0, r1) - 3, min(r0, r1) + 4) for c in range(max(c0, c1) - 4, min(c0, c1) + 5)]
            for name, plane in (("terrain", block), ("mobs", mob)):
                changed = [plane[t][r - r0 + 3, c - c0 + 4] != plane[t + k][r - r1 + 3, c - c1 + 4] for r, c in cells]
                out[name][0] += int(any(changed))
                out[name][1] += float(np.mean(changed))
    return {"ego_any": ego[0] / pairs, "ego_share": ego[1] / pairs} | {f"{n}_{s}": out[n][i] / pairs for n in out for i, s in enumerate(("any", "share"))}


@pytest.fixture(scope="module")
def world_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("world") / "w.h5"
    return write(path, [{"n": 70}, {"n": 45, "diamond": 30}, {"n": 60}])


def rows_by_name(matrix):
    return {r["name"]: r for r in matrix["rows"]}


def test_pairs_do_not_cross_episodes(tmp_path):
    d = data.Split(write(tmp_path / "a.h5", [{"n": 5}, {"n": 5}], state=False))
    assert change.pair_starts(d, 2).tolist() == [0, 1, 2, 5, 6, 7]
    assert len(change.pair_starts(d, 5)) == 0


def test_change_of_small_columns_has_known_shares(world_file):
    d = data.Split(world_file)
    m = change.change_matrix(d)
    rows = rows_by_name(m)
    lag = m["lags"].index
    # wood = t // 4: differs for a pair (t, t + k) when (t + k) // 4 != t // 4
    for k in (1, 2, 5):
        expected = np.mean([(t + k) // 4 != t // 4 for n in (70, 45, 60) for t in range(n - k)])
        assert rows["wood"]["share"][lag(k)] == pytest.approx(expected)
    assert rows["stone"]["share"][lag(1)] == 0  # never changes
    # daylight moves 0.02 per frame: unchanged at lag 1 and 2 (0.04 <= 0.05), changed from lag 3 on
    assert rows["daylight"]["share"][lag(2)] == 0 and rows["daylight"]["share"][lag(3)] == 1
    assert rows["daylight"]["half_lag"] == 3
    # lags beyond every episode have no pairs
    assert m["pairs"]["100"] == 0 and rows["wood"]["share"][lag(100)] is None
    # one achievement flag turns on in every episode (frame 10 + e) and the diamond in one: any new unlock
    unlock = rows["any new unlock"]["share"][lag(1)]
    assert 0 < unlock < 0.1


def test_the_view_rows_match_a_brute_force_comparison(world_file):
    d = data.Split(world_file)
    m = change.change_matrix(d)
    rows = rows_by_name(m)
    for k in (1, 2, 3, 5):
        expected = brute_world(world_file, k)
        for key, label in change.VIEW_LABELS.items():
            assert rows[label]["share"][m["lags"].index(k)] == pytest.approx(expected[key]), (k, key)


def test_world_frame_rows_ignore_the_walk_and_stop_at_the_overlap_limit(tmp_path):
    path = write(tmp_path / "static.h5", [{"n": 80, "static": True}, {"n": 80, "static": True}])
    m = change.change_matrix(data.Split(path))
    rows = rows_by_name(m)
    for k in (1, 2, 3, 5):
        i = m["lags"].index(k)
        assert rows["terrain (world), any tile"]["share"][i] == 0
        assert rows["mobs (world), share of tiles"]["share"][i] == 0
        assert rows["view, any tile"]["share"][i] > 0  # walking changes the egocentric view
    assert rows["terrain (world), any tile"]["share"][m["lags"].index(10)] is None
    assert rows["view, any tile"]["share"][m["lags"].index(10)] is not None
    assert rows["mean tiles shared by the two views"]["share"][m["lags"].index(1)] >= 18


def test_a_file_without_a_view_has_no_view_rows(tmp_path):
    path = write(tmp_path / "noview.h5", [{"n": 40}], state=False)
    m = change.change_matrix(data.Split(path))
    assert not any(r["group"] == "view" for r in m["rows"])


def test_length_time_to_diamond_achievements_and_actions(world_file):
    d = data.Split(world_file)
    length = episodes.length_numbers(d)
    assert length["lengths"] == [45, 60, 70] and length["quantiles"]["p50"] == 60 and length["frames"] == 175
    ttd = episodes.time_to_diamond(d)
    assert ttd["episodes"] == 1 and ttd["of"] == 3 and ttd["steps"] == [30]
    ach = episodes.achievement_numbers(d)
    assert ach["per_episode"]["histogram"][1] == 2 and ach["per_episode"]["histogram"][2] == 1
    wood = ach["ladder"]["collect_wood"]
    assert wood["share"] == 1 and wood["p50"] == 11 and wood["p10"] == pytest.approx(10.2)
    assert ach["ladder"]["collect_diamond"]["episodes"] == 1 and "p50" not in ach["ladder"]["place_table"]
    counts = actions.action_numbers(d)
    assert sum(counts["steps"].values()) == 175 - 3 and set(counts["steps"]) == set(data.ACTIONS)
    assert sum(counts["share"].values()) == pytest.approx(1)


def test_clip_coverage_follows_the_span_formula(world_file):
    d = data.Split(world_file)
    clips = episodes.clip_coverage(d, strides=(1, 10))
    s = clips["settings"]["context5_rollout5"]
    assert s["span"] == [10, 91]  # (5 + 5 - 1) * T + 1
    assert s["clips"][0] == sum(max(n - 10 + 1, 0) for n in (70, 45, 60))
    assert s["clips"][1] == 0 and s["episodes_long_enough"] == [3, 0]
    assert clips["settings"]["context1_rollout1"]["span"] == [2, 11]


def test_ks_statistic():
    assert numbers.ks_statistic([1, 2, 3], [1, 2, 3]) == 0
    assert numbers.ks_statistic([1, 2, 3], [10, 11]) == 1
    assert numbers.ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)


def test_compute_and_plot_end_to_end(tmp_path):
    pytest.importorskip("matplotlib")
    files = {
        "train": write(tmp_path / "train.h5", [{"n": 520, "diamond": 300}, {"n": 640, "diamond": 350}, {"n": 300, "source": 3}], seed_base=0),
        "val": write(tmp_path / "val.h5", [{"n": 500, "diamond": 310}, {"n": 330, "source": 3}], seed_base=1000),
        "test": write(tmp_path / "test.h5", [{"n": 600, "diamond": 320}, {"n": 560, "diamond": 280}, {"n": 540, "diamond": 260}], seed_base=2000),
    }
    out = tmp_path / "out"
    rec = numbers.compute([str(p) for p in files.values()], list(files), out, with_sha=True)
    saved = json.loads((out / "dataset_numbers.json").read_text())
    assert set(saved["splits"]) == {"train", "val", "test"} and saved["files"]["train"]["sha256"]
    assert saved["splits"]["test"]["spectral"] and "spectral" not in saved["splits"]["train"]
    assert saved["splits"]["train"]["time_to_diamond"]["episodes"] == 2
    cmp = saved["val_vs_train"]["val"]
    assert 0 <= cmp["length_ks"] <= 1 and cmp["action_share_max_abs_diff"] < 0.1 and cmp["change_share_max_abs_diff"] is not None
    assert rec["splits"]["train"]["chunks"]["1"]["distinct_chunks"] <= 17
    written = {p.name for p in figures.plot(out)}
    assert {"fig_length.pdf", "fig_achievements.pdf", "fig_change.pdf", "fig_clips.pdf", "fig_actions.pdf", "table_length.tex",
            "table_spectral.tex", "table_achievements.tex", "table_chunks.tex", "change_matrix.csv"} <= written  # fmt: skip
    for name in written:
        assert (out / name).stat().st_size > 0
    assert "Median" in (out / "table_length.tex").read_text()
    csv_head = (out / "change_matrix.csv").read_text().splitlines()[0]
    assert csv_head.startswith("split,group,component,half_lag,lag_1,lag_2")


# --------------------------------------------------------------------------
# the moved estimators: spectral gap, chunk information


def write_dynamics(path, lengths=(700,) * 4):
    """Episodes with known dynamics: iid facing and mobs, a random walk, the environment's clocks (food every 26 steps,
    drink 21, energy 31), a day of 300 steps, and staircases (wood at 100 + e, table at 250 + e)."""
    rng = np.random.default_rng(0)
    cols = {k: [] for k in data.Split.COLUMNS}
    for e, n in enumerate(lengths):
        t = np.arange(n)
        action = rng.integers(0, 17, n).astype(np.float32)
        action[-1] = np.nan
        flags = np.zeros((n, 22), np.uint8)
        flags[100 + e :, 0] = 1
        flags[250 + e :, 1] = 1
        inventory = np.zeros((n, 12), np.int16)
        for start in (100, 250, 400, 550):
            inventory[start + e :, 0] += 1
        vitals = np.stack([np.full(n, 9), 9 - (t // 26) % 9, 9 - (t // 21) % 9, 9 - (t // 31) % 9], 1).astype(np.int8)
        light = (1 - np.abs(np.cos(np.pi * ((t / 300) % 1 + 0.3))) ** 3).astype(np.float32)
        cols["action"].append(action[:, None])
        cols["achievements"].append(flags)
        cols["inventory"].append(inventory)
        cols["vitals"].append(vitals)
        cols["facing"].append(rng.integers(1, 5, (n, 1)).astype(np.int8))
        cols["daylight"].append(light[:, None])
        cols["position"].append(np.cumsum(rng.integers(-1, 2, (n, 2)), 0).astype(np.int16))
        cols["mobs_in_view"].append(rng.integers(0, 3, (n, 4)).astype(np.int8))
    with h5py.File(path, "w") as f:
        for name, parts in cols.items():
            f[name] = np.concatenate(parts)
        f["ep_len"] = np.array(lengths, np.int32)
        f["ep_offset"] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return path


def test_the_gap_of_known_dynamics(tmp_path):
    s = data.Split(write_dynamics(tmp_path / "dyn.h5"))
    result = spectral.entity_gap(s, windows=(64, 256), n_boot=40)
    gap = result["gap"]["all"]
    assert gap["ratio"] > 10
    assert gap["fastest"] in ("facing", "mobs_in_view")  # iid: period ~4
    assert gap["slowest"] == "daylight"
    assert gap["lower_bound"]  # the slowest entity is limited by the window
    assert gap["ci"]["p5"] <= gap["ratio"] * 1.05
    assert gap["ci"]["p5"] <= gap["ci"]["p50"] <= gap["ci"]["p95"]
    assert result["gap"]["without_daylight"]["slowest"] != "daylight"
    entities = result["entities"]
    assert entities["facing"]["period"] == pytest.approx(4, rel=0.1)
    assert not entities["facing"]["limited"]
    for slow in ("inventory", "achievements", "daylight"):
        assert entities[slow]["limited"]  # staircases and a day of 300 steps
    assert result["episodes"] == 4 and result["episodes_used"] == 4
    gaps = {w: v["gap"]["all"]["ratio"] for w, v in result["sensitivity"].items()}
    assert set(gaps) == {64, 256} and gaps[64] < gaps[256]  # a shorter window limits more


def test_episodes_shorter_than_the_window_have_no_segment(tmp_path):
    s = data.Split(write_dynamics(tmp_path / "dyn.h5", lengths=(700, 700, 100)))
    result = spectral.entity_gap(s, windows=(256,), n_boot=10)
    assert result["episodes"] == 3 and result["episodes_used"] == 2


def test_mutual_information_of_chunk_and_effect():
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 5, 20000)
    same = actions.mutual_information(chunk, chunk.copy())
    assert same["mi_bits"] == pytest.approx(np.log2(5), abs=0.01)
    assert same["coverage"] == 1 and same["chunk_classes"] == 5
    noise = actions.mutual_information(chunk, rng.integers(0, 5, 20000))
    assert abs(noise["mi_bits"]) < 0.01  # the correction removes the bias
    assert actions.mutual_information(chunk, chunk, min_count=10**6) is None
    rare = np.concatenate([chunk, np.arange(100, 130)])  # 30 unique chunks
    info = actions.mutual_information(rare, rare.copy(), min_count=20)
    assert info["coverage"] == pytest.approx(20000 / 20030, abs=1e-4)


def test_chunk_information_where_do_gives_wood_and_noop_does_nothing(tmp_path):
    n = 4000
    action = np.where(np.arange(n) % 2 == 0, 5, 0).astype(np.float32)
    action[-1] = np.nan
    inventory = np.zeros((n, 12), np.int16)
    inventory[1:, 0] = np.cumsum(action[:-1] == 5)  # a `do` gives wood in the next frame
    path = tmp_path / "still.h5"
    with h5py.File(path, "w") as f:
        f["action"] = action[:, None]
        f["inventory"] = inventory
        f["achievements"] = np.zeros((n, 22), np.uint8)
        f["vitals"] = np.full((n, 4), 9, np.int8)
        f["position"] = np.zeros((n, 2), np.int16)
        f["ep_len"] = np.array([n], np.int32)
        f["ep_offset"] = np.array([0], np.int64)
    # alternating do / noop: a window of one step is `do` (wood) or `noop` (nothing)
    out = actions.chunk_numbers(data.Split(path), chunks=(1,))["1"]
    info = out["information_interaction"]
    assert out["distinct_chunks"] == 2 and info["mi_bits"] > 0.5
    assert info["mi_bits"] <= info["h_effect_bits"]


def test_a_file_with_episodes_out_of_order_is_refused(tmp_path):
    path = write(tmp_path / "bad.h5", [{"n": 10}, {"n": 10}], state=False)
    with h5py.File(path, "r+") as f:
        f["ep_offset"][:] = [10, 0]
    with pytest.raises(SystemExit):
        data.Split(path)
