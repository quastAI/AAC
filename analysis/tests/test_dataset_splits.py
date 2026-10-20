"""dataset_section.splits on a synthetic file with known answers."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from dataset_section import splits as dt  # noqa: E402
from dataset_section.__main__ import main  # noqa: E402

# (source, policy_updates, length, diamond, dead, reward per frame)
EPISODES = [
    (0, -1, 100, True, False, 0.5),
    (0, -1, 300, True, True, 0.25),
    (3, 10, 50, False, True, 0.1),
    (3, 20, 50, False, True, 0.2),
]


def write(path):
    cols = {k: [] for k in ("source", "policy_updates", "seed", "reward", "achievements", "dead")}
    lengths, offsets, offset = [], [], 0
    for e, (src, upd, n, diamond, dead, rew) in enumerate(EPISODES):
        flags = np.zeros((n, dt.NUM_ACHIEVEMENTS), np.uint8)
        flags[n - 1, :3] = 1
        flags[n - 1, dt.DIAMOND_INDEX] = diamond
        reward = np.full(n, rew, np.float32)
        reward[0] = 0
        d = np.zeros(n, np.uint8)
        d[n - 1] = dead
        cols["source"].append(np.full(n, src, np.int8))
        cols["policy_updates"].append(np.full(n, upd, np.int32))
        cols["seed"].append(np.full(n, 1000 + e, np.int64))
        cols["reward"].append(reward)
        cols["achievements"].append(flags)
        cols["dead"].append(d)
        lengths.append(n)
        offsets.append(offset)
        offset += n
    with h5py.File(path, "w") as f:
        for k, v in cols.items():
            f[k] = np.concatenate(v)
        f["ep_len"] = np.array(lengths, np.int32)
        f["ep_offset"] = np.array(offsets, np.int64)
        f.attrs["runs"] = json.dumps([{"finished": True}])


def test_rows(tmp_path):
    write(tmp_path / "x.h5")
    rows = dt.split_rows(dt.read_episodes(tmp_path / "x.h5"), steps_per_update=1000, final_updates=99)
    by = {r["policy"]: r for r in rows}
    assert list(by) == ["Snapshot 1", "Snapshot 2", "Final expert", "All"]
    assert by["Snapshot 1"]["train_steps"] == 10_000 and by["Snapshot 2"]["train_steps"] == 20_000
    assert by["Final expert"]["train_steps"] == 99_000 and by["All"]["train_steps"] is None
    exp = by["Final expert"]
    assert exp["episodes"] == 2 and exp["frames"] == 400 and exp["share"] == 400 / 500
    assert exp["length"] == 200
    assert exp["diamond"] == 1.0 and exp["death"] == 0.5
    assert np.isclose(exp["return"], (99 * 0.5 + 299 * 0.25) / 2)
    assert exp["achievements"] == 3 + 1  # three flags and the diamond
    assert by["Snapshot 1"]["diamond"] == 0 and by["Snapshot 1"]["death"] == 1
    assert by["All"]["episodes"] == 4 and by["All"]["diamond"] == 0.5
    assert np.isclose(sum(r["share"] for r in rows[:-1]), 1)


def test_latex_and_files(tmp_path):
    write(tmp_path / "x.h5")
    ep = dt.read_episodes(tmp_path / "x.h5")
    rows = dt.split_rows(ep, 1000, 99)
    tex = dt.latex_table("train", rows)
    assert "Snapshot 1 & 0M & 10.0\\% & 1 & 50 & 0.0\\% & 4.90 & 3.00 & 100\\% \\\\" in tex  # the episode count is a column
    assert "Final expert & 0M & 80.0\\% & 2 & 200 & 100.0\\% & " in tex  # the diamond rate always has a decimal
    assert tex.count("\\midrule") == 2 and "\\label{tab:data-train}" in tex and tex.startswith("\\begin{table}[H]")
    assert "training dataset by data-collection policy" in tex and "\\textbf{Columns.}" in tex and "Episodes: number of episodes" in tex
    assert "the remaining expert episodes end 500 steps after the diamond" in tex
    test_tex = dt.latex_table("test", rows[2:3])
    assert "collected with the final expert only" in test_tex and "the remaining episodes end 500 steps after the diamond" in test_tex
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "run_config.yaml").write_text("NUM_ENVS: 4\nNUM_STEPS: 8\nNUM_UPDATES: 7\n")
    assert dt.training_steps(ckpt) == (32, 7)
    out = tmp_path / "out"
    main(["splits", "--data", str(tmp_path / "x.h5"), "--label", "val", "--checkpoints", str(ckpt), "--out", str(out)])
    record = json.loads((out / "dataset_tables.json").read_text())
    assert record["splits"]["val"]["unique_seeds"] == 4 and record["splits"]["val"]["runs"] == [{"finished": True}]
    assert (out / "table_val.tex").exists() and (out / "dataset_tables.csv").exists()
