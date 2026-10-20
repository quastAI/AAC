"""The dataset tables of the appendix (one per split: composition by policy) from the episode files, with provenance.

Reads only the small columns of the collector's HDF5 files (numpy, h5py, pyyaml; no pixels, no full_state). One row per
policy that acted, the rows of a source grouped by `policy_updates` (the snapshot of the expert's training), plus an All
row. Per episode, in the table's columns:

* share: frames of the policy / frames of the split (the mix is counted in frames);
* length: frames per episode (`ep_len`, L + 1 frames for L actions);
* diamond: the final frame has `collect_diamond` (achievement 19);
* return: sum of `reward` over the episode (undiscounted);
* achievements: number of the 22 flags set in the final frame;
* death: the `dead` flag of the final frame (other episodes end at the horizon, or the expert 500 steps after the
  diamond);
* train steps: environment steps of the expert's training at the snapshot, `policy_updates` x NUM_ENVS x NUM_STEPS of the
  training config in `--checkpoints`; the final expert (`policy_updates` -1) takes NUM_UPDATES of the config.

    python -m dataset_section splits --data expert_train.h5 --label train --data expert_val.h5 --label val \
        --data expert_test.h5 --label test --checkpoints <hf>/checkpoints --out reports/dataset/

Writes `table_<label>.tex`, `dataset_tables.csv` and `dataset_tables.json`. The JSON records, per file, its size, mtime,
(with `--sha256`) its hash, the collector's `runs` records and the git commit of this repository, so a number in the paper
can be traced to the file and the code that produced it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from .data import ACHIEVEMENTS, DIAMOND_INDEX, file_record, git_state

NUM_ACHIEVEMENTS = len(ACHIEVEMENTS)
SOURCE_NAMES = {0: "Final expert", 1: "Noised expert", 2: "Random", 3: "Snapshot"}  # the collector's `source` column
COLUMNS = ["policy", "train_steps", "episodes", "frames", "share", "length", "diamond", "return", "achievements", "death"]


def read_episodes(path) -> dict:
    """Per episode: source, snapshot updates, seed, length, return, diamond, achievements and death."""
    with h5py.File(path, "r") as f:
        length = f["ep_len"][:].astype(np.int64)
        offset = f["ep_offset"][:].astype(np.int64)
        last = offset + length - 1
        reward = f["reward"][:].astype(np.float64)
        csum = np.concatenate([[0.0], np.cumsum(reward)])
        final = f["achievements"][last]
        return {
            "source": f["source"][offset].astype(int),
            "updates": f["policy_updates"][offset].astype(int),
            "seed": f["seed"][offset].astype(np.int64),
            "length": length,
            "return": csum[last + 1] - csum[offset],
            "diamond": final[:, DIAMOND_INDEX] > 0,
            "achievements": final.sum(axis=1),
            "death": f["dead"][last].astype(bool),
            "runs": json.loads(f.attrs["runs"]) if "runs" in f.attrs else [],
        }


def training_steps(checkpoints) -> tuple[int, int]:
    """(environment steps per update, number of updates of the final expert) from the training config."""
    import yaml  # only the train steps need it, so that `compute` runs with numpy and h5py alone

    configs = sorted(Path(checkpoints).glob("*_config.yaml"))
    if len(configs) != 1:
        raise SystemExit(f"expected one *_config.yaml in {checkpoints}, found {len(configs)}")
    cfg = yaml.safe_load(configs[0].read_text())
    return int(cfg["NUM_ENVS"]) * int(cfg["NUM_STEPS"]), int(cfg["NUM_UPDATES"])


def split_rows(ep: dict, steps_per_update: int | None = None, final_updates: int | None = None) -> list[dict]:
    """One row per (source, snapshot), the snapshots numbered by their updates, then the All row."""
    n_frames = int(ep["length"].sum())
    keys = sorted(set(zip(ep["source"].tolist(), ep["updates"].tolist())), key=lambda k: (k[0] != 3, k[0], k[1]))
    snapshots = [k for k in keys if k[0] == 3]
    rows = []
    for key in keys + [None]:
        m = np.ones(len(ep["length"]), bool) if key is None else (ep["source"] == key[0]) & (ep["updates"] == key[1])
        if key is None:
            name, updates = "All", None
        elif key[0] == 3:
            name, updates = f"Snapshot {snapshots.index(key) + 1}", key[1]
        else:
            name, updates = SOURCE_NAMES.get(key[0], f"Source {key[0]}"), final_updates if key[1] < 0 else key[1]
        frames = int(ep["length"][m].sum())
        rows.append(
            {
                "policy": name,
                "train_steps": None if updates is None or steps_per_update is None else updates * steps_per_update,
                "episodes": int(m.sum()),
                "frames": frames,
                "share": frames / n_frames,
                "length": float(ep["length"][m].mean()),
                "diamond": float(ep["diamond"][m].mean()),
                "return": float(ep["return"][m].mean()),
                "achievements": float(ep["achievements"][m].mean()),
                "death": float(ep["death"][m].mean()),
            }
        )
    return rows


def millions(x) -> str:
    return "---" if x is None else f"{x / 1e6:.0f}M"


def percent(x, digits=1) -> str:
    return "100\\%" if x == 1 else f"{100 * x:.{digits}f}\\%"


def human(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1_000_000 else f"{n / 1e3:.0f}K"


def latex_table(label: str, rows: list[dict], ep: dict, caption: str) -> str:
    frames, episodes = rows[-1]["frames"], rows[-1]["episodes"]
    thousands = lambda n: f"{n:,}".replace(",", "\\,")  # noqa: E731
    seeds = f"world seeds {thousands(int(ep['seed'].min()))} to {thousands(int(ep['seed'].max()))}"
    cap = f"{caption}: {human(frames)} frames, {thousands(episodes)} episodes, {seeds}."
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{cap}}}",
        f"\\label{{tab:data-{label}}}",
        "\\small",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Policy & Train steps & Share & Length & Diamond & Return & Achiev. & Death \\\\",
        "\\midrule",
    ]
    body = [r for r in rows if r["policy"] != "All"]
    for r in body:
        lines.append(table_row(r))
    if len(body) > 1:
        lines += ["\\midrule", table_row(rows[-1])]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def table_row(r: dict) -> str:
    cells = [
        r["policy"],
        millions(r["train_steps"]),
        percent(r["share"]),
        f"{r['length']:.0f}",
        percent(r["diamond"]),
        f"{r['return']:.2f}",
        f"{r['achievements']:.2f}",
        percent(r["death"]),
    ]
    return " & ".join(cells) + " \\\\"


CAPTIONS = {
    "train": "Train split",
    "val": "Validation split",
    "test": "Test split (task set, final expert only)",
}


def write_split_tables(data: list[str], labels: list[str], out: Path, checkpoints: str | None = None, with_sha: bool = False):
    """`table_<label>.tex`, `dataset_tables.csv` and `dataset_tables.json` for every file (label train, val or test)."""
    spu, final_updates = training_steps(checkpoints) if checkpoints else (None, None)
    out.mkdir(parents=True, exist_ok=True)
    record = {"git": git_state(), "steps_per_update": spu, "final_updates": final_updates, "splits": {}}
    csv_rows, all_seeds = [], {}
    for path, label in zip(data, labels):
        ep = read_episodes(path)
        all_seeds[label] = set(ep["seed"].tolist())
        rows = split_rows(ep, spu, final_updates)
        (out / f"table_{label}.tex").write_text(latex_table(label, rows, ep, CAPTIONS.get(label, f"Split {label}")))
        record["splits"][label] = {
            "file": file_record(path, with_sha),
            "runs": ep["runs"],
            "seed_min": int(ep["seed"].min()),
            "seed_max": int(ep["seed"].max()),
            "unique_seeds": int(len(set(ep["seed"].tolist()))),
            "episodes": int(len(ep["length"])),
            "rows": rows,
        }
        csv_rows += [{"split": label, **{c: r[c] for c in COLUMNS}} for r in rows]
        print(f"{label}: {rows[-1]['episodes']} episodes, {rows[-1]['frames']} frames")
        for r in rows:
            print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    names = list(all_seeds)
    record["seed_overlap"] = {
        f"{a}/{b}": len(all_seeds[a] & all_seeds[b]) for i, a in enumerate(names) for b in names[i + 1 :]
    }
    print("seed overlap between files:", record["seed_overlap"])
    with open(out / "dataset_tables.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split"] + COLUMNS)
        w.writeheader()
        w.writerows(csv_rows)
    (out / "dataset_tables.json").write_text(json.dumps(record, indent=1))
    print(f"wrote {out}/")

