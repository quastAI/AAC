"""Every number of the dataset section in one JSON: the splits' numbers, their provenance, validation against train."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .actions import action_numbers, chunk_numbers
from .change import LAGS, change_matrix
from .data import ACTIONS, Split, file_record, git_state
from .episodes import STRIDES, achievement_numbers, clip_coverage, length_numbers, time_to_diamond
from .spectral import entity_gap

JSON_NAME = "dataset_numbers.json"
SPECTRAL_SPLIT = "test"  # the final expert only: the entity gap is a property of the expert's behaviour


def split_numbers(path, spectral: bool) -> dict:
    """All numbers of one episode file, taken as one set."""
    s = Split(path)
    out = {
        "length": length_numbers(s), "time_to_diamond": time_to_diamond(s), "achievements": achievement_numbers(s),
        "change": change_matrix(s), "clips": clip_coverage(s), "actions": action_numbers(s), "chunks": chunk_numbers(s),
    }  # fmt: skip
    if spectral:
        out["spectral"] = entity_gap(s)
    return out


def ks_statistic(x, y) -> float:
    """Two-sample Kolmogorov-Smirnov distance: the largest gap between the two empirical CDFs."""
    x, y = np.sort(np.asarray(x, float)), np.sort(np.asarray(y, float))
    grid = np.concatenate([x, y])
    cdf = lambda v: np.searchsorted(v, grid, side="right") / len(v)  # noqa: E731
    return float(np.abs(cdf(x) - cdf(y)).max())


def _achievement_counts(numbers: dict) -> np.ndarray:
    histogram = numbers["achievements"]["per_episode"]["histogram"]
    return np.repeat(np.arange(len(histogram)), histogram)


def compare_with_train(split: dict, train: dict) -> dict:
    """Distances between a split and train, the content of the figures as a few numbers (val should follow train)."""
    cells = [
        abs(a - b)
        for ra, rb in zip(split["change"]["rows"], train["change"]["rows"])
        for a, b in zip(ra["share"], rb["share"])
        if a is not None and b is not None
    ]
    return {
        "length_ks": ks_statistic(split["length"]["lengths"], train["length"]["lengths"]),
        "achievements_per_episode_ks": ks_statistic(_achievement_counts(split), _achievement_counts(train)),
        "action_share_max_abs_diff": max(abs(split["actions"]["share"][a] - train["actions"]["share"][a]) for a in ACTIONS),
        "change_share_max_abs_diff": max(cells) if cells else None,
    }  # fmt: skip


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def compute(data: list[str], labels: list[str], out: Path, with_sha: bool = False) -> dict:
    """Numbers of every file (label train, val or test) into `out / dataset_numbers.json`."""
    out.mkdir(parents=True, exist_ok=True)
    record = {"git": git_state(), "lags": list(LAGS), "strides": list(STRIDES), "splits": {}, "files": {}}
    for path, label in zip(data, labels):
        print(f"{label}: {path}", flush=True)
        record["files"][label] = file_record(path, with_sha) | {"runs": Split(path).runs}
        record["splits"][label] = split_numbers(path, spectral=label == SPECTRAL_SPLIT)
    if "train" in record["splits"] and "val" in record["splits"]:
        record["val_vs_train"] = {"val": compare_with_train(record["splits"]["val"], record["splits"]["train"])}
    (out / JSON_NAME).write_text(json.dumps(jsonable(record), indent=1))
    print(f"wrote {out / JSON_NAME}")
    return record
