"""Evaluation tasks for Craftax-Classic: start frame, goal frame, required progress.

A task is (episode, start, goal). Its *required* set is the achievements the goal
frame has and the start frame lacks; the environment's goal test succeeds when
all of them are unlocked, so a task always contains progress. Tasks come from
held-out dataset episodes, at most one per episode (the evaluation asserts that
all world seeds differ). Only expert episodes (source 0, the ones that reach
the diamond) are used: a goal frame from a noised or random episode is not a goal
the expert reaches.

Modes
    offset        goal frame = start + one of `offsets`; only pairs with progress.
    achievements  goal frame = the frame of the d-th next unlock after the start,
                  d in `depths`. The goal distance is the number of achievements
                  and not a number of steps.
    episode       protocol B: start at frame 0, goal = the frame where `target`
                  (default diamond) is unlocked, so the whole chain is required.

The payload has the format `World.evaluate_from_dataset(load_eval_trajs_path=)`
reads, plus `task_info` (one entry per task) for the planning report.

    python eval_tasks.py --dataset craftax_expert_val --mode offset \
        --offsets 25 50 100 --num-tasks 200 --out tasks.pt
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ACHIEVEMENT_NAMES = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie",
    "collect_stone", "place_stone", "eat_plant", "defeat_skeleton",
    "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace",
    "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe",
    "make_iron_sword",
]  # fmt: skip
DIAMOND = ACHIEVEMENT_NAMES.index("collect_diamond")


def episode_achievements(dataset) -> list[np.ndarray]:
    """Per episode, the cumulative achievement flags (length, 22) as bool."""
    flags = np.asarray(dataset.get_col_data("achievements")).astype(bool)
    return [flags[o : o + n] for o, n in zip(dataset.offsets, dataset.lengths)]


def episode_sources(dataset) -> np.ndarray:
    """Per episode, the `source` column (0 expert, 1 noised expert, 2 random)."""
    if "source" not in dataset.column_names:
        raise ValueError(
            "the dataset has no `source` column, so the expert episodes cannot be told apart; "
            "pass sources=None (--all-sources) to use every episode"
        )
    source = np.asarray(dataset.get_col_data("source")).reshape(-1)
    return np.array([source[o] for o in dataset.offsets])


def unlock_frames(flags: np.ndarray) -> np.ndarray:
    """Frames at which an achievement is newly set, one entry per unlock."""
    new = flags[1:] & ~flags[:-1]
    return np.sort(np.nonzero(new)[0] + 1)  # one entry per achievement, ties repeat


def _task(episode, start, goal, flags, **extra) -> dict:
    required = np.nonzero(flags[goal] & ~flags[start])[0].tolist()
    return {
        "episode": int(episode),
        "start": int(start),
        "goal": int(goal),
        "distance": int(goal - start),
        "required": required,
        "required_names": [ACHIEVEMENT_NAMES[i] for i in required],
        **extra,
    }


def _offset_task(rng, e, flags, offsets, preferred):
    """Pick a start with progress for the preferred offset, else any offset."""
    order = [preferred] + [o for o in offsets if o != preferred]
    for offset in order:
        if flags.shape[0] <= offset:
            continue
        starts = np.arange(flags.shape[0] - offset)
        progress = (flags[starts + offset] & ~flags[starts]).any(1)
        if progress.any():
            start = int(rng.choice(starts[progress]))
            return _task(e, start, start + offset, flags, offset=int(offset))
    return None


def _depth_task(rng, e, flags, depths, preferred, max_distance):
    events = unlock_frames(flags)
    for depth in [preferred] + [d for d in depths if d != preferred]:
        if len(events) < depth:
            continue
        # starts that still have `depth` unlocks ahead: the d-th next unlock is the goal
        starts = np.arange(events[-depth])
        goals = events[np.searchsorted(events, starts, side="right") + depth - 1]
        ok = np.ones(len(starts), dtype=bool) if max_distance is None else goals - starts <= max_distance
        if ok.any():
            i = int(rng.choice(np.nonzero(ok)[0]))
            return _task(e, starts[i], goals[i], flags, depth=int(depth))
    return None


def make_tasks(
    dataset,
    *,
    mode: str,
    num_tasks: int,
    seed: int = 0,
    offsets=(25, 50, 100, 200),
    depths=(1, 2, 3),
    max_distance: int | None = None,
    target: int = DIAMOND,
    sources: tuple[int, ...] | None = (0,),
) -> list[dict]:
    """Sample up to `num_tasks` tasks, one per episode, spread over offsets or depths.

    Only episodes whose `source` is in `sources` are used (default: expert episodes); None uses all.
    """
    if mode not in ("offset", "achievements", "episode"):
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed)
    flags = episode_achievements(dataset)
    allowed = np.ones(len(flags), dtype=bool)
    if sources is not None:
        allowed = np.isin(episode_sources(dataset), list(sources))
    tasks = []
    for i, e in enumerate(rng.permutation(np.nonzero(allowed)[0])):
        if len(tasks) >= num_tasks:
            break
        f = flags[e]
        if mode == "offset":
            task = _offset_task(rng, e, f, list(offsets), offsets[i % len(offsets)])
        elif mode == "achievements":
            task = _depth_task(rng, e, f, list(depths), depths[i % len(depths)], max_distance)
        else:
            if not f[:, target].any():
                continue
            goal = int(np.argmax(f[:, target]))
            task = _task(e, 0, goal, f) if goal > 0 else None
            if task is not None and max_distance is not None and task["distance"] > max_distance:
                task = None
        if task is not None and task["required"]:
            task["mode"] = mode
            tasks.append(task)
    return tasks


def build_payload(dataset, tasks: list[dict], eval_budget: int, seed: int = 0) -> dict:
    """The task file: frames [start, goal] of every task, in the format the evaluation loads."""
    data = [
        dataset.load_chunk(np.array([t["episode"]]), np.array([t["start"]]), np.array([t["goal"] + 1]))[0]
        for t in tasks
    ]
    return {
        "data": data,
        "columns": list(dataset.column_names),
        "episodes_idx": [t["episode"] for t in tasks],
        "start_steps": [t["start"] for t in tasks],
        "goal_offset_steps": max(t["distance"] for t in tasks),
        "eval_budget": int(eval_budget),
        "process": None,
        "task_info": {"seed": seed, "tasks": tasks},
    }


def describe(tasks: list[dict]) -> str:
    """Short summary of a task set, to check the spread before running anything."""
    if not tasks:
        return "no tasks"
    dist = np.array([t["distance"] for t in tasks])
    by_required = Counter(len(t["required"]) for t in tasks)
    by_name = Counter(n for t in tasks for n in t["required_names"])
    lines = [
        f"{len(tasks)} tasks, goal distance min/median/max {dist.min()}/{int(np.median(dist))}/{dist.max()} steps",
        "required achievements per task: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_required.items())),
        "most common required: " + ", ".join(f"{n} ({c})" for n, c in by_name.most_common(6)),
    ]
    for key in ("offset", "depth"):
        if key in tasks[0]:
            counts = Counter(t[key] for t in tasks)
            lines.append(f"by {key}: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset name (HDF5 under the cache dir)")
    parser.add_argument("--cache-dir", default=os.environ.get("HJEPA_HOME"))
    parser.add_argument("--mode", choices=["offset", "achievements", "episode"], default="offset")
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--offsets", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-distance", type=int, default=None)
    parser.add_argument("--eval-budget", type=int, required=True, help="environment steps the planner gets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sources", type=int, nargs="+", default=[0], help="episode sources to use (0 expert)")
    parser.add_argument("--all-sources", action="store_true", help="use every episode, whatever its source")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import stable_worldmodel as swm

    dataset = swm.data.HDF5Dataset(
        args.dataset, cache_dir=Path(args.cache_dir), level1={"frameskip": 1, "num_steps": 1}
    )
    tasks = make_tasks(
        dataset, mode=args.mode, num_tasks=args.num_tasks, seed=args.seed, offsets=args.offsets,
        depths=args.depths, max_distance=args.max_distance, sources=None if args.all_sources else tuple(args.sources),
    )  # fmt: skip
    print(describe(tasks))
    if len(tasks) < args.num_tasks:
        print(f"warning: only {len(tasks)} of {args.num_tasks} tasks found (one per episode, with progress)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_payload(dataset, tasks, args.eval_budget, args.seed), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
