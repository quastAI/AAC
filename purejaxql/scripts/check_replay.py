"""Replay stored episodes and compare every column with the file: do they replay on this machine?

An episode is determined by its world seed and its actions, but floating point differs between a GPU and a CPU, and
the planning environment (H-JEPA) runs on the CPU. Collect on the GPU, then run this check on the CPU:

    JAX_PLATFORMS=cpu python scripts/check_replay.py --data craftax_expert_train.h5 --num-episodes 20

The discrete columns (achievements, inventory, vitals, facing, position, mobs, dead) must be identical, otherwise
the episode is not replayable here and the script exits with 1, naming the first frame that differs.
The full state is compared leaf by leaf (the layout of `pack_state`): every integer and boolean leaf (map, positions,
inventory, random keys, timestep) must be identical; a float leaf may differ by FLOAT_TOLERANCE. The one float that
is computed and not counted is the light level: a GPU and a CPU differ in its last bits, which changes the state bytes
without changing the game. The float columns (observation, reward, daylight, rendered pixels) are reported with the
largest difference.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import jax
import numpy as np

DISCRETE = ("full_state", "achievements", "inventory", "vitals", "facing", "position", "mobs_in_view", "dead")
TOLERANCE = {"state": 1e-3, "reward": 1e-6, "daylight": 1e-6, "pixels": 1}  # largest difference that is no failure
FLOAT_TOLERANCE = 1e-6  # float leaves of the full state (the light level, in last-bit differences about 2e-7)


def load_collector():
    path = Path(__file__).with_name("collect_craftax_episodes.py")
    spec = importlib.util.spec_from_file_location("collect_craftax_episodes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_layout(collector, env=None) -> list[tuple]:
    """(name, dtype, byte offset, bytes) of every leaf of the packed state, in the order of `pack_state`."""
    env = env or collector.make_craftax_env_from_name(collector.ENV_NAME, auto_reset=False)
    _, state = env.reset(jax.random.PRNGKey(0), env.default_params)
    layout, offset = [], 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        leaf = np.asarray(leaf)  # booleans are packed as one byte
        layout.append((jax.tree_util.keystr(path), leaf.dtype, offset, leaf.size * leaf.dtype.itemsize))
        offset += layout[-1][3]
    return layout


def compare_state(old: np.ndarray, new: np.ndarray, layout: list[tuple]) -> dict:
    """Per leaf of the packed state: where it differs, and whether that is a failure.

    Integer and boolean leaves must be identical. A float leaf may differ by FLOAT_TOLERANCE; it is listed in `leaves`
    with its largest difference either way.
    """
    leaves = {}
    for name, dtype, offset, size in layout:
        a, b = (
            np.ascontiguousarray(old[:, offset : offset + size]),
            np.ascontiguousarray(new[:, offset : offset + size]),
        )
        differs = (a != b).any(1)
        if not differs.any():
            continue
        entry = {"first_frame": int(np.argmax(differs)), "ok": False}
        if np.issubdtype(dtype, np.floating):
            diff = np.abs(a.view(dtype).astype(np.float64) - b.view(dtype).astype(np.float64))
            entry["max_abs_diff"] = float(diff.max())
            entry["ok"] = entry["max_abs_diff"] <= FLOAT_TOLERANCE
        leaves[name] = entry
    bad = [e["first_frame"] for e in leaves.values() if not e["ok"]]
    return {"ok": not bad, "first_frame": min(bad) if bad else None, "leaves": leaves}


def compare(rendered: dict, stored: dict, layout: list[tuple] | None = None) -> dict:
    """Per column: how the replay differs from the file ({} entries for columns the file does not have)."""
    out = {}
    for name, new in rendered.items():
        if name not in stored:
            continue
        old = stored[name]
        if old.shape != new.shape:
            out[name] = {"ok": False, "first_frame": 0, "note": f"shape {new.shape} against {old.shape}"}
            continue
        if name == "full_state" and layout is not None:
            out[name] = compare_state(old, new, layout)
            continue
        differs = (old != new).reshape(len(old), -1).any(1)
        entry = {"ok": not differs.any(), "first_frame": int(np.argmax(differs)) if differs.any() else None}
        if name not in DISCRETE:
            diff = np.abs(old.astype(np.float64) - new.astype(np.float64))
            entry["max_abs_diff"] = float(diff.max()) if diff.size else 0.0
            entry["ok"] = entry["max_abs_diff"] <= TOLERANCE.get(name, 0)
        out[name] = entry
    return out


def check(path, num_episodes=20, seed=0, source=None, collector=None) -> dict:
    """Replay a random sample of the episodes of `path`; returns the per-column results and the failing episodes."""
    collector = collector or load_collector()
    env = collector.make_craftax_env_from_name(collector.ENV_NAME, auto_reset=False)
    replay = collector.make_replay(env, env.default_params)
    layout = state_layout(collector, env)
    with h5py.File(path, "r") as f:
        offsets, lengths = f["ep_offset"][:], f["ep_len"][:]
        eligible = np.arange(len(lengths))
        if source is not None:
            eligible = eligible[f["source"][offsets] == source]
        chosen = np.random.default_rng(seed).choice(eligible, min(num_episodes, len(eligible)), replace=False)
        columns = [k for k in f if k not in ("ep_offset", "ep_len")]
        report = {"backend": str(jax.devices()[0]), "episodes": [], "columns": {}, "leaves": {}}
        if "full_state" in f and f["full_state"].shape[1] != sum(leaf[3] for leaf in layout):
            raise SystemExit("the state layout of this Craftax version does not match the file: not the pinned version")
        for e in sorted(chosen):
            o, n = int(offsets[e]), int(lengths[e])
            stored = {k: f[k][o : o + n] for k in columns if k in DISCRETE or k in TOLERANCE}
            seed_of = int(f["seed"][o])
            actions = f["action"][o : o + n - 1, 0].astype(np.int32)  # n frames, n - 1 actions
            result = compare(replay(seed_of, actions), stored, layout)
            failed = {k: v for k, v in result.items() if not v["ok"]}
            report["episodes"].append({"episode": int(e), "seed": seed_of, "frames": n, "failed": failed})
            for leaf, v in result.get("full_state", {}).get("leaves", {}).items():
                info = report["leaves"].setdefault(leaf, {"episodes": 0, "failed": 0, "max_abs_diff": 0.0})
                info["episodes"] += 1
                info["failed"] += int(not v["ok"])
                info["max_abs_diff"] = max(info["max_abs_diff"], v.get("max_abs_diff", 0.0))
            for k, v in result.items():
                col = report["columns"].setdefault(k, {"episodes_ok": 0, "episodes": 0, "max_abs_diff": 0.0})
                col["episodes"] += 1
                col["episodes_ok"] += int(v["ok"])
                col["max_abs_diff"] = max(col["max_abs_diff"], v.get("max_abs_diff", 0.0))
    report["replayable"] = bool(report["episodes"]) and all(
        report["columns"][k]["episodes_ok"] == report["columns"][k]["episodes"]
        for k in DISCRETE
        if k in report["columns"]
    )
    return report


def render(report: dict) -> str:
    lines = [f"Replayed {len(report['episodes'])} episodes on {report['backend']}", "",
             "| column | episodes identical | largest difference |", "|---|---|---|"]  # fmt: skip
    for name, c in report["columns"].items():
        kind = "" if name in DISCRETE else f"{c['max_abs_diff']:.3g}"
        lines.append(f"| {name} | {c['episodes_ok']} of {c['episodes']} | {kind or 'exact required'} |")
    for leaf, info in report["leaves"].items():  # what differs inside the full state, even where it is within tolerance
        float_leaf = "max_abs_diff" in info and info["max_abs_diff"] > 0
        lines.append(f"full_state leaf {leaf}: differs in {info['episodes']} of {len(report['episodes'])} episodes, "
                     + (f"largest difference {info['max_abs_diff']:.3g} (a float leaf: tolerance {FLOAT_TOLERANCE:g})"
                        if float_leaf else "an integer or boolean leaf, which must be identical")
                     + ("" if info["failed"] else ": no failure"))  # fmt: skip
    for e in report["episodes"]:
        for name, v in e["failed"].items():
            bad = [leaf for leaf, x in v.get("leaves", {}).items() if not x["ok"]]
            lines.append(f"episode {e['episode']} (seed {e['seed']}): {name} differs from frame {v['first_frame']}"
                         + (f" ({v['note']})" if "note" in v else "")
                         + (f" in {', '.join(bad)}" if bad else ""))  # fmt: skip
    lines.append("")
    if not report["episodes"]:
        return "\n".join(lines + ["No episode to replay (is --source in the file?)."])
    lines.append("REPLAYABLE: the discrete columns are identical." if report["replayable"]
                 else "NOT REPLAYABLE on this backend: a discrete column differs (see above).")  # fmt: skip
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="HDF5 episode file")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0, help="seed of the sample of episodes")
    p.add_argument("--source", type=int, default=None, help="only episodes of this source (0 expert, 3 early, ...)")
    args = p.parse_args(argv)
    report = check(args.data, args.num_episodes, args.seed, args.source)
    print(render(report))
    return 0 if report["replayable"] else 1


if __name__ == "__main__":
    sys.exit(main())
