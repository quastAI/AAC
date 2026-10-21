"""Relabel the actions taken while asleep as noop, in place: `action` becomes the action the game executed.

Craftax replaces every action by a noop while the player sleeps (`craftax_step` starts with
`action = select(state.is_sleeping, NOOP, action)`), so the policy's choice in those steps had no effect. Files
collected before the collector wrote both columns hold the policy's choice in `action`. This script keeps it in a new
column `action_policy` (written once, never overwritten) and writes the executed actions into `action`, with the
collector's own rule (`executed_actions`: noop where the sleeping flag of the frame, the last entry of `state`, is
set). Nothing else in the file changes. Running it again recomputes `action` from `action_policy`, with the same
result. The file attribute `relabel` records the rule, the counts and the code that wrote them.

    python scripts/relabel_asleep.py --data craftax_expert_train.h5 --data craftax_expert_val.h5

Then check on the CPU that the file still replays (`check_replay.py`: the replay of `action` must reproduce every
column, and `action` must follow the rule).
"""

import argparse
import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

STATE_DIM = 1345
CHUNK = 100_000  # frames per read of the sleeping flag
RULE = "action = noop (0) where the player is asleep in the frame of the action (state[:, -1] > 0.5), else action_policy"


def load_collector():
    path = Path(__file__).with_name("collect_craftax_episodes.py")
    spec = importlib.util.spec_from_file_location("collect_craftax_episodes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_commit():
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def relabel(path, collector=None) -> dict:
    """Relabel one file in place; returns the record written to its `relabel` attribute."""
    collector = collector or load_collector()
    with h5py.File(path, "r+") as f:
        if "state" not in f or f["state"].shape[1] != STATE_DIM:
            raise SystemExit(f"{path}: no `state` column of the collector, so the sleeping flag is unknown")
        if "ep_len" not in f or "action" not in f:
            raise SystemExit(f"{path}: not an episode file of the collector")
        n = f["action"].shape[0]
        if "action_policy" not in f:
            # first the copy of the policy's actions, so an interrupted run can always start again from it
            f.create_dataset("action_policy", data=f["action"][:], maxshape=f["action"].maxshape, chunks=f["action"].chunks)
            f.flush()
        policy = f["action_policy"][:]
        if policy.shape != (n, 1):
            raise SystemExit(f"{path}: action_policy has shape {policy.shape}, action {f['action'].shape}")
        executed = np.empty_like(policy)
        asleep = np.empty(n, bool)
        for a in range(0, n, CHUNK):
            sleeping = f["state"][a : a + CHUNK, -1:]  # the flag alone, as the last (and only) column
            executed[a : a + CHUNK] = collector.executed_actions(policy[a : a + CHUNK], sleeping)
            asleep[a : a + CHUNK] = sleeping[:, 0] > 0.5
        f["action"][:] = executed
        f.flush()
        stored = f["action"][:]
        if not np.array_equal(stored, executed, equal_nan=True):
            raise SystemExit(f"{path}: the written actions do not read back")
        valid = ~np.isnan(policy[:, 0])
        asleep &= valid
        changed = valid & (executed[:, 0] != policy[:, 0])
        if (changed & ~asleep).any() or (executed[asleep, 0] != collector.NOOP).any():
            raise SystemExit(f"{path}: an action changed outside the asleep steps, or an asleep step is not a noop")
        by_action = np.bincount(policy[changed, 0].astype(np.int64), minlength=17)
        record = {
            "rule": RULE,
            "steps": int(valid.sum()),
            "asleep_steps": int(asleep.sum()),  # all steps taken asleep (now noop)
            "changed": int(changed.sum()),  # of those, the ones whose policy action was not already a noop
            "changed_share": float(changed.sum() / max(1, valid.sum())),
            "changed_by_policy_action": {str(i): int(c) for i, c in enumerate(by_action) if c},
            "git_commit": _git_commit(),
            "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        f.attrs["relabel"] = json.dumps(record)
        f.flush()
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", required=True, help="HDF5 episode file (repeat for several)")
    args = p.parse_args(argv)
    collector = load_collector()
    for path in args.data:
        record = relabel(path, collector)
        print(f"{path}: {record['asleep_steps']:,} of {record['steps']:,} steps taken asleep; {record['changed']:,} of them "
              f"({100 * record['changed_share']:.1f}% of all steps) relabeled as noop; by the policy's action "
              f"{record['changed_by_policy_action']}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
