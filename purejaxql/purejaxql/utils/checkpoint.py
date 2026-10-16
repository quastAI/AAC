"""Checkpoints of a long training run: the full runner state (to resume) and policy snapshots.

The runner state is a pytree (train state with optimizer, rollout memory, environment state, keys).
It is saved as the flat list of its leaves; to load it, the tree structure comes from the code
(`jax.eval_shape` of the initialisation), so a state file only restores into the same configuration.
"""

import hashlib
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# keys that do not change what is trained: a run may be resumed with other values
NOT_TRAINING = {
    "RESUME",
    "CHECKPOINT_INTERVAL",
    "SAVE_PATH",
    "WANDB_MODE",
    "ENTITY",
    "PROJECT",
    "WANDB_LOG_ALL_SEEDS",
}


def config_hash(config) -> str:
    """Hash of the keys that define the training run (not of where it is saved or logged)."""
    relevant = {k: v for k, v in config.items() if k not in NOT_TRAINING and k != "alg"}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_state(path, state, metadata) -> None:
    """Write the leaves of `state` atomically (temporary file, then rename)."""
    leaves = jax.tree_util.tree_leaves(state)
    tensors = {f"leaf_{i:05d}": np.ascontiguousarray(np.asarray(leaf)) for i, leaf in enumerate(leaves)}
    tmp = f"{path}.tmp"
    save_file(tensors, tmp, metadata={k: str(v) for k, v in metadata.items()})
    os.replace(tmp, path)


def read_metadata(path) -> dict:
    with safe_open(path, "np") as f:
        return dict(f.metadata() or {})


def load_state(path, skeleton, expected_hash=None):
    """Restore a state saved by `save_state` into the structure of `skeleton` (ShapeDtypeStructs)."""
    meta = read_metadata(path)
    if expected_hash is not None and meta.get("config_hash") != expected_hash:
        raise ValueError(
            f"{path} was saved with another training configuration "
            f"({meta.get('config_hash')} != {expected_hash}); change it back or start without RESUME"
        )
    spec, treedef = jax.tree_util.tree_flatten(skeleton)
    with safe_open(path, "np") as f:
        keys = sorted(f.keys())
        if len(keys) != len(spec):
            raise ValueError(f"{path} holds {len(keys)} arrays, the run state has {len(spec)}")
        leaves = []
        for key, want in zip(keys, spec):
            array = f.get_tensor(key)
            if array.shape != tuple(want.shape) or array.dtype != np.dtype(want.dtype):
                raise ValueError(f"{path}: {key} is {array.dtype}{array.shape}, expected {want.dtype}{want.shape}")
            leaves.append(jnp.asarray(array))
    return jax.tree_util.tree_unflatten(treedef, leaves)


def write_json(path, payload) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
