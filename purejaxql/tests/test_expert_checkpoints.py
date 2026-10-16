"""Intermediate checkpoints and resume of the PQN-RNN expert training (tiny run on CPU)."""

import json
from pathlib import Path

import numpy as np
import pytest
from flax.traverse_util import flatten_dict

pytest.importorskip("craftax")
pytest.importorskip("wandb")
from omegaconf import OmegaConf  # noqa: E402

from purejaxql import pqn_rnn_craftax as expert  # noqa: E402
from purejaxql.utils import checkpoint as ckpt  # noqa: E402
from purejaxql.utils.save_load import load_params  # noqa: E402

CONFIG_DIR = Path(expert.__file__).parent / "config"
UPDATES = 6


def tiny_config(save_path, **overrides):
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "config.yaml"))
    alg = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "alg" / "pqn_rnn_craftax_classic.yaml"))
    alg.update(
        NUM_ENVS=4, NUM_STEPS=8, NUM_MINIBATCHES=2, NUM_EPOCHS=1, HIDDEN_SIZE=16,
        TOTAL_TIMESTEPS=UPDATES * 4 * 8, TOTAL_TIMESTEPS_DECAY=UPDATES * 4 * 8,
        USE_OPTIMISTIC_RESETS=False, TEST_DURING_TRAINING=True, TEST_INTERVAL=0.5,
        TEST_NUM_ENVS=2, TEST_NUM_STEPS=20, SAVE_PATH=str(save_path), CHECKPOINT_INTERVAL=0.5, RESUME=False,
    )  # fmt: skip
    for key, value in overrides.items():
        (alg if key in alg else config)[key] = value
    config["alg"] = {**config["alg"], **alg}  # single_run lays config["alg"] over the top level
    return config


def final_policy(save_path):
    (path,) = Path(save_path).rglob("*_vmap0.safetensors")
    return load_params(path)


def test_snapshots_and_final_policy(tmp_path):
    expert.single_run(tiny_config(tmp_path))
    snapshots = sorted(tmp_path.rglob("*_vmap0_u*.safetensors"))
    assert [p.stem.rsplit("_u", 1)[1] for p in snapshots] == ["0000003", "0000006"]
    info = json.loads(snapshots[-1].with_suffix(".json").read_text())
    assert info["n_updates"] == UPDATES and info["env_steps"] == UPDATES * 4 * 8
    assert info["test"], "the latest greedy-test metrics are stored next to the snapshot"
    variables = final_policy(tmp_path)
    assert set(variables) == {"params", "batch_stats"}
    assert list(tmp_path.rglob("*_state.safetensors"))


def test_resume_gives_the_same_policy(tmp_path, monkeypatch):
    straight, killed = tmp_path / "straight", tmp_path / "killed"
    expert.single_run(tiny_config(straight))

    real_save = expert._save_checkpoint
    calls = []

    def die_after_first(*args, **kwargs):
        real_save(*args, **kwargs)
        calls.append(1)
        raise KeyboardInterrupt("wall-time kill")

    monkeypatch.setattr(expert, "_save_checkpoint", die_after_first)
    with pytest.raises(KeyboardInterrupt):
        expert.single_run(tiny_config(killed))
    monkeypatch.setattr(expert, "_save_checkpoint", real_save)
    assert not list(killed.rglob("*_vmap0.safetensors")), "the run was cut before the end"

    expert.single_run(tiny_config(killed, RESUME=True))
    a, b = flatten_dict(final_policy(straight)), flatten_dict(final_policy(killed))
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]), err_msg=key)


def test_resume_refuses_another_configuration(tmp_path, monkeypatch):
    real_save = expert._save_checkpoint

    def die(*args, **kwargs):
        real_save(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(expert, "_save_checkpoint", die)
    with pytest.raises(KeyboardInterrupt):
        expert.single_run(tiny_config(tmp_path))
    monkeypatch.setattr(expert, "_save_checkpoint", real_save)
    with pytest.raises(ValueError, match="another training configuration"):
        expert.single_run(tiny_config(tmp_path, RESUME=True, LR=0.5))


def test_config_hash_ignores_where_it_is_saved():
    a = {"LR": 1.0, "SAVE_PATH": "x", "RESUME": False, "alg": {"LR": 1.0}}
    b = {"LR": 1.0, "SAVE_PATH": "y", "RESUME": True, "alg": {"LR": 1.0}}
    assert ckpt.config_hash(a) == ckpt.config_hash(b)
    assert ckpt.config_hash(a) != ckpt.config_hash({**a, "LR": 2.0})


def chunk_setup(tmp_path, num_seeds=1):
    import jax

    config = tiny_config(tmp_path, TEST_NUM_STEPS=50)
    config = {**config, **config["alg"]}
    train = expert.make_train(config)
    rngs = jax.random.split(jax.random.PRNGKey(0), num_seeds)
    init = jax.jit(jax.vmap(lambda r: train(r, num_updates=0)["runner_state"]))
    return train, rngs, init(rngs)


def test_the_greedy_test_is_a_cond_not_a_select_for_one_seed(tmp_path):
    # Under vmap the cond on n_updates gets a batched predicate and becomes a select: the test (TEST_NUM_STEPS
    # sequential steps) would then run in every update. chunk_fn keeps the real cond for one seed.
    import re

    import jax

    count = lambda fn, rngs, state: len(re.findall(r"\bcond\[", str(jax.make_jaxpr(fn)(rngs, state))))  # noqa: E731
    train, rngs, state = chunk_setup(tmp_path)
    assert count(expert.chunk_fn(train, 1, 2), rngs, state) == 1
    many, rngs2, state2 = chunk_setup(tmp_path / "two", num_seeds=2)
    assert count(expert.chunk_fn(many, 2, 2), rngs2, state2) == 0  # the known limit of several seeds


def test_the_unbatched_chunk_gives_the_same_result_as_the_vmapped_one(tmp_path):
    import jax

    train, rngs, state = chunk_setup(tmp_path)
    vmapped = jax.vmap(lambda r, s: train(r, runner_state=s, num_updates=2)["runner_state"])
    a = jax.jit(expert.chunk_fn(train, 1, 2))(rngs, state)
    b = jax.jit(vmapped)(rngs, state)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert x.shape == y.shape
        np.testing.assert_allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float), rtol=1e-5, atol=1e-6)
