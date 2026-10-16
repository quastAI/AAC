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
