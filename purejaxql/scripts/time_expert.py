"""Time the PQN-RNN expert training: seconds per update, steps per second, the greedy test, the hours of a run.

    python scripts/time_expert.py                  # the Craftax-Classic configuration as it is
    python scripts/time_expert.py NUM_ENVS=2048    # any KEY=value of the configuration

Runs a few updates of the real configuration (config.yaml plus alg/pqn_rnn_craftax_classic.yaml) without checkpoints or
wandb. The updates are timed without the greedy test; the test (TEST_NUM_ENVS x TEST_NUM_STEPS sequential steps) is
timed on one update on its own. The projection adds `1 / TEST_INTERVAL` tests to the updates. Writing the snapshots
and the resumable state (about the size of the stored transitions, every CHECKPOINT_INTERVAL) is not in it: measure it
on the volume you train on. The first call of every function compiles, so only the second call is timed.
"""

import argparse
import time
from pathlib import Path

import jax
from omegaconf import OmegaConf

from purejaxql import pqn_rnn_craftax as expert

CONFIG_DIR = Path(expert.__file__).parent / "config"
TOTALS = (1e8, 3e8, 5e8, 1e9)  # run lengths (environment steps) of the projection


def load_config(overrides=(), alg="pqn_rnn_craftax_classic") -> dict:
    """The flat configuration the training script runs with, with `KEY=value` overrides on top."""
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "config.yaml"))
    config["alg"] = {**config["alg"], **OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "alg" / f"{alg}.yaml"))}
    config = {**config, **config["alg"], **OmegaConf.to_container(OmegaConf.from_dotlist(list(overrides)))}
    config.update(SAVE_PATH=None, WANDB_MODE="disabled", NUM_SEEDS=1)
    return config


def seconds(fn, *args) -> float:
    """Wall time of the second call of `fn` (the first one compiles)."""
    jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - start


def time_chunk(config: dict, updates: int, test: bool) -> float:
    """Seconds for `updates` updates, with or without a greedy test in each of them."""
    steps = config["NUM_STEPS"] * config["NUM_ENVS"]
    config = {**config, "TOTAL_TIMESTEPS": updates * steps, "TOTAL_TIMESTEPS_DECAY": updates * steps}
    config.update(TEST_DURING_TRAINING=test, TEST_INTERVAL=1.0 / updates + 1e-9)  # a test in every update
    train = expert.make_train(config)
    rngs = jax.random.split(jax.random.PRNGKey(0), 1)
    state = jax.block_until_ready(jax.jit(jax.vmap(lambda r: train(r, num_updates=0)["runner_state"]))(rngs))
    return seconds(jax.jit(expert.chunk_fn(train, 1, updates)), rngs, state)


def project(t_update: float, t_test: float, steps_per_update: int, totals=TOTALS, test_interval=0.05) -> list[dict]:
    """Hours of a run of each length in `totals` (steps): the updates plus 1 / test_interval greedy tests."""
    rows = []
    for total in totals:
        updates = int(total // steps_per_update)
        rows.append({"steps": total, "updates": updates, "hours": (updates * t_update + t_test / test_interval) / 3600,
                     "test_hours": t_test / test_interval / 3600})  # fmt: skip
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("overrides", nargs="*", help="KEY=value of the configuration, e.g. NUM_ENVS=2048")
    p.add_argument("--updates", type=int, default=5, help="updates timed together")
    p.add_argument("--alg", default="pqn_rnn_craftax_classic")
    args = p.parse_args(argv)
    config = load_config(args.overrides, args.alg)
    steps = config["NUM_STEPS"] * config["NUM_ENVS"]
    device = jax.devices()[0]
    print(f"device {device} | {config['NUM_ENVS']} envs x {config['NUM_STEPS']} steps = {steps} steps per update")
    if device.platform == "cpu":
        print("WARNING: no GPU, the numbers say nothing about the pod")

    t_train = time_chunk(config, args.updates, test=False) / args.updates
    t_with_test = time_chunk(config, 1, test=True)
    t_one = time_chunk(config, 1, test=False)
    t_test = max(t_with_test - t_one, 0.0)
    stats = device.memory_stats() or {}
    print(f"\nper update      {t_train:.3f} s  ({steps / t_train:,.0f} environment steps per second)")
    print(f"one greedy test {t_test:.1f} s  ({config['TEST_NUM_ENVS']} envs x {config['TEST_NUM_STEPS']} steps), "
          f"{1 / config['TEST_INTERVAL']:.0f} tests in a run")  # fmt: skip
    if "peak_bytes_in_use" in stats:
        print(
            f"peak GPU memory {stats['peak_bytes_in_use'] / 2**30:.1f} GiB of {stats.get('bytes_limit', 0) / 2**30:.1f}"
        )
    print("\n| run length (steps) | updates | hours | of which tests |\n|---|---|---|---|")
    for r in project(t_train, t_test, steps, test_interval=config["TEST_INTERVAL"]):
        print(f"| {r['steps']:.0e} | {r['updates']} | {r['hours']:.1f} | {r['test_hours']:.2f} |")
    print("\nsnapshots and the resumable state are not included (measure them on the volume you train on)")


if __name__ == "__main__":
    main()
