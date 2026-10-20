"""Time one training step and one planning solve, then size the experiments from it.

Training: random batches of the geometry of the config, forward + backward +
optimiser step (data loading is not included), median step time, FLOPs and peak
memory, extrapolated to an epoch and to the GPU-hours of the whole run.
Data loading (`--loader-batches N`): the training loader of the config on the
real training file, clips per second against what the step consumes; the epoch
then has the real number of clips.
Planning: one solve with the solver of an evaluation config on random frames,
extrapolated to a task and to a task set. The planning numbers are for the
planner only; stepping the simulator is not included.

Meaningful on the GPU of the real runs; it also runs on the CPU as a smoke test.

    python time_run.py --config-name craftax_hjepa --eval-config-name craftax_plan \
        --batch-size 128 --clips 200000 --num-eval 100 num_levels=2
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

CONFIG_DIR = Path(__file__).parent / "config"


def required_frames(cfg) -> int:
    """Level-1 frames in a batch: what the highest level needs, as the dataset computes it."""
    levels = int(cfg.num_levels)
    frames = 1
    for level in range(1, levels + 1):
        wm = cfg[f"level{level}"].wm
        required = int(wm.history_size) + int(wm.get("rollout_n", 1))
        for upper in range(level, 1, -1):  # expand through the strides of the levels in between
            lv = cfg[f"level{upper}"]
            required = (required - 1) * int(lv.stride) + int(lv.get("window_size", 1))
        frames = max(frames, required)
    return frames


def compose(config_dir: str, name: str, overrides=()):
    import hydra
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR / config_dir), version_base=None):
        return hydra.compose(config_name=name, overrides=list(overrides))


class _Harness(torch.nn.Module):
    """What hjepa_forward expects of the Lightning module."""

    def __init__(self, model, losses):
        super().__init__()
        self.model = model
        for name, loss in losses.items():
            setattr(self, name, loss)

    def log_dict(self, *args, **kwargs):
        pass


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_training(cfg, batch_size: int, steps: int, warmup: int, device) -> dict:
    from hjepa_utils import create_world_model, hjepa_forward

    model, losses = create_world_model(cfg)
    harness = _Harness(model, losses).to(device).train()
    optimizer = torch.optim.AdamW(harness.parameters(), lr=float(cfg.optimizer.lr))
    frames = required_frames(cfg)
    size = int(cfg.img_size)
    num_actions = cfg.level1.wm.get("num_actions")
    frameskip = int(cfg.data.dataset.level1.frameskip)
    batch = {
        "pixels_level1": torch.randn(batch_size, frames, 3, size, size, device=device),
        "action_level1": (
            # one one-hot per frameskip step, concatenated: (batch, frames, frameskip * num_actions)
            torch.nn.functional.one_hot(
                torch.randint(0, int(num_actions), (batch_size, frames, frameskip), device=device), int(num_actions)
            ).float().flatten(-2)
            if num_actions
            else torch.randn(batch_size, frames, int(cfg.level1.wm.action_dim) * int(cfg.data.dataset.level1.frameskip), device=device)
        ),
    }
    bf16 = device.type == "cuda" and "bf16" in str(cfg.trainer.get("precision", ""))

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            out = hjepa_forward(harness, dict(batch), "fit", cfg, normalize_batch=lambda b: None)
        out["loss"].backward()
        optimizer.step()

    for _ in range(warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        step()
        _sync(device)
        times.append(time.perf_counter() - t0)
    with FlopCounterMode(display=False) as counter:
        step()
    return {
        "frames_per_clip": frames,
        "step_seconds": statistics.median(times),
        "samples_per_second": batch_size / statistics.median(times),
        "tflops_per_step": counter.get_total_flops() / 1e12,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
        "parameters_m": sum(p.numel() for p in harness.parameters()) / 1e6,
    }


def time_loading(cfg, cache_dir, batches: int, num_workers: int | None = None) -> dict:
    """The training loader of `cfg` on the real training file, as main_hjepa builds it: clips per second."""
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from data import Compose, build_column_transforms, build_hdf5_dataset, dataset_kwargs

    dataset_cfg = dataset_kwargs(cfg)
    dataset_cfg.pop("val_total_transitions", None)
    dataset = build_hdf5_dataset(dataset_cfg, cache_dir=cache_dir)
    dataset.transform = Compose(*build_column_transforms(cfg, dataset))
    loader_cfg = OmegaConf.to_container(cfg.loader, resolve=True)
    if num_workers is not None:
        loader_cfg["num_workers"] = num_workers
    if not loader_cfg["num_workers"]:  # these need worker processes
        loader_cfg.update(persistent_workers=False, prefetch_factor=None)
    loader = iter(DataLoader(dataset, **loader_cfg, generator=torch.Generator().manual_seed(0)))
    next(loader)  # starts the workers
    t0 = time.perf_counter()
    clips = 0
    for _ in range(batches):
        clips += len(next(loader)["pixels_level1"])
    seconds = time.perf_counter() - t0
    return {"clips_per_second": clips / seconds, "clips_per_epoch": len(dataset), "num_workers": loader_cfg["num_workers"],
            "batches": batches}  # fmt: skip


def time_planning(train_cfg, eval_cfg, num_envs: int, solves: int, device) -> dict:
    from omegaconf import OmegaConf

    from eval_config_utils import _build_policy_plan_config, _is_hierarchical_solver
    from hjepa_utils import create_world_model
    from planning_eval import build_solver
    from stable_worldmodel.solver.solver import num_planning_calls

    model = create_world_model(train_cfg)[0].to(device).eval()
    cfg = OmegaConf.create(OmegaConf.to_container(eval_cfg, resolve=False))
    hierarchical = _is_hierarchical_solver(cfg)
    solvers = cfg.solver.solvers.values() if hierarchical else [cfg.solver]
    for solver_cfg in solvers:
        solver_cfg.device = str(device)
    if hierarchical:
        solver = build_solver(cfg, model)
    else:
        solver = build_solver(cfg, model.get_level(1))
    plan = _build_policy_plan_config(cfg)
    import gymnasium as gym

    solver.configure(action_space=gym.spaces.MultiDiscrete([17] * num_envs), n_envs=num_envs, config=plan)
    size = int(cfg.eval.img_size)
    info = {
        "pixels": torch.randn(num_envs, 1, 3, size, size),
        "goal": torch.randn(num_envs, 1, 3, size, size),
        "action": torch.zeros(num_envs, 1, 17),
    }
    budget = int(cfg.eval.eval_budget)
    kwargs = {"steps_taken": 0, "eval_budget": budget}
    solver.solve(dict(info), **kwargs)  # warmup
    times = []
    for _ in range(solves):
        _sync(device)
        t0 = time.perf_counter()
        solver.solve(dict(info), **kwargs)
        _sync(device)
        times.append(time.perf_counter() - t0)
    with FlopCounterMode(display=False) as counter:
        solver.solve(dict(info), **kwargs)
    level1 = plan.level1 if hierarchical else plan
    interval = int(plan.receding_horizon) * int(level1.action_block)
    per_task = num_planning_calls(eval_budget=budget, replanning_interval=interval)
    median = statistics.median(times)
    return {
        "envs_per_solve": num_envs,
        "solve_seconds": median,
        "tflops_per_solve": counter.get_total_flops() / 1e12,
        "solves_per_task": per_task,
        "seconds_per_task": median / num_envs * per_task,
        "tflops_per_task": counter.get_total_flops() / 1e12 / num_envs * per_task,
        "note": "solves for finished tasks are counted: an upper bound when tasks succeed early",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", required=True, help="training config")
    parser.add_argument("--eval-config-name", help="evaluation config (timing of a solve)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--clips", type=int, default=100_000, help="training clips per epoch, to extrapolate")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-eval", type=int, default=100, help="tasks of an evaluation, to extrapolate")
    parser.add_argument("--num-envs", type=int, default=8, help="environments per timed solve")
    parser.add_argument("--solves", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loader-batches", type=int, default=0, help="time the real training loader on N batches")
    parser.add_argument("--num-workers", type=int, help="loader workers (default: the config's)")
    parser.add_argument("--cache-dir", default=os.environ.get("HJEPA_HOME"), help="where the training file is")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides of the training config, e.g. num_levels=3")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = compose("train", args.config_name, args.overrides)
    batch_size = args.batch_size or int(cfg.loader.batch_size)
    epochs = args.epochs or int(cfg.trainer.max_epochs)
    print(f"device {device}; config {args.config_name}; batch size {batch_size}")
    t = time_training(cfg, batch_size, args.steps, args.warmup, device)
    print(f"\nTraining step: {t['step_seconds']:.3f} s ({t['samples_per_second']:.0f} clips/s), "
          f"{t['tflops_per_step']:.2f} TFLOPs, {t['parameters_m']:.1f} M parameters, "
          f"{t['frames_per_clip']} level-1 frames per clip"
          + (f", peak memory {t['peak_memory_gb']:.1f} GB" if t["peak_memory_gb"] else ""))
    clips = args.clips
    if args.loader_batches:
        cfg.loader.batch_size = batch_size
        d = time_loading(cfg, args.cache_dir, args.loader_batches, args.num_workers)
        clips = d["clips_per_epoch"]
        verdict = "the bottleneck" if d["clips_per_second"] < t["samples_per_second"] else "fast enough"
        print(f"\nData loading: {d['clips_per_second']:.0f} clips/s with {d['num_workers']} workers over {d['batches']} "
              f"batches; the step takes {t['samples_per_second']:.0f} clips/s, so loading is {verdict}. "
              f"The training file has {clips} clips per epoch.")
    steps_per_epoch = clips // batch_size
    epoch = steps_per_epoch * t["step_seconds"]
    print(f"Epoch of {clips} clips: {steps_per_epoch} steps, {epoch / 60:.1f} min. "
          f"{epochs} epochs: {epochs * epoch / 3600:.2f} GPU-hours (data loading not included).")

    if args.eval_config_name:
        eval_cfg = compose("eval", args.eval_config_name)
        p = time_planning(cfg, eval_cfg, args.num_envs, args.solves, device)
        print(f"\nPlanning solve for {p['envs_per_solve']} envs: {p['solve_seconds']:.2f} s, {p['tflops_per_solve']:.2f} TFLOPs; "
              f"{p['solves_per_task']} solves per task.")
        print(f"Per task: {p['seconds_per_task']:.1f} s, {p['tflops_per_task']:.2f} TFLOPs. "
              f"{args.num_eval} tasks: {args.num_eval * p['seconds_per_task'] / 3600:.2f} h on one GPU ({p['note']}).")


if __name__ == "__main__":
    main()
