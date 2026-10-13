"""Planning metrics: what the planner did, how far it got, and whether to trust its plans.

`RecordingPolicy` wraps `WorldModelPolicy`. At every environment step it records
progress toward the required achievements, the latent distance to the goal at
every level, and the executed action. At every replanning it records the
planner's own statistics (CEM improvement and convergence, FLOPs) and, in the
plan diagnostics, executes the chosen plan open-loop in a cloned simulator from
exactly the same state and compares reality with the model's prediction:

* prediction error of the chosen plan against random plans (an optimiser that
  finds model errors picks plans that are wrong *more* than random ones),
* optimism: the real goal distance after the plan against the predicted one,
* ranking: Spearman correlation between predicted and real goal distance over
  the chosen, random and perturbed plans (can the cost order plans at all?),
* support: distance of the predicted latents to a bank of dataset latents.

`summarize_planning` turns the records into metrics and `health.diagnose_planning`
into findings. All diagnostics run in the level-1 latent space, so they work for
flat and hierarchical models.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as logging
from torch.utils.flop_counter import FlopCounterMode
from torchvision import tv_tensors

from eval_tasks import ACHIEVEMENT_NAMES, DIAMOND
from stable_worldmodel.policy import RandomPolicy, WorldModelPolicy

EPS = 1e-12


# --------------------------------------------------------------------------
# small utilities


def mse(a: torch.Tensor, b: torch.Tensor, dim=-1) -> torch.Tensor:
    return (a - b).pow(2).mean(dim)


def spearman(x, y) -> float | None:
    """Rank correlation; None when either side is constant or too short."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def _mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(values)) if values else None


# --------------------------------------------------------------------------
# the solver proxy: captures the last output and counts FLOPs per solve


class RecordingSolver:
    """Delegates to a solver, remembering its last output and FLOPs."""

    def __init__(self, solver):
        self._solver = solver
        self.last: dict | None = None

    def configure(self, **kwargs):
        return self._solver.configure(**kwargs)

    @property
    def action_dim(self):
        return self._solver.action_dim

    @property
    def n_envs(self):
        return self._solver.n_envs

    @property
    def horizon(self):
        return self._solver.horizon

    @property
    def solve_records(self):
        return self._solver.solve_records

    @solve_records.setter
    def solve_records(self, value):
        self._solver.solve_records = value

    def solve(self, info_dict, *args, **kwargs):
        with FlopCounterMode(display=False) as counter:
            outputs = self._solver.solve(info_dict, *args, **kwargs)
        self.last = {"outputs": outputs, "info": info_dict, "flops": float(counter.get_total_flops())}
        return outputs

    __call__ = solve

    def __getattr__(self, name):  # level_solvers, level_models, model, ...
        return getattr(self._solver, name)


# --------------------------------------------------------------------------
# support bank


@torch.no_grad()
def build_support_bank(jepa, h5_path, transform, n_frames: int, seed: int = 0, batch: int = 256):
    """Level-1 latents of random dataset frames, and the median nearest-neighbour distance."""
    import h5py

    device = next(jepa.parameters()).device
    with h5py.File(h5_path, "r") as f:
        total = f["pixels"].shape[0]
        idx = np.sort(np.random.default_rng(seed).choice(total, size=min(n_frames, total), replace=False))
        frames = f["pixels"][idx]
    latents = []
    jepa.eval()
    for s in range(0, len(frames), batch):
        x = torch.stack([transform(tv_tensors.Image(np.transpose(fr, (2, 0, 1)))) for fr in frames[s : s + batch]])
        latents.append(jepa.encode({"pixels": x[:, None].to(device)}, key="pixels")["embed_0"][:, 0].float())
    bank = torch.cat(latents)
    sample = bank[torch.randperm(len(bank), generator=torch.Generator().manual_seed(seed))[:1024]]
    dist = torch.cdist(sample, bank)
    dist[dist == 0] = float("inf")  # a latent is not its own neighbour
    return bank, dist.min(1).values.median().item()


def offsupport_fraction(latents: torch.Tensor, bank: torch.Tensor, scale: float, factor: float = 3.0) -> float:
    """Share of latents whose nearest dataset latent is further than `factor` typical neighbour gaps."""
    d = torch.cdist(latents.reshape(-1, latents.shape[-1]).float(), bank).min(1).values
    return float((d > factor * scale).float().mean())


# --------------------------------------------------------------------------
# records


def new_records(n: int) -> list[dict]:
    return [
        {"step": [], "progress": [], "required": 0, "achievements": [], "flags": [], "terminated": [],
         "dist": {}, "actions": [], "solves": [], "subgoals": []}
        for _ in range(n)
    ]  # fmt: skip


def record_state(records, raw, step, obs_z=None, goal_z=None) -> None:
    """Progress, achievements and (if given) the latent distance to the goal of every task."""
    for i, rec in enumerate(records):
        if rec["terminated"] and rec["terminated"][-1]:
            continue  # the evaluation keeps stepping after success; stop recording
        rec["step"].append(step)
        rec["progress"].append(float(np.asarray(raw["goal_progress"])[i, -1]))
        rec["required"] = int(np.asarray(raw["goal_required"])[i, -1])
        flags = np.asarray(raw["achievements"])[i, -1]
        rec["achievements"].append(int(flags.sum()))
        rec["flags"] = flags.astype(int).tolist()  # the latest state: which achievements are unlocked
        rec["terminated"].append(bool(np.asarray(raw["terminated"])[i, -1]))
        for level in obs_z or {}:
            rec["dist"].setdefault(str(level), []).append(float(mse(obs_z[level][i], goal_z[level][i])))


class RandomRecordingPolicy(RandomPolicy):
    """Uniform random actions, recorded like the planner: the baseline of a task set.

    Tasks defined by progress can be easy (a random agent wakes up, drinks, plants a
    sapling), so every planning result needs this reference on the same tasks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records = None
        self._steps = 0

    def reset(self) -> None:
        self.records, self._steps = None, 0

    def get_action(self, info_dict, **kwargs):
        n = self.env.num_envs
        if self.records is None:
            self.records = new_records(n)
        record_state(self.records, dict(info_dict), self._steps)
        action = super().get_action(info_dict, **kwargs)
        for i in range(n):
            self.records[i]["actions"].append(int(np.asarray(action).reshape(n, -1)[i, 0]))
        self._steps += 1
        return action

    def finish(self, info_dict) -> None:
        """The state after the last action: the evaluation calls it once the budget is spent."""
        if self.records is not None:
            record_state(self.records, dict(info_dict), self._steps)


# --------------------------------------------------------------------------
# the recording policy


class RecordingPolicy(WorldModelPolicy):
    def __init__(self, solver, config, *, metrics_cfg=None, bank=None, **kwargs):
        super().__init__(solver, config, **kwargs)
        self.mcfg = dict(metrics_cfg or {})
        self.bank = bank  # (latents, nn_scale) or None
        inner = solver._solver if isinstance(solver, RecordingSolver) else solver
        self.levels = list(getattr(inner, "level_models", None) or [inner.model])
        self.jepa1 = self.levels[0]
        self.hierarchical = hasattr(inner, "level_models")
        self.inner = inner
        self.records: list[dict] | None = None
        self._rng = np.random.default_rng(int(self.mcfg.get("seed", 0)))
        self._sims: dict[int, Any] = {}
        self._replans = 0
        self._diag_disabled = False
        self._pending: dict[int, dict] = {}

    # ------------------------------------------------------------------ setup

    def reset(self) -> None:
        super().reset()
        self.records = None
        self._pending = {}
        self._replans = 0

    @property
    def device(self):
        return next(self.jepa1.parameters()).device

    def _solvers(self) -> dict:
        return getattr(self.inner, "level_solvers", None) or {1: self.inner}

    def _ensure(self, n: int) -> None:
        if self.records is None:
            self.records = new_records(n)

    # --------------------------------------------------------------- encoding

    @torch.no_grad()
    def _encode_levels(self, prepared: dict) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Latents of the current frame and of the goal at every level, (B, D) each."""
        was = [m.training for m in self.levels]
        for m in self.levels:
            m.eval()
        try:
            if hasattr(self.inner, "_encode_all_levels"):
                obs = self.inner._encode_all_levels(info_dict=prepared, key="pixels")
                goal = self.inner._encode_all_levels(info_dict=prepared, key="goal")
                return (
                    {int(k.split("_")[1]): v[:, -1].float() for k, v in obs.items()},
                    {int(k.split("_")[1]): v[:, -1].float() for k, v in goal.items()},
                )
            enc = lambda x: self.jepa1.encode({"pixels": x.to(self.device)}, key="pixels")["embed_0"][:, -1].float()  # noqa: E731
            return {1: enc(prepared["pixels"])}, {1: enc(prepared["goal"])}
        finally:
            for m, w in zip(self.levels, was):
                m.train(w)

    def _transform_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Frames (T, H, W, 3) uint8 -> model input, as the policy prepares them."""
        tf = self.transform["pixels"]
        return torch.stack([tf(tv_tensors.Image(np.transpose(f, (2, 0, 1)))) for f in frames])

    # ------------------------------------------------------------- the action

    def get_action(self, info_dict: dict, **kwargs):
        n = self.env.num_envs
        self._ensure(n)
        raw = dict(info_dict)  # the policy prepares info_dict in place
        prepared = self._prepare_info(dict(raw))
        replanning = len(self._action_buffer) == 0
        step = self._steps_taken

        obs_z, goal_z = self._encode_levels(prepared)
        self._record_state(raw, obs_z, goal_z, step)
        if replanning:
            self._check_subgoals(obs_z, step)
            before = {level: len(s.solve_records) for level, s in self._solvers().items()}

        action = super().get_action(info_dict, **kwargs)

        if replanning:
            self._replans += 1
            self._after_solve(prepared, obs_z, goal_z, step, before, n)
        for i in range(n):
            self.records[i]["actions"].append(int(np.asarray(action).reshape(n, -1)[i, 0]))
        return action

    # -------------------------------------------------------------- recording

    def _record_state(self, raw, obs_z, goal_z, step) -> None:
        record_state(self.records, raw, step, obs_z, goal_z)

    def finish(self, info_dict) -> None:
        """The state after the last action: a success or unlock on the final step is recorded too."""
        if self.records is not None:
            record_state(self.records, dict(info_dict), self._steps_taken)

    def _active(self, i: int) -> bool:
        rec = self.records[i]
        return not (rec["terminated"] and rec["terminated"][-1])

    def _after_solve(self, prepared, obs_z, goal_z, step, before, n) -> None:
        cap = getattr(self.solver, "last", None)
        outputs = cap["outputs"] if cap else {}
        flops = (cap or {}).get("flops", 0.0) / n
        stats = {}
        for level, solver in self._solvers().items():
            records = solver.solve_records
            if len(records) > before.get(level, 0):
                stats[level] = records[-1]
        for i in range(n):
            if not self._active(i):
                continue
            entry = {"step": step, "flops": flops, "levels": {}}
            for level, r in stats.items():
                entry["levels"][str(level)] = {
                    "horizon": r.get("horizon"),
                    "wall_clock": (r.get("wall_clock_per_env") or [None] * n)[i],
                    "best_cost": (r.get("best_cost") or [None] * n)[i],
                    "first_cost": (r.get("first_cost") or [None] * n)[i],
                    "entropy": r.get("final_entropy"),
                }
            self.records[i]["solves"].append(entry)

        levels_out = outputs.get("levels") or {}
        top = max((l for l in levels_out if l >= 2), default=None)
        if top is not None:
            pred = levels_out[top]["predictions"].get("predicted_embed_0")
            if pred is not None and pred.shape[1] > 1:
                for i in range(n):
                    if self._active(i):
                        self._pending[i] = {
                            "level": top, "sub": pred[i, 1].float(), "ctx": pred[i, 0].float(), "step": step,
                            "due": self._env_steps_per_step(top),
                        }  # fmt: skip

        every = int(self.mcfg.get("diag_every", 1))
        if every and self._replans % every == 0 and not self._diag_disabled and "actions" in outputs:
            self._plan_diagnostics(prepared, outputs["actions"], goal_z[1], n)

    def _env_steps_per_step(self, level: int) -> int:
        """Environment steps one step of `level` spans (the subgoal lies this far ahead)."""
        return max(1, math.prod(int(m.temporal_stride) for m in self.levels[1:level]))

    def _check_subgoals(self, obs_z, step) -> None:
        for i, p in list(self._pending.items()):
            if p["level"] not in obs_z or not self._active(i):
                self._pending.pop(i, None)
                continue
            now = obs_z[p["level"]][i].cpu()
            err, start = float(mse(now, p["sub"].cpu())), float(mse(p["ctx"].cpu(), p["sub"].cpu()))
            # The subgoal lies `due` environment steps ahead but the plan is replaced every
            # `receding_horizon` steps, so it is checked after `elapsed` < due: `expected` is
            # how far a straight approach would have come by then.
            elapsed = step - p["step"]
            self.records[i]["subgoals"].append(
                {"elapsed": elapsed, "progress": 1.0 - err / (start + EPS), "level": p["level"],
                 "expected": min(1.0, elapsed / p["due"])}
            )  # fmt: skip
            self._pending.pop(i)

    # ------------------------------------------------------ plan diagnostics

    def _sim_env(self, i):
        env = self.env.envs[i].unwrapped
        if not hasattr(env, "get_state"):
            return None, None
        if i not in self._sims:
            from stable_worldmodel.envs.craftax.env import CraftaxClassicEnv

            self._sims[i] = CraftaxClassicEnv()
        return env, self._sims[i]

    @torch.no_grad()
    def _real_latents(self, sim, env, plans: np.ndarray) -> torch.Tensor:
        """Run each plan (K, H) from the env's current state; encode the frames -> (K, H, D)."""
        snapshot, seed = env.get_state(), env._seed
        frames = []
        for plan in plans:
            sim.reset(seed=seed)
            sim._set_state(snapshot)
            rendered = []
            for a in plan:
                sim.step(int(a))
                rendered.append(sim.render())
            frames.append(np.stack(rendered))
        x = torch.stack([self._transform_frames(f) for f in frames]).to(self.device)
        self.jepa1.eval()
        return self.jepa1.encode({"pixels": x}, key="pixels")["embed_0"].float()

    @torch.no_grad()
    def _predicted_latents(self, prepared, i, plans: np.ndarray) -> torch.Tensor:
        """The model's rollout of each plan from the current frame -> (K, H+1, D)."""
        k, h = plans.shape
        pixels = prepared["pixels"][i : i + 1].unsqueeze(1).expand(1, k, *prepared["pixels"].shape[1:])
        info = {"pixels": pixels.to(self.device)}
        candidates = F.one_hot(torch.as_tensor(plans, device=self.device), 17).float()[None]
        self.jepa1.eval()
        return self.jepa1.rollout(info, candidates)["predicted_embed_0"][0].float()

    def _candidates(self, plan: np.ndarray, k_total: int) -> np.ndarray:
        """The chosen plan, random plans and one- or two-action perturbations of it."""
        h = len(plan)
        n_random = max((k_total - 1) // 2, 1)
        plans = [plan]
        plans += [self._rng.integers(0, 17, h) for _ in range(n_random)]
        while len(plans) < k_total:
            p = plan.copy()
            for _ in range(self._rng.integers(1, 3)):
                p[self._rng.integers(0, h)] = self._rng.integers(0, 17)
            plans.append(p)
        return np.stack(plans)

    @torch.no_grad()
    def _plan_diagnostics(self, prepared, plan_actions, goal1, n) -> None:
        k_total = int(self.mcfg.get("rank_candidates", 8))
        if plan_actions.shape[-1] != 1:
            logging.warning("planning metrics: plan diagnostics need action_block 1; switched off")
            self._diag_disabled = True
            return
        for i in range(n):
            if not self._active(i):
                continue
            env, sim = self._sim_env(i)
            if env is None:
                logging.warning("planning metrics: the environment has no state API; plan diagnostics off")
                self._diag_disabled = True
                return
            plan = plan_actions[i, :, 0].cpu().numpy()
            plans = self._candidates(plan, max(k_total, 3))
            pred = self._predicted_latents(prepared, i, plans)  # (K, H+1, D)
            real = self._real_latents(sim, env, plans)  # (K, H, D)
            goal, z0 = goal1[i], pred[0, 0]
            err = mse(pred[:, 1:], real).mean(1)  # (K,) over steps and dims
            pred_goal, real_goal = mse(pred[:, -1], goal), mse(real[:, -1], goal)
            n_random = max((len(plans) - 1) // 2, 1)
            entry = {
                "horizon": int(plan.shape[0]),
                "pred_err": float(err[0]),
                "pred_err_curve": mse(pred[0, 1:], real[0]).tolist(),
                "pred_err_random": float(err[1 : 1 + n_random].mean()),
                "goal0": float(mse(z0, goal)),
                "pred_goal": float(pred_goal[0]),
                "real_goal": float(real_goal[0]),
                "pred_goal_random": float(pred_goal[1 : 1 + n_random].mean()),
                "real_goal_random": float(real_goal[1 : 1 + n_random].mean()),
                "rank_corr": spearman(pred_goal.cpu().numpy(), real_goal.cpu().numpy()),
            }
            if self.bank is not None:
                bank, scale = self.bank
                entry["offsupport_plan"] = offsupport_fraction(pred[0, 1:], bank, scale)
                entry["offsupport_random"] = offsupport_fraction(pred[1 : 1 + n_random, 1:], bank, scale)
                entry["offsupport_real"] = offsupport_fraction(real[0], bank, scale)
            self.records[i]["solves"][-1]["plan"] = entry


# --------------------------------------------------------------------------
# summary


def _task_groups(records, task_info):
    """Index lists of tasks by required-achievement count (and depth/offset if known)."""
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault(f"required_{r['required']}", []).append(i)
        if task_info and i < len(task_info):
            for key in ("offset", "depth"):
                if key in task_info[i]:
                    groups.setdefault(f"{key}_{task_info[i][key]}", []).append(i)
    return groups


def summarize_planning(records: list[dict], task_info: list[dict] | None = None) -> dict:
    """Aggregate per-task records into the planning metrics."""
    n = len(records)
    if not n:
        return {}
    final_progress = [r["progress"][-1] if r["progress"] else 0.0 for r in records]
    success = [bool(r["terminated"] and r["terminated"][-1]) for r in records]
    first_progress = [next((s for s, p in zip(r["step"], r["progress"]) if p > 0), None) for r in records]

    out: dict[str, Any] = {
        "num_tasks": n,
        "success_rate": float(np.mean(success)),
        "progress_mean": float(np.mean(final_progress)),
        "progress_any": float(np.mean([p > 0 for p in final_progress])),
        "steps_to_first_progress": _mean([s for s in first_progress if s is not None]),
    }

    groups = _task_groups(records, task_info)
    out["by_group"] = {
        g: {
            "n": len(idx),
            "success": float(np.mean([success[i] for i in idx])),
            "progress": float(np.mean([final_progress[i] for i in idx])),
        }
        for g, idx in sorted(groups.items())
    }

    # which of the required achievements were unlocked (protocol B: the diamond rate)
    if task_info:
        unlocked: dict[int, list[bool]] = {}
        for i, r in enumerate(records):
            if i < len(task_info) and r.get("flags"):
                for a in task_info[i].get("required", []):
                    unlocked.setdefault(int(a), []).append(bool(r["flags"][a]))
        out["by_achievement"] = {
            ACHIEVEMENT_NAMES[a]: {"n": len(v), "unlocked": float(np.mean(v))} for a, v in sorted(unlocked.items())
        }
        if DIAMOND in unlocked:
            out["diamond"] = {"n": len(unlocked[DIAMOND]), "rate": float(np.mean(unlocked[DIAMOND]))}

    # latent distance to the goal, per level
    dist = {}
    levels = sorted({int(l) for r in records for l in r["dist"]})
    for level in levels:
        series = [r["dist"][str(level)] for r in records if r["dist"].get(str(level))]
        ratio = [s[-1] / (s[0] + EPS) for s in series if len(s) > 1]
        dist[level] = {
            "final_over_initial": _mean(ratio),
            "decreased_frac": float(np.mean([x < 1 for x in ratio])) if ratio else None,
            "min_over_initial": _mean([min(s) / (s[0] + EPS) for s in series if len(s) > 1]),
        }
    out["dist"] = dist

    # planner health
    solves = [s for r in records for s in r["solves"]]
    per_level = {}
    for level in sorted({int(l) for s in solves for l in s["levels"]}):
        rows = [s["levels"][str(level)] for s in solves if str(level) in s["levels"]]
        improvement = [
            (r["first_cost"] - r["best_cost"]) / (abs(r["first_cost"]) + EPS)
            for r in rows
            if r.get("first_cost") is not None and r.get("best_cost") is not None
        ]
        per_level[level] = {
            "solves": len(rows),
            "cem_improvement": _mean(improvement),
            "cem_entropy_final": _mean([r.get("entropy") for r in rows]),
            "wall_clock": _mean([r.get("wall_clock") for r in rows]),
            "horizon": _mean([r.get("horizon") for r in rows]),
        }
    out["planner"] = per_level

    # compute: FLOPs per task until it succeeded (or the budget ended)
    flops = [sum(s["flops"] for s in r["solves"]) for r in records]
    out["compute"] = {
        "tflops_per_task": float(np.mean(flops) / 1e12) if flops else None,
        "tflops_per_solve": _mean([s["flops"] / 1e12 for s in solves]),
        "solves_per_task": float(np.mean([len(r["solves"]) for r in records])),
    }

    # plan reliability
    plans = [s["plan"] for s in solves if "plan" in s]
    if plans:
        ratio = [p["pred_err"] / (p["pred_err_random"] + EPS) for p in plans]
        opt = [p["real_goal"] / (p["pred_goal"] + EPS) for p in plans]
        opt_random = [p["real_goal_random"] / (p["pred_goal_random"] + EPS) for p in plans]
        out["plans"] = {
            "n": len(plans),
            "pred_err": _mean([p["pred_err"] for p in plans]),
            "pred_err_random": _mean([p["pred_err_random"] for p in plans]),
            "err_ratio_vs_random": _mean(ratio),
            "optimism_chosen": _mean(opt),
            "optimism_random": _mean(opt_random),
            "rank_corr": _mean([p["rank_corr"] for p in plans]),
            "pred_gain": _mean([1 - p["pred_goal"] / (p["goal0"] + EPS) for p in plans]),
            "real_gain": _mean([1 - p["real_goal"] / (p["goal0"] + EPS) for p in plans]),
        }
        horizon = max(len(p["pred_err_curve"]) for p in plans)
        curves = [p["pred_err_curve"] for p in plans if len(p["pred_err_curve"]) == horizon]
        out["plans"]["err_curve"] = np.mean(curves, axis=0).tolist()
        for key in ("offsupport_plan", "offsupport_random", "offsupport_real"):
            if key in plans[0]:
                out["plans"][key] = _mean([p[key] for p in plans])

    # actions
    counts = [Counter(r["actions"]) for r in records if r["actions"]]
    if counts:
        shares = [max(c.values()) / sum(c.values()) for c in counts]
        out["actions"] = {
            "max_share_mean": float(np.mean(shares)),
            "degenerate_frac": float(np.mean([s > 0.9 for s in shares])),
            "noop_share": float(np.mean([c.get(0, 0) / sum(c.values()) for c in counts])),
            "distinct_mean": float(np.mean([len(c) for c in counts])),
        }

    # subgoals
    subgoals = [g for r in records for g in r["subgoals"]]
    if subgoals:
        out["subgoals"] = {
            "n": len(subgoals),
            "progress_mean": _mean([g["progress"] for g in subgoals]),
            # at least half of the way a straight approach would have gone in the elapsed time
            "reached_frac": float(np.mean([g["progress"] >= 0.5 * g.get("expected", 1.0) for g in subgoals])),
            "expected_mean": _mean([g.get("expected", 1.0) for g in subgoals]),
        }
    return out
