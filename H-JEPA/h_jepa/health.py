"""Health metrics and rule-based findings for training, planning and probing.

Everything here is a pure function of tensors or plain dicts, so it can be unit
tested and reused by the Lightning callbacks, the planning evaluation and the
probing report.

Metrics answer two questions. Is the model collapsed (representation, VQ
codebook, gradients)? And if it is not collapsed, does it still fail
(predictor no better than copying the last latent, ignores actions, drifts,
cost geometry useless for planning, IDM at chance)? `diagnose_*` turns the
metrics into `Finding`s with a severity and a hint on what to look at. The
default thresholds are heuristics set before the first run: adjust them in
`Thresholds` once real numbers exist.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

EPS = 1e-12


# --------------------------------------------------------------------------
# thresholds and findings


@dataclass
class Thresholds:
    # representation
    erank_frac_critical: float = 0.02  # effective rank / dim
    erank_frac_warn: float = 0.10
    std_mean_critical: float = 1e-4  # per-dimension std of the latents
    dead_dim_frac_warn: float = 0.25  # dims with std < 1% of the largest
    static_step_ratio: float = 1e-3  # latent change per step / unrelated-sample distance
    incoherent_step_ratio: float = 0.8
    # dynamics
    pred_over_copy_warn: float = 0.95  # predictor mse / copy-last-latent mse
    pred_over_copy_critical: float = 1.10
    action_lift_critical: float = 0.005  # (shuffled - true) / shuffled prediction error
    action_lift_warn: float = 0.05
    drift_ratio_warn: float = 1.0  # open-loop error / copy error at the longest horizon
    history_mismatch_ratio: float = 2.0  # planning-history drift / training-history drift
    # cost geometry
    saturation_gap_warn: int = 2  # distance to the goal stops growing within this many steps
    monotone_frac_warn: float = 0.7
    # vq
    vq_perplexity_norm_critical: float = 0.02  # perplexity / codes
    vq_perplexity_norm_warn: float = 0.15
    vq_dead_frac_warn: float = 0.5
    # idm
    idm_lift_critical: float = 0.02  # (acc - chance) / (1 - chance)
    idm_lift_warn: float = 0.10
    # optimisation
    grad_explosion_factor: float = 20.0  # raw grad norm / its running median
    grad_vanishing: float = 1e-7
    loss_improvement_warn: float = 0.05  # relative val-loss improvement over training
    overfit_ratio_warn: float = 1.5  # val loss / train loss
    spike_factor: float = 3.0  # a step loss above this times the running median
    spike_count_warn: int = 5
    # planning
    plan_gap_ratio_warn: float = 1.5  # error of chosen plans / error of random plans
    plan_rank_corr_warn: float = 0.3  # predicted vs real cost ranking of candidates
    support_offsupport_warn: float = 0.25  # share of predicted latents far from the data
    degenerate_action_share: float = 0.9  # one action in this share of executed steps
    subgoal_reach_warn: float = 0.2
    # probing
    probe_group_r2_warn: float = 0.1


@dataclass
class Finding:
    severity: str  # 'critical' | 'warn' | 'info'
    code: str
    level: int | None
    message: str
    hint: str

    def line(self) -> str:
        where = "" if self.level is None else f" [level {self.level}]"
        return f"{self.severity.upper():8s}{where} {self.code}: {self.message} -> {self.hint}"


SEVERITY_ORDER = {"critical": 0, "warn": 1, "info": 2}


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.level or 0, f.code))


def has_critical(findings: list[Finding]) -> bool:
    return any(f.severity == "critical" for f in findings)


# --------------------------------------------------------------------------
# helpers


@contextlib.contextmanager
def eval_mode(module: torch.nn.Module):
    """Temporarily put a module in eval mode (no dropout, frozen BatchNorm statistics)."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def _flat(e: torch.Tensor) -> torch.Tensor:
    return e.detach().float().reshape(-1, e.shape[-1])


def _num(x) -> float | None:
    return None if x is None else float(x)


# --------------------------------------------------------------------------
# representation metrics (embeddings of shape (B, T, D))


def effective_rank(e: torch.Tensor) -> dict[str, float]:
    """Exponential of the entropy of the normalised singular values of the centred latents."""
    x = _flat(e)
    x = x - x.mean(0)
    if x.shape[0] < 2:
        return {"erank": float("nan"), "erank_frac": float("nan")}
    s = torch.linalg.svdvals(x).pow(2)
    p = s / s.sum().clamp_min(EPS)
    erank = torch.exp(-(p * (p + EPS).log()).sum()).item()
    return {"erank": erank, "erank_frac": erank / x.shape[1]}


def dim_stats(e: torch.Tensor) -> dict[str, float]:
    x = _flat(e)
    std = x.std(0, unbiased=False)
    top = std.max().clamp_min(EPS)
    return {
        "std_mean": std.mean().item(),
        "std_min": std.min().item(),
        "std_max": std.max().item(),
        "dead_dim_frac": ((std < 0.01 * top) | (std < 1e-6)).float().mean().item(),
        "nonfinite_frac": (~torch.isfinite(x)).float().mean().item(),
    }


def within_clip_share(e: torch.Tensor) -> float | None:
    """Share of the latent variance that lies within clips (over time) rather than between them."""
    if e.dim() != 3 or e.shape[1] < 2:
        return None
    e = e.detach().float()
    total = e.reshape(-1, e.shape[-1]).var(0, unbiased=False).sum()
    within = e.var(1, unbiased=False).mean(0).sum()
    return (within / total.clamp_min(EPS)).item()


def step_to_scale(e: torch.Tensor) -> float | None:
    """Latent change per step relative to the distance between unrelated samples.

    About 0: the latent does not move (static). About 1: consecutive latents
    are as different as random ones (no temporal coherence).
    """
    if e.dim() != 3 or e.shape[1] < 2 or e.shape[0] < 2:
        return None
    e = e.detach().float()
    step = (e[:, 1:] - e[:, :-1]).pow(2).mean()
    scale = (e - e.roll(1, dims=0)).pow(2).mean()
    return (step / scale.clamp_min(EPS)).item()


# --------------------------------------------------------------------------
# dynamics metrics


def teacher_forced_metrics(
    jepa: Any,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history: int,
    generator: torch.Generator | None = None,
) -> dict[str, float | None]:
    """One-step prediction against two trivial baselines.

    `copy`: predict the previous latent. `shuffled`: right context, actions
    taken from other clips. A predictor that does not clearly beat both has not
    learned action-conditioned dynamics.
    """
    if emb.shape[1] < history + 1:
        return {}
    ctx, act = emb[:, :history], act_emb[:, :history]
    target = emb[:, 1 : history + 1]
    with torch.no_grad(), eval_mode(jepa.predictor):
        mse = F.mse_loss(jepa.predict(ctx, act), target).item()
        copy = F.mse_loss(ctx, target).item()
        shuffled = lift = None
        if emb.shape[0] > 1:
            perm = torch.randperm(emb.shape[0], generator=generator).to(emb.device)
            shuffled = F.mse_loss(jepa.predict(ctx, act[perm]), target).item()
            lift = (shuffled - mse) / (shuffled + EPS)
    return {
        "pred_mse": mse,
        "copy_mse": copy,
        "pred_over_copy": mse / (copy + EPS),
        "shuffled_mse": shuffled,
        "action_lift": lift,
    }


def drift_curve(
    jepa: Any,
    emb: torch.Tensor,
    act_emb: torch.Tensor,
    history: int,
    horizon: int,
) -> dict[str, Any]:
    """Open-loop rollout error against the true latents, with the true actions.

    `history` is the number of context frames the predictor sees at every step.
    Planning rolls out with a history of 1 while training uses a longer one, so
    both are worth measuring. `copy` is the error of repeating the last true
    latent. `useful_horizon` counts the leading steps where the rollout beats it.
    """
    horizon = min(horizon, emb.shape[1] - history)
    if horizon < 1:
        return {}
    roll = emb[:, :history].clone()
    mse, copy = [], []
    with torch.no_grad(), eval_mode(jepa.predictor):
        for k in range(horizon):
            t = history + k
            pred = jepa.predict(roll[:, -history:], act_emb[:, t - history : t])[:, -1:]
            mse.append(F.mse_loss(pred, emb[:, t : t + 1]).item())
            copy.append(F.mse_loss(emb[:, history - 1 : history], emb[:, t : t + 1]).item())
            roll = torch.cat([roll, pred], dim=1)
    useful = 0
    for m, c in zip(mse, copy):
        if m >= c:
            break
        useful += 1
    return {"mse": mse, "copy": copy, "useful_horizon": useful, "history": history}


def cost_geometry(emb: torch.Tensor, gaps: list[int]) -> dict[str, Any]:
    """How the latent distance grows with the time gap, against unrelated samples.

    The planner minimises the latent distance to the goal, so this is the cost
    landscape: it should grow with the gap and keep growing far out. If it
    saturates after a few steps, far goals carry no signal.
    """
    e = emb.detach().float()
    if e.dim() != 3 or e.shape[0] < 2:
        return {}
    unrelated = (e - e.roll(1, dims=0)).pow(2).mean().item()
    used, ratio = [], []
    for g in gaps:
        if g < e.shape[1]:
            used.append(g)
            ratio.append((e[:, g:] - e[:, :-g]).pow(2).mean().item() / (unrelated + EPS))
    if len(used) < 2:
        return {}
    saturation = next((g for g, r in zip(used, ratio) if r >= 0.9), None)
    rises = sum(b > a for a, b in zip(ratio, ratio[1:]))
    return {
        "gaps": used,
        "ratio": ratio,
        "saturation_gap": saturation,
        "monotone_frac": rises / (len(ratio) - 1),
        "ratio_last": ratio[-1],
    }


# --------------------------------------------------------------------------
# vq and idm


def vq_metrics(vq: Any, indices: torch.Tensor | None = None) -> dict[str, float]:
    """Codebook use (of `indices`, default the last forward) and the number of dead codes."""
    out: dict[str, float] = {}
    if indices is None:
        indices = getattr(vq, "last_indices", None)
    if indices is not None:
        flat = indices.reshape(-1, vq.groups)
        perplexities, usages = [], []
        for g in range(vq.groups):
            counts = torch.bincount(flat[:, g], minlength=vq.num_codes).float()
            p = counts / counts.sum().clamp_min(1.0)
            perplexities.append(torch.exp(-(p * (p + EPS).log()).sum()).item())
            usages.append((counts > 0).float().mean().item())
        out["perplexity"] = sum(perplexities) / len(perplexities)
        out["perplexity_norm"] = out["perplexity"] / vq.num_codes
        out["usage"] = sum(usages) / len(usages)
    out["dead_frac"] = (vq.ema_count < vq.dead_threshold).float().mean().item()
    out["num_codes"] = vq.num_codes
    out["groups"] = vq.groups
    return out


def idm_lift(accuracy: float | None, chance: float) -> float | None:
    return None if accuracy is None else (accuracy - chance) / (1 - chance)


# --------------------------------------------------------------------------
# one level, from embeddings


def representation_summary(emb: torch.Tensor) -> dict[str, Any]:
    out = {**effective_rank(emb), **dim_stats(emb)}
    out["within_clip_share"] = within_clip_share(emb)
    out["step_to_scale"] = step_to_scale(emb)
    out["dim"] = emb.shape[-1]
    return out


# --------------------------------------------------------------------------
# diagnosis: training


def _f(findings, severity, code, level, message, hint):
    findings.append(Finding(severity, code, level, message, hint))


def diagnose_training(summary: dict, thr: Thresholds | None = None) -> list[Finding]:
    """Findings from a training summary.

    summary = {
      'levels': {n: {'rep': .., 'dyn': .., 'drift': {history: ..}, 'geometry': .., 'vq': .., 'idm': .., 'grad': ..}},
      'training': {'val_first', 'val_last', 'train_last', 'spikes', 'nonfinite'},
    }
    """
    thr = thr or Thresholds()
    findings: list[Finding] = []
    levels = summary.get("levels", {})

    for n, m in levels.items():
        n = int(n)
        rep = m.get("rep") or {}
        if rep:
            if rep.get("nonfinite_frac", 0) > 0:
                _f(findings, "critical", "REP_NONFINITE", n, "latents contain NaN/inf",
                   "check the learning rate, bf16 overflow and the data for NaNs")
            if rep.get("std_mean", 1) < thr.std_mean_critical:
                _f(findings, "critical", "REP_CONSTANT", n,
                   f"latents are constant (mean std {rep['std_mean']:.2e})",
                   "complete collapse: raise the SIGReg weight, lower the learning rate, check the projector")
            frac = rep.get("erank_frac")
            if frac is not None and not math.isnan(frac):
                if frac < thr.erank_frac_critical:
                    _f(findings, "critical", "REP_COLLAPSE", n,
                       f"effective rank {rep['erank']:.1f} of {rep['dim']} ({frac:.1%})",
                       "dimensional collapse: raise the SIGReg weight, check the IDM weight, lower the learning rate")
                elif frac < thr.erank_frac_warn:
                    _f(findings, "warn", "REP_LOW_RANK", n,
                       f"effective rank {rep['erank']:.1f} of {rep['dim']} ({frac:.1%})",
                       "latents use few dimensions: watch if it keeps falling")
            if rep.get("dead_dim_frac", 0) > thr.dead_dim_frac_warn:
                _f(findings, "warn", "REP_DEAD_DIMS", n,
                   f"{rep['dead_dim_frac']:.0%} of the dimensions are nearly constant",
                   "partial collapse: raise the SIGReg weight")
            ratio = rep.get("step_to_scale")
            if ratio is not None:
                if ratio < thr.static_step_ratio:
                    _f(findings, "warn", "REP_STATIC", n,
                       f"latents barely change per step (step/scale {ratio:.1e})",
                       "the level ignores temporal change: check stride and whether the encoder discards the dynamics")
                elif ratio > thr.incoherent_step_ratio:
                    _f(findings, "warn", "REP_INCOHERENT", n,
                       f"consecutive latents are almost as different as random ones (step/scale {ratio:.2f})",
                       "no temporal coherence: the encoder keys on high-frequency detail")

        dyn = m.get("dyn") or {}
        if dyn.get("pred_over_copy") is not None:
            r = dyn["pred_over_copy"]
            if r >= thr.pred_over_copy_critical:
                _f(findings, "critical", "DYN_WORSE_THAN_COPY", n,
                   f"predictor error is {r:.2f}x the copy-last-latent error",
                   "the predictor has not learned the dynamics: more steps, a higher predictor learning rate, check the action path")
            elif r >= thr.pred_over_copy_warn:
                _f(findings, "warn", "DYN_NOT_BETTER_THAN_COPY", n,
                   f"predictor error is {r:.2f}x the copy-last-latent error",
                   "it only copies: normal very early, a problem if it persists")
        lift = dyn.get("action_lift")
        if lift is not None:
            if lift < thr.action_lift_critical:
                _f(findings, "critical", "DYN_IGNORES_ACTIONS", n,
                   f"shuffling the actions changes the prediction error by only {lift:.2%}",
                   "planning cannot work: check the action embedding, the IDM weight and that actions align with frames")
            elif lift < thr.action_lift_warn:
                _f(findings, "warn", "DYN_WEAK_ACTION_USE", n,
                   f"action shuffling raises the prediction error by only {lift:.1%}",
                   "weak action conditioning: check the IDM weight and the share of no-op actions in the data")

        drift = m.get("drift") or {}
        for hist, curve in drift.items():
            if not curve or not curve.get("mse"):
                continue
            ratio = curve["mse"][-1] / (curve["copy"][-1] + EPS)
            if ratio >= thr.drift_ratio_warn and int(hist) == int(summary.get("plan_history", 1)):
                _f(findings, "warn", "DRIFT_WORSE_THAN_COPY", n,
                   f"open-loop rollout (history {hist}) is {ratio:.2f}x copy error at step {len(curve['mse'])}; "
                   f"useful for {curve['useful_horizon']} steps",
                   "errors compound: plan with shorter horizons at this level or train with a longer rollout")
        if len(drift) >= 2:
            hists = sorted(int(h) for h in drift)
            short, long = drift[hists[0]], drift[hists[-1]]
            if short.get("mse") and long.get("mse"):
                k = min(len(short["mse"]), len(long["mse"])) - 1
                r = short["mse"][k] / (long["mse"][k] + EPS)
                if r > thr.history_mismatch_ratio:
                    _f(findings, "warn", "PLAN_HISTORY_MISMATCH", n,
                       f"rollout with history {hists[0]} (as in planning) is {r:.1f}x worse than with history {hists[-1]} (as in training)",
                       "plan with the training history or train with history 1 in the rollout")

        geo = m.get("geometry") or {}
        if geo:
            sat = geo.get("saturation_gap")
            if sat is not None and sat <= thr.saturation_gap_warn:
                _f(findings, "warn", "GEOM_SATURATES", n,
                   f"latent distance reaches the unrelated-sample level after {sat} step(s)",
                   "the cost carries no signal beyond that horizon: this level cannot guide far goals")
            if geo.get("monotone_frac", 1) < thr.monotone_frac_warn:
                _f(findings, "warn", "GEOM_NONMONOTONE", n,
                   f"latent distance grows with the time gap in only {geo['monotone_frac']:.0%} of the steps",
                   "the cost landscape is noisy: planning toward distant goals will be unreliable")

        vq = m.get("vq") or {}
        if vq:
            norm = vq.get("perplexity_norm")
            if norm is not None:
                if norm < thr.vq_perplexity_norm_critical:
                    _f(findings, "critical", "VQ_COLLAPSE", n,
                       f"codebook perplexity {vq['perplexity']:.1f} of {vq['num_codes']} codes",
                       "codebook collapse: raise the dead-code threshold, lower the decay, fewer groups, or check the encoder output scale")
                elif norm < thr.vq_perplexity_norm_warn:
                    _f(findings, "warn", "VQ_LOW_USE", n,
                       f"codebook perplexity {vq['perplexity']:.1f} of {vq['num_codes']} codes",
                       "few codes in use: a smaller codebook may be enough, or the encoder is under-trained")
            if vq.get("dead_frac", 0) > thr.vq_dead_frac_warn:
                _f(findings, "warn", "VQ_DEAD_CODES", n, f"{vq['dead_frac']:.0%} of the codes are dead",
                   "codes are not being re-seeded fast enough")

        idm = m.get("idm") or {}
        if idm.get("lift") is not None:
            if idm["lift"] < thr.idm_lift_critical:
                _f(findings, "critical", "IDM_CHANCE", n,
                   f"inverse dynamics is at chance (accuracy {idm['accuracy']:.3f}, chance {idm['chance']:.3f})",
                   "latents do not encode what the actions did: check frame/action alignment and the encoder")
            elif idm["lift"] < thr.idm_lift_warn:
                _f(findings, "warn", "IDM_WEAK", n,
                   f"inverse dynamics barely above chance (accuracy {idm['accuracy']:.3f})",
                   "many actions have no visible effect, or the latents are too coarse for them")

        grad = m.get("grad") or {}
        if grad.get("explosions", 0) > 0:
            _f(findings, "warn", "GRAD_EXPLOSION", n,
               f"{grad['explosions']} step(s) with a gradient norm above {thr.grad_explosion_factor:g}x the median",
               "lower the learning rate or check for outlier batches")
        if grad.get("median") is not None and grad["median"] < thr.grad_vanishing:
            _f(findings, "warn", "GRAD_VANISHING", n, f"median gradient norm {grad['median']:.1e}",
               "this level hardly learns: check detach settings and the loss weights")

    # upper levels should be slower than the level below
    steps = {int(n): (m.get("rep") or {}).get("step_to_scale") for n, m in levels.items()}
    for n in sorted(steps):
        if n > 1 and steps[n] is not None and steps.get(n - 1) is not None and steps[n] <= steps[n - 1]:
            _f(findings, "info", "NO_ABSTRACTION", n,
               f"latent change per step ({steps[n]:.3f}) is not larger than at level {n - 1} ({steps[n - 1]:.3f})",
               "an upper level should span more time per step: check the stride and the encoder")

    tr = summary.get("training") or {}
    if tr.get("nonfinite"):
        _f(findings, "critical", "LOSS_NONFINITE", None, "the loss became NaN/inf", "see the last logged gradient norms")
    first, last = tr.get("val_first"), tr.get("val_last")
    if first and last is not None and tr.get("epochs", 0) >= 2 and (first - last) / abs(first) < thr.loss_improvement_warn:
        _f(findings, "warn", "LOSS_STALLED", None,
           f"validation loss improved by only {(first - last) / abs(first):.1%} (from {first:.4f} to {last:.4f})",
           "training barely progressed: learning rate, data, or a collapsed term dominating the loss")
    if last and tr.get("train_last") and last / tr["train_last"] > thr.overfit_ratio_warn:
        _f(findings, "warn", "OVERFIT", None,
           f"validation loss is {last / tr['train_last']:.2f}x the training loss",
           "more data or seeds, stronger regularisation, or a smaller model")
    if tr.get("spikes", 0) >= thr.spike_count_warn:
        _f(findings, "warn", "LOSS_SPIKES", None, f"{tr['spikes']} loss spikes", "unstable: lower the learning rate")
    return sort_findings(findings)


# --------------------------------------------------------------------------
# reports


def _fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.{digits}g}"
    return str(x)


def findings_block(findings: list[Finding]) -> str:
    if not findings:
        return "No findings: no collapse signal and no failure signal above threshold.\n"
    return "\n".join(f"- **{f.severity.upper()}** `{f.code}`"
                     + ("" if f.level is None else f" (level {f.level})")
                     + f": {f.message}. *{f.hint}.*" for f in findings) + "\n"


def render_training_report(summary: dict, findings: list[Finding]) -> str:
    """Markdown report of a training run."""
    out = ["# Training report", "", "## Findings", "", findings_block(findings), "## Per level", ""]
    out.append("| level | erank | erank/dim | dead dims | step/scale | pred/copy | action lift | useful horizon | VQ perplexity | IDM acc (chance) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for n, m in sorted(summary.get("levels", {}).items(), key=lambda kv: int(kv[0])):
        rep, dyn, vq, idm = (m.get(k) or {} for k in ("rep", "dyn", "vq", "idm"))
        drift = (m.get("drift") or {}).get(summary.get("plan_history", 1)) or {}
        out.append(
            f"| {n} | {_fmt(rep.get('erank'))} | {_fmt(rep.get('erank_frac'))} | {_fmt(rep.get('dead_dim_frac'))} "
            f"| {_fmt(rep.get('step_to_scale'))} | {_fmt(dyn.get('pred_over_copy'))} | {_fmt(dyn.get('action_lift'))} "
            f"| {_fmt(drift.get('useful_horizon'))} | {_fmt(vq.get('perplexity'))}/{_fmt(vq.get('num_codes'))} "
            f"| {_fmt(idm.get('accuracy'))} ({_fmt(idm.get('chance'))}) |"
        )
    for n, m in sorted(summary.get("levels", {}).items(), key=lambda kv: int(kv[0])):
        geo = m.get("geometry") or {}
        if geo:
            out += ["", f"### Level {n}: latent distance vs time gap (1 = unrelated samples)", "",
                    "| gap (level steps) | " + " | ".join(str(g) for g in geo["gaps"]) + " |",
                    "|---|" + "---|" * len(geo["gaps"]),
                    "| distance ratio | " + " | ".join(_fmt(r, 2) for r in geo["ratio"]) + " |"]
        for hist, curve in sorted((m.get("drift") or {}).items(), key=lambda kv: int(kv[0])):
            if curve and curve.get("mse"):
                ks = range(1, len(curve["mse"]) + 1)
                out += ["", f"### Level {n}: open-loop rollout error, context of {hist} frame(s)", "",
                        "| step | " + " | ".join(str(k) for k in ks) + " |", "|---|" + "---|" * len(ks),
                        "| rollout mse | " + " | ".join(_fmt(v, 2) for v in curve["mse"]) + " |",
                        "| copy mse | " + " | ".join(_fmt(v, 2) for v in curve["copy"]) + " |"]
    tr = summary.get("training") or {}
    if tr:
        out += ["", "## Optimisation", "", "```", *(f"{k}: {_fmt(v, 5)}" for k, v in tr.items()), "```"]
    return "\n".join(out) + "\n"


def to_jsonable(x: Any) -> Any:
    """Plain Python types, for json."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return to_jsonable(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        return to_jsonable(x.item())
    return x


def findings_to_json(findings: list[Finding]) -> list[dict]:
    return [asdict(f) for f in findings]
