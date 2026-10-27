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
    std_mean_low_warn: float = 0.3  # SIGReg pulls the latents toward unit variance: a level with SIGReg
    std_mean_high_warn: float = 3.0  # whose mean std leaves this band is not held at that scale
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
    vq_reseed_rate_warn: float = 0.05  # share of codes re-seeded per step (moving average)
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
    dist_decrease_warn: float = 0.3  # share of tasks whose goal distance falls
    cem_improvement_warn: float = 0.01  # relative cost gain of CEM over its first samples
    cem_entropy_warn: float = 0.9  # normalised entropy of the final distributions
    optimism_warn: float = 2.0  # real / predicted goal distance after a plan
    plan_gap_ratio_warn: float = 1.5  # error of chosen plans / error of random plans
    plan_rank_corr_warn: float = 0.3  # predicted vs real cost ranking of candidates
    support_offsupport_warn: float = 0.25  # share of predicted latents far from the data
    degenerate_action_share: float = 0.9  # one action in this share of executed steps
    subgoal_reach_warn: float = 0.2
    # probing
    probe_group_r2_warn: float = 0.1
    probe_event_shown_warn: float = 0.25  # share of the real inventory change a one-step prediction shows
    probe_invented_change_warn: float = 0.1  # copy minus predicted accuracy where the entity did not change


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


# --------------------------------------------------------------------------
# representation metrics (embeddings of shape (B, T, D))


def effective_rank(e: torch.Tensor) -> dict[str, float]:
    """Exponential of the entropy of the normalised squared singular values (the covariance spectrum) of the centred latents."""
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
    # codes are re-seeded as soon as they fall below the threshold, so a count below it is never
    # seen after an update; the re-seed rate is what shows codes that keep dying
    out["reseed_rate"] = float(getattr(vq, "reseed_rate", 0.0))
    out["num_codes"] = vq.num_codes
    out["groups"] = vq.groups
    return out


def majority_baseline(target: torch.Tensor, num_classes: int) -> float:
    """Accuracy of always predicting the most frequent class of each slot (-1 targets are ignored).

    The reference for an inverse-dynamics model: expert actions are far from uniform (movement and
    `do` dominate, and so do a few codes), so 1 / num_classes is a too low bar and a model that only
    knows the class frequencies would look like it had learned something.
    """
    shares = []
    for slot in range(target.shape[-1]):
        t = target[..., slot].reshape(-1)
        t = t[t >= 0]
        if t.numel():
            shares.append(torch.bincount(t, minlength=num_classes).max().item() / t.numel())
    return max(1.0 / num_classes, sum(shares) / len(shares)) if shares else 1.0 / num_classes


def idm_lift(accuracy: float | None, chance: float) -> float | None:
    if accuracy is None or chance >= 1.0:  # one class only: nothing to learn, no lift to measure
        return None
    return (accuracy - chance) / (1 - chance)


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
      'sigreg_levels': [n, ..],  # levels whose latents SIGReg regularises (the scale check applies to them)
      'training': {'val_first', 'val_last', 'train_last', 'spikes', 'nonfinite'},
    }
    """
    thr = thr or Thresholds()
    findings: list[Finding] = []
    levels = summary.get("levels", {})
    sigreg_levels = {int(n) for n in summary.get("sigreg_levels", [])}

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
            elif n in sigreg_levels and not thr.std_mean_low_warn <= rep.get("std_mean", 1) <= thr.std_mean_high_warn:
                _f(findings, "warn", "REP_SCALE", n,
                   f"mean latent std {rep['std_mean']:.2g}, SIGReg pulls it toward 1",
                   "SIGReg does not hold the scale (the effective rank cannot see this): raise its weight, "
                   "or check whether the prediction loss shrinks the latents")
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
            if vq.get("reseed_rate", 0) > thr.vq_reseed_rate_warn:
                _f(findings, "warn", "VQ_CODE_CHURN", n, f"{vq['reseed_rate']:.1%} of the codes are re-seeded per step",
                   "codes keep dying: too few chunks per batch for this many codes, a collapsing encoder, or a threshold that is too high")

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
    if isinstance(x, torch.Tensor) or (hasattr(x, "tolist") and getattr(x, "ndim", 0) > 0):  # tensors, numpy arrays
        return to_jsonable(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        return to_jsonable(x.item())
    return x


def findings_to_json(findings: list[Finding]) -> list[dict]:
    return [asdict(f) for f in findings]


# --------------------------------------------------------------------------
# diagnosis: planning


def diagnose_planning(summary: dict, thr: Thresholds | None = None) -> list[Finding]:
    """Findings from `planning_metrics.summarize_planning`.

    No progress is reported first; the other findings narrow down the cause:
    the optimiser (CEM does not optimise or converge), the cost landscape (goal
    distance does not fall), the model's reliability (plans wrong more often
    than random ones, cost cannot rank plans, optimism), support, and plan
    degeneracy.
    """
    thr = thr or Thresholds()
    f: list[Finding] = []
    if not summary:
        return f

    if summary.get("progress_any", 0) == 0:
        _f(f, "critical", "PLAN_NO_PROGRESS", None,
           "no task unlocked any of its required achievements",
           "see the findings below for the likely cause")
    elif summary.get("success_rate", 0) == 0:
        _f(f, "info", "PLAN_PARTIAL_ONLY", None,
           f"no task succeeded, mean progress {summary['progress_mean']:.1%} of the required achievements",
           "the planner moves toward the goal but does not finish: budget, horizon, or far goals")

    for level, d in (summary.get("dist") or {}).items():
        if d.get("decreased_frac") is not None and d["decreased_frac"] < thr.dist_decrease_warn:
            _f(f, "warn", "DIST_NOT_DECREASING", int(level),
               f"latent distance to the goal fell in only {d['decreased_frac']:.0%} of the tasks",
               "plans do not bring the agent closer in this latent space: cost landscape or plan reliability")

    for level, p in (summary.get("planner") or {}).items():
        if p.get("cem_improvement") is not None and p["cem_improvement"] < thr.cem_improvement_warn:
            _f(f, "warn", "CEM_NOT_OPTIMIZING", int(level),
               f"CEM improves the cost by only {p['cem_improvement']:.2%} over its first samples",
               "the cost is flat over the candidates: the model barely distinguishes plans at this level")
        if p.get("cem_entropy_final") is not None and p["cem_entropy_final"] > thr.cem_entropy_warn:
            _f(f, "info", "CEM_NOT_CONVERGED", int(level),
               f"the final distributions are still {p['cem_entropy_final']:.0%} of uniform entropy",
               "more iterations or fewer candidate steps; or the cost gives no direction")

    pl = summary.get("plans") or {}
    if pl:
        if pl.get("err_ratio_vs_random") is not None and pl["err_ratio_vs_random"] > thr.plan_gap_ratio_warn:
            _f(f, "critical", "PLAN_EXPLOITATION", None,
               f"chosen plans are predicted {pl['err_ratio_vs_random']:.1f}x worse than random plans",
               "the optimiser finds model errors: restrict the search (proposals, trust region), check support")
        opt, opt_random = pl.get("optimism_chosen"), pl.get("optimism_random")
        if opt is not None and opt > thr.optimism_warn and (opt_random is None or opt > 1.5 * opt_random):
            _f(f, "warn", "PLAN_OPTIMISM", None,
               f"after a chosen plan the real goal distance is {opt:.1f}x the predicted one"
               + ("" if opt_random is None else f" (random plans: {opt_random:.1f}x)"),
               "the model promises progress that reality does not deliver")
        if pl.get("rank_corr") is not None and pl["rank_corr"] < thr.plan_rank_corr_warn:
            _f(f, "warn", "COST_RANKING", None,
               f"predicted and real goal distance of candidate plans correlate at {pl['rank_corr']:.2f}",
               "the model cannot order nearby plans: use shorter horizons or a better cost space")
        off = pl.get("offsupport_plan")
        if off is not None and off > thr.support_offsupport_warn and off > 2 * (pl.get("offsupport_real") or 0):
            _f(f, "warn", "OFF_SUPPORT", None,
               f"{off:.0%} of the predicted latents of chosen plans are far from the data "
               f"(reality: {pl.get('offsupport_real', 0):.0%})",
               "plans leave the data support: constrain the planner")
        if pl.get("pred_gain") is not None and pl["pred_gain"] > 0.05 and (pl.get("real_gain") or 0) <= 0:
            _f(f, "warn", "PREDICTED_GAIN_NOT_REAL", None,
               f"plans are predicted to close {pl['pred_gain']:.0%} of the goal distance, reality {pl['real_gain']:.0%}",
               "the dynamics model is wrong in the direction the planner exploits")

    act = summary.get("actions") or {}
    if act.get("degenerate_frac", 0) > 0.5:
        _f(f, "warn", "PLAN_DEGENERATE", None,
           f"in {act['degenerate_frac']:.0%} of the tasks one action made up over 90% of the steps",
           "collapsed plans: check the action embedding and the cost scale")
    elif act.get("noop_share", 0) > 0.5:
        _f(f, "info", "NOOP_HEAVY", None, f"{act['noop_share']:.0%} of the executed steps are no-ops",
           "the planner finds no useful action")

    sg = summary.get("subgoals") or {}
    if sg and sg.get("reached_frac", 1) < thr.subgoal_reach_warn:
        _f(f, "warn", "SUBGOAL_NOT_TRACKED", None,
           f"only {sg['reached_frac']:.0%} of the upper-level subgoals were approached by half of what the elapsed time allows",
           "the level below cannot follow the level above: check the action cost and the horizons")

    groups = summary.get("by_group") or {}
    easy, hard = groups.get("required_1"), [g for k, g in groups.items() if k.startswith("required_") and k != "required_1" and k != "required_0"]
    if easy and hard and easy["success"] > 0.3 and max(g["success"] for g in hard) == 0:
        _f(f, "info", "FAR_GOALS_FAIL", None,
           f"success is {easy['success']:.0%} with one required achievement and 0 with more",
           "goal distance is the limit: the cost geometry or the hierarchy at depth")
    # the headline first, then the causes by severity
    return sorted(sort_findings(f), key=lambda x: x.code != "PLAN_NO_PROGRESS")


def render_planning_report(summary: dict, findings: list[Finding]) -> str:
    """Markdown report of a planning evaluation."""
    out = ["# Planning report", "", "## Findings", "", findings_block(findings), "## Outcome", ""]
    out.append(f"{summary.get('num_tasks')} tasks. Success {_fmt(summary.get('success_rate'))}, "
               f"mean progress {_fmt(summary.get('progress_mean'))}, "
               f"any progress {_fmt(summary.get('progress_any'))}, "
               f"steps to first progress {_fmt(summary.get('steps_to_first_progress'))}.")
    groups = summary.get("by_group") or {}
    if groups:
        out += ["", "| group | tasks | success | progress |", "|---|---|---|---|"]
        out += [f"| {g} | {v['n']} | {_fmt(v['success'])} | {_fmt(v['progress'])} |" for g, v in groups.items()]
    if summary.get("diamond"):
        d = summary["diamond"]
        out += ["", f"Diamond: unlocked in {_fmt(d['rate'])} of the {d['n']} tasks that require it."]
    if summary.get("by_achievement"):
        out += ["", "| required achievement | tasks | unlocked |", "|---|---|---|"]
        out += [f"| {a} | {v['n']} | {_fmt(v['unlocked'])} |" for a, v in summary["by_achievement"].items()]
    if summary.get("dist"):
        out += ["", "## Latent distance to the goal (per level)", "",
                "| level | final / initial | best / initial | tasks with a fall |", "|---|---|---|---|"]
        out += [f"| {l} | {_fmt(d['final_over_initial'])} | {_fmt(d['min_over_initial'])} | {_fmt(d['decreased_frac'])} |"
                for l, d in summary["dist"].items()]
    if summary.get("planner"):
        out += ["", "## Planner", "", "| level | solves | CEM improvement | final entropy | horizon | wall clock (s) |",
                "|---|---|---|---|---|---|"]
        out += [f"| {l} | {p['solves']} | {_fmt(p['cem_improvement'])} | {_fmt(p['cem_entropy_final'])} "
                f"| {_fmt(p['horizon'])} | {_fmt(p['wall_clock'])} |" for l, p in summary["planner"].items()]
    c = summary.get("compute") or {}
    if c:
        out += ["", f"Compute: {_fmt(c.get('tflops_per_task'))} TFLOPs per task, "
                    f"{_fmt(c.get('tflops_per_solve'))} per solve, {_fmt(c.get('solves_per_task'))} solves per task.", ""]
    pl = summary.get("plans") or {}
    if pl:
        out += ["## Plan reliability (executed open-loop in the simulator)", "", "| quantity | chosen plans | random plans |",
                "|---|---|---|",
                f"| latent prediction error | {_fmt(pl['pred_err'])} | {_fmt(pl['pred_err_random'])} |",
                f"| real / predicted goal distance | {_fmt(pl['optimism_chosen'])} | {_fmt(pl['optimism_random'])} |"]
        if "offsupport_plan" in pl:
            out.append(f"| predicted latents off the data support | {_fmt(pl['offsupport_plan'])} | {_fmt(pl['offsupport_random'])} |")
        out += ["", f"Rank correlation of predicted and real goal distance over candidate plans: {_fmt(pl['rank_corr'])}. "
                    f"Predicted / real goal-distance reduction by the chosen plans: {_fmt(pl['pred_gain'])} / {_fmt(pl['real_gain'])}.", ""]
        if pl.get("err_curve"):
            ks = range(1, len(pl["err_curve"]) + 1)
            out += ["| step | " + " | ".join(map(str, ks)) + " |", "|---|" + "---|" * len(ks),
                    "| error of chosen plan | " + " | ".join(_fmt(v, 2) for v in pl["err_curve"]) + " |", ""]
    act = summary.get("actions") or {}
    if act:
        out += ["## Executed actions", "", f"largest single-action share {_fmt(act['max_share_mean'])}, "
                f"degenerate tasks {_fmt(act['degenerate_frac'])}, no-op share {_fmt(act['noop_share'])}, "
                f"distinct actions per task {_fmt(act['distinct_mean'])}", ""]
    sg = summary.get("subgoals") or {}
    if sg:
        out += ["## Subgoal tracking", "", f"{sg['n']} subgoals: mean progress {_fmt(sg['progress_mean'])}, "
                f"approached by half of what the elapsed time allows {_fmt(sg['reached_frac'])} "
                f"(the subgoal is checked after {_fmt(sg.get('expected_mean'))} of the time it lies ahead)", ""]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# diagnosis: probing


def _ratio(a, b):
    """Share of the base score that is kept; a score below zero counts as nothing."""
    return None if a is None or b is None or b <= 0.1 else max(a, 0.0) / b


def diagnose_probing(summary: dict, thr: Thresholds | None = None) -> list[Finding]:
    """Findings from `probe_suite.run_suite`: what each level keeps of the symbolic state, and the suite's level-1 results."""
    thr = thr or Thresholds()
    f: list[Finding] = []
    levels = {int(n): e for n, e in summary.get("levels", {}).items()}
    for n, samples in sorted((int(k), v) for k, v in (summary.get("n_train") or {}).items()):
        dim = (summary.get("dim") or {}).get(n) or (summary.get("dim") or {}).get(str(n))
        if dim and samples < 10 * dim:
            _f(f, "warn", "PROBE_FEW_SAMPLES", n, f"{samples} training samples for {dim} latent dimensions",
               "the probe scores are unreliable (overfit, often below zero): use more or longer clips")
    for n, e in sorted(levels.items()):
        groups = e["groups"]
        control = e.get("control_groups") or {}
        scores = [v for v in groups.values() if v is not None]
        if scores and max(scores) < 0.05:
            _f(f, "critical", "PROBE_NOTHING", n, "no group of the symbolic state can be read from the latents",
               "the level encodes none of the state: collapse, or the wrong frame/target alignment")
        gains = [groups[g] - control[g] for g in groups if groups[g] is not None and control.get(g) is not None]
        if gains and max(gains) < 0.05:
            _f(f, "warn", "PROBE_NO_GAIN", n, "the latents are no more informative than those of an untrained encoder",
               "training added no readable state information at this level")
        if scores and max(scores) >= 0.3:
            for g, v in groups.items():
                if v is not None and v < thr.probe_group_r2_warn:
                    _f(f, "warn", "PROBE_GROUP_MISSING", n, f"the {g} state is not readable (score {v:.2f}) while others are",
                       {"fast": "the level ignores position, facing and nearby mobs",
                        "medium": "vitals are not encoded: they change slowly and look alike in pixels",
                        "slow": "inventory and progress are not encoded: they sit in a small strip of the frame"}[g])

    base = levels.get(1)
    if base:
        for n, e in sorted(levels.items()):
            if n == 1:
                continue
            ratios = {g: _ratio(e["groups"].get(g), base["groups"].get(g)) for g in ("fast", "medium", "slow")}
            shown = ", ".join(f"{g} {r:.0%}" for g, r in ratios.items() if r is not None)
            if shown:
                _f(f, "info", "PROBE_RETENTION", n, f"share of the level-1 probe score kept: {shown}",
                   "the frequency gap: an upper level should keep slow state and may drop fast state")
            slow = ratios["slow"]
            if slow is not None and slow < 0.3:
                _f(f, "warn", "PROBE_UPPER_LOSES_SLOW", n, f"keeps only {slow:.0%} of the slow state of level 1",
                   "slow state (inventory, progress) is what a high level should keep: check the encoder and the stride")
            fast = ratios["fast"]
            if fast is not None and slow is not None and fast >= 0.9 and slow >= 0.9:
                _f(f, "info", "PROBE_NO_SPECIALISATION", n, "keeps fast and slow state alike",
                   "no frequency specialisation between the levels")
    _diagnose_suite(f, summary, thr)
    return sort_findings(f)


def rollout_reach(by_k: dict, target: str, floor: float = 0.5) -> int:
    """The largest k up to which the predictions recover at least `floor` of the encoded latent's advantage over copying
    on the windows where the target changed (the change test), at every k scored."""
    reached = 0
    for k in sorted(by_k, key=int):
        recovered = (by_k[k].get(target) or {}).get("recovered")
        if recovered is None or recovered < floor:
            break
        reached = int(k)
    return reached


def _diagnose_suite(f: list[Finding], summary: dict, thr: Thresholds):
    """Level-1 findings of the probe suite: the rollout's change test, the events, the ridge grid, the verdicts."""
    change = (summary.get("change_plan") or {}).get("by_k") or {}
    # the state targets only (the view concepts have their own section in the report)
    groups = {t: e.get("group") for t, e in (((summary.get("state") or {}).get("ridge") or {}).get("latent") or {})
              .get("targets", {}).items()}  # fmt: skip
    state = lambda t: groups.get(t, "fast") in ("fast", "medium", "slow")  # noqa: E731
    if change:
        reach = {t: rollout_reach(change, t) for t in change[min(change, key=int)] if state(t)}
        _f(f, "info", "PROBE_ROLLOUT_HORIZON", 1, "where the entity changed, predictions recover half of the encoded latent's "
           "advantage over copying up to k = " + ", ".join(f"{v} ({t})" for t, v in reach.items()),
           "the probe-readable rollout horizon per entity (planning history): an input to the planning horizon")
        if reach and max(reach.values()) == 0:
            _f(f, "warn", "PROBE_ROLLOUT_NO_GAIN", 1, "one step ahead, the predictions recover less than half of every change",
               "the predictor does not move the state: check the action conditioning")
        first = change[min(change, key=int)]
        invented = [t for t, e in first.items() if state(t) and e.get("pred_same") is not None
                    and e.get("copy_same") is not None and e["copy_same"] - e["pred_same"] > thr.probe_invented_change_warn]  # fmt: skip
        if invented:
            _f(f, "warn", "PROBE_INVENTS_CHANGES", 1, f"one step ahead, the readout of {', '.join(invented)} is wrong more "
               "often than copying where nothing changed", "the predictor changes what did not change")
    for group in ("make", "place", "collect"):
        e = (summary.get("events") or {}).get(f"{group} (all)") or {}
        if e.get("shown") is not None and e["shown"] < thr.probe_event_shown_warn:
            _f(f, "warn", "PROBE_EVENT_IGNORED", 1,
               f"a one-step prediction after a successful {group} shows {e['shown']:.0%} of the change "
               f"({e['succeeded']} successful, {e['failed']} failed presses)",
               "rare transitions are predicted as no change: oversampling them is the open decision")
    edge = (summary.get("state") or {}).get("lambda_at_edge") or []
    if edge:
        _f(f, "info", "PROBE_LAMBDA_EDGE", 1, f"the ridge strength of {', '.join(edge)} is at the edge of the grid",
           "widen probing.readout.LAMBDAS")
    lost: dict[str, list[str]] = {}
    for kind in ("blocks", "mobs", "items", "other"):
        for row in (summary.get("verdicts") or {}).get(kind) or []:
            if row["verdict"] != "kept":
                lost.setdefault(row["verdict"], []).append(row["entity"])
    for verdict, entities in sorted(lost.items()):
        _f(f, "info", "PROBE_VERDICT", 1, f"{verdict}: {', '.join(entities)}",
           "the verdict tables of the report have the score of every stage")
