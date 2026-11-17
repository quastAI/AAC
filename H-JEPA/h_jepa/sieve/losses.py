"""The Sieve's pixel terms (aac_recipe.md, "Losses"); the next-latent MSE against a stop-gradient target is the usual
prediction term of `hjepa_forward` (`level1.wm.detach_pred_target: true`).

    stage 1   recon_weight * sum_i w_i e_i(D(z_t), x_t)                  every frame of the clip, w_i error-weighted
    stage 2   sum_i e_i(D(p), x_next)                                     decoded prediction
              + sum_i exp(-sg(g_i) / tau) e_i(D(z_next), x_next)          prediction-gated reconstruction
    joint     sum_i e_i(D(p), x_next)                                     decoded prediction
              + sum_i G_i w_i e_i(D(z_next), x_next)                      relative, floored gate (one stage, V5)
              G_i = floor + (1 - floor) exp(-sg(g_i) / (beta mean_j g_j))

The gate's error g_i (`gate`): `abs` (the recipe's) e_i(D(p), x_next), the decoded prediction against the pixels, so a
patch the decoder cannot draw within tau is shut even where the prediction is right; `diff` e_i(D(p), D(z_next)), the
decoded prediction against the decoded latent of the next frame: only what the prediction changes in the picture, the
decoder's precision left out (`gate_analysis.py`, 2026-11-17: on stage-1 epoch 8 it opens the ores and stations from
about 0.1 to 0.6, keeps entering and changed tiles at about half the unchanged ones, and does not shut night noise the
latent does not hold).

`joint` (the one-stage Sieve, README id V5, config `craftax_sieve_joint`) runs both stages as one training from the
start. Its gate is relative: each patch's error is measured against the batch's mean error, so the gate needs no tau
and calibrates itself. Early in training the errors are all large and alike, the gate is about uniform and the run
reconstructs every next frame, as stage 1 does; as the predictor gets some content right, that content's error falls
below the mean and opens while the rest closes towards the floor, as stage 2 does. The floor keeps a minimum
reconstruction pressure on every patch, so content the predictor cannot foresee yet is not shut for good (with the
recipe's gate a shut patch stopped the decoder from improving there, `gate_analysis.py`). `w_i` are stage 1's error
weights (`error_power` 0 in `craftax_sieve_joint`: all ones, the gate alone weighs the patches).

`e_i` is the mean squared error of patch i (pixels in [0, 1]); sums run over the 81 patches, means over frames and
clips. Every value whose key holds `loss` is logged by `hjepa_forward` (train and validation): the diagnostic ones are
computed without gradient.
"""

from __future__ import annotations

import torch

from .decoder import patch_errors, patchify, to_unit

STAGES = (1, 2, "joint")


def stage_of(cfg) -> int | str:
    """The Sieve mode of `level1.sieve`: 1, 2 or "joint" (the one-stage Sieve)."""
    raw = cfg.get("stage", 1) if cfg is not None else 1
    if str(raw) == "joint":
        return "joint"
    try:
        stage = int(raw)
    except (TypeError, ValueError):
        stage = None
    if stage not in (1, 2):
        raise ValueError(f"level1.sieve.stage must be 1, 2 or joint, got {raw}")
    return stage


def error_weights(errors: torch.Tensor, power: float = 1.0, cap: float = 20.0) -> torch.Tensor:
    """Label-free weights of the stage-1 reconstruction: (e_i / mean e)^power over the batch's patches, normalised to mean
    1, then capped. A badly reconstructed patch counts more, so small content a plain MSE neglects is pulled in. Call
    it on detached errors (the weights are a stop-gradient)."""
    w = (errors / errors.mean().clamp(min=1e-12)).pow(power)
    w = w / w.mean().clamp(min=1e-12)
    return w.clamp(max=cap)


def gate_error(kind: str, decoded_pred: torch.Tensor, decoded_real: torch.Tensor, e_pred: torch.Tensor) -> torch.Tensor:
    """The detached error a gate reads per patch: `abs` the decoded prediction against the pixels, `diff` against the
    decoded latent of the next frame."""
    if kind == "abs":
        return e_pred.detach()
    if kind == "diff":
        return patch_errors(decoded_pred.detach(), decoded_real.detach())
    raise ValueError(f"level1.sieve.gate must be abs or diff, got {kind}")


def relative_gate(errors: torch.Tensor, floor: float = 0.1, beta: float = 1.0) -> torch.Tensor:
    """floor + (1 - floor) exp(-e_i / (beta mean e)): the one-stage gate. Scale-free (errors times any factor give the same
    gate), about uniform when the errors are alike, never under `floor`. Call it on detached errors."""
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"level1.sieve.gate_floor must be in [0, 1), got {floor}")
    if beta <= 0:
        raise ValueError(f"level1.sieve.gate_beta must be positive, got {beta}")
    scale = errors.mean().clamp(min=1e-12)
    return floor + (1.0 - floor) * torch.exp(-errors / (beta * scale))


def sieve_terms(decoder, emb: torch.Tensor, pred: torch.Tensor, pixels: torch.Tensor, cfg, history: int) -> dict:
    """The pixel terms of one batch.

    emb (B, T, D): the encoded latents of the clip; pred (B, D): the predicted latent of frame `history` (from the
    frames before it); pixels (B, T, 3, 63, 63): the frames, ImageNet-normalised (as the encoder reads them).
    Returns the logged values (keys ending in `_loss`) and `sieve_loss`, the weighted sum that joins the level's loss.
    """
    stage = stage_of(cfg)
    target = patchify(to_unit(pixels))  # (B, T, 81, 147)
    x_next = target[:, history]
    out = {}
    if stage == 1:
        errors = patch_errors(decoder(emb), target)  # (B, T, 81)
        weights = error_weights(errors.detach(), float(cfg.get("error_power", 1.0)), float(cfg.get("error_cap", 20.0)))
        recon = (weights * errors).sum(-1).mean()
        out["recon_loss"] = recon
        with torch.no_grad():
            out["recon_plain_loss"] = errors.sum(-1).mean()  # without the weights: comparable between runs
            out["recon_worst_weight_loss"] = weights.max()
            e_pred = patch_errors(decoder(pred), x_next)
        out["sieve_loss"] = float(cfg.get("recon_weight", 0.3)) * recon
    else:
        decoded_pred, decoded_real = decoder(pred), decoder(emb[:, history])
        e_pred = patch_errors(decoded_pred, x_next)  # (B, 81)
        e_real = patch_errors(decoded_real, x_next)
        g_err = gate_error(str(cfg.get("gate", "abs")), decoded_pred, decoded_real, e_pred)
        if stage == 2:
            tau = decoder.tau
            if not torch.isfinite(tau):
                raise ValueError("the decoder's tau is not set: the training sets it from the pixel variance (PatchDecoder.set_pixel_variance)")
            gate = torch.exp(-g_err / tau)
            weights = torch.ones_like(gate)
        else:
            floor = float(cfg.get("gate_floor", 0.1))
            gate = relative_gate(g_err, floor, float(cfg.get("gate_beta", 1.0)))
            weights = error_weights(e_real.detach(), float(cfg.get("error_power", 0.0)), float(cfg.get("error_cap", 20.0)))
            with torch.no_grad():
                out["gate_shut_share_loss"] = (gate < floor + 0.05 * (1.0 - floor)).float().mean()  # patches near the floor
                out["gate_std_loss"] = gate.std()  # how selective the gate is (0: uniform)
        decoded = e_pred.sum(-1).mean()
        gated = (gate * weights * e_real).sum(-1).mean()
        out["decoded_pred_loss"] = decoded
        out["gated_recon_loss"] = gated
        with torch.no_grad():
            out["gate_open_loss"] = gate.mean()  # the share of the reconstruction the gate lets through
            out["recon_plain_loss"] = e_real.sum(-1).mean()
        out["sieve_loss"] = float(cfg.get("decoded_pred_weight", 1.0)) * decoded + float(cfg.get("gated_recon_weight", 1.0)) * gated
    with torch.no_grad():
        out.setdefault("decoded_pred_loss", e_pred.sum(-1).mean())  # stage 1: what the predictor would draw already
        tau = decoder.tau
        if torch.isfinite(tau):
            out["sharp_share_loss"] = (e_pred < tau).float().mean()  # patches whose decoded prediction beats tau
    return out
