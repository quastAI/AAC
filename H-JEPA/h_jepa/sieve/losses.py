"""The Sieve's pixel terms (aac_recipe.md, "Losses"); the next-latent MSE against a stop-gradient target is the usual
prediction term of `hjepa_forward` (`level1.wm.detach_pred_target: true`).

    stage 1   recon_weight * sum_i w_i e_i(D(z_t), x_t)                  every frame of the clip, w_i error-weighted
    stage 2   sum_i e_i(D(p), x_next)                                     decoded prediction
              + sum_i exp(-sg(e_i(D(p), x_next)) / tau) e_i(D(z_next), x_next)   prediction-gated reconstruction

`e_i` is the mean squared error of patch i (pixels in [0, 1]); sums run over the 81 patches, means over frames and
clips. Every value whose key holds `loss` is logged by `hjepa_forward` (train and validation): the diagnostic ones are
computed without gradient.
"""

from __future__ import annotations

import torch

from .decoder import patch_errors, patchify, to_unit


def error_weights(errors: torch.Tensor, power: float = 1.0, cap: float = 20.0) -> torch.Tensor:
    """Label-free weights of the stage-1 reconstruction: (e_i / mean e)^power over the batch's patches, normalised to mean
    1, then capped. A badly reconstructed patch counts more, so small content a plain MSE neglects is pulled in. Call
    it on detached errors (the weights are a stop-gradient)."""
    w = (errors / errors.mean().clamp(min=1e-12)).pow(power)
    w = w / w.mean().clamp(min=1e-12)
    return w.clamp(max=cap)


def sieve_terms(decoder, emb: torch.Tensor, pred: torch.Tensor, pixels: torch.Tensor, cfg, history: int) -> dict:
    """The pixel terms of one batch.

    emb (B, T, D): the encoded latents of the clip; pred (B, D): the predicted latent of frame `history` (from the
    frames before it); pixels (B, T, 3, 63, 63): the frames, ImageNet-normalised (as the encoder reads them).
    Returns the logged values (keys ending in `_loss`) and `sieve_loss`, the weighted sum that joins the level's loss.
    """
    stage = int(cfg.get("stage", 1))
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
    elif stage == 2:
        e_pred = patch_errors(decoder(pred), x_next)  # (B, 81)
        e_real = patch_errors(decoder(emb[:, history]), x_next)
        tau = decoder.tau
        if not torch.isfinite(tau):
            raise ValueError("the decoder's tau is not set: the training sets it from the pixel variance (PatchDecoder.set_pixel_variance)")
        gate = torch.exp(-e_pred.detach() / tau)
        decoded = e_pred.sum(-1).mean()
        gated = (gate * e_real).sum(-1).mean()
        out["decoded_pred_loss"] = decoded
        out["gated_recon_loss"] = gated
        with torch.no_grad():
            out["gate_open_loss"] = gate.mean()  # the share of the reconstruction the gate lets through
            out["recon_plain_loss"] = e_real.sum(-1).mean()
        out["sieve_loss"] = float(cfg.get("decoded_pred_weight", 1.0)) * decoded + float(cfg.get("gated_recon_weight", 1.0)) * gated
    else:
        raise ValueError(f"level1.sieve.stage must be 1 or 2, got {stage}")
    with torch.no_grad():
        out.setdefault("decoded_pred_loss", e_pred.sum(-1).mean())  # stage 1: what the predictor would draw already
        tau = decoder.tau
        if torch.isfinite(tau):
            out["sharp_share_loss"] = (e_pred < tau).float().mean()  # patches whose decoded prediction beats tau
    return out
