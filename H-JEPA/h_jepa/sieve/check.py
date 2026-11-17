"""Every Sieve metric of one checkpoint (aac_recipe.md, "Metrics"), on the fixed sample and the counterfactual bank.

What the latent holds (labels only to read; the probes are fitted again for every checkpoint, fixed steps and seed):
    holds_<entity>            the MLP probe's score on the held-out worlds (`probe.py` has the scores and chance)
    holds_<entity>_linear     the linear probe's; `linear_vs_mlp`: the linear probe's skill over the MLP's (how much a plain
                              distance can see: CEM compares latents by MSE), the mean over the entities the MLP reads
    foresees_<entity>         the MLP probe on the predicted latent, right where the entity changed; `copy_<entity>`: the probe
                              on the last latent on the same transitions (copying); `changed_<entity>`: how many
The latent and the predictor (no labels):
    effective_rank            of the held-out latents (collapse guard: SIGReg is off); `dead_axes` under a tenth of the mean variance
    latent_error_vs_copy      the predictor's next-latent error over copying the last latent
    latent_error_gap          held-out over training next-latent error (overfitting)
    recon_error_mean          the own decoder's error per patch on held-out frames, the mean ...
    recon_error_worst1pct     ... and its 99th percentile (is small content reconstructed?); `recon_gap`: held-out over training
    decoded_pred_error        the decoded predicted latent against the real next frame, the mean per patch (flat early: no steering)
    sharp_patch_share         the share of patches whose decoded prediction is under tau (the gate's own quantity)
    recon_vs_pred_gradient    the encoder gradient of the pixel terms over the prediction term's, as weighted in training
Actions (the bypass: content drawn from the action alone is not held in the latent):
    action_lift               next-latent error with shuffled actions over the error with the true ones
    action_follow             how far the predicted latent moves when the action is replaced, over the real one-step change
    action_follow_frame       the same for the decoded frame
    action_follow_facing      fed a move, the share of predictions whose probed facing is that move's direction
What moves the latent (counterfactual frames, squared distance over that to an unrelated frame, the planner's unit):
    moves_for_facing, moves_for_mob, moves_for_tile, moves_for_item, moves_for_noise
Content held at all (stage 2 and the one-stage `joint`; in stage 1 the own decoder trains on real latents and says the same):
    fresh_decoder_error       a new decoder fitted to the frozen training latents, its held-out error per patch (mean) ...
    fresh_decoder_worst1pct   ... and its 99th percentile
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from . import probe as pr
from .decoder import PatchDecoder, patch_errors, patchify, uint8_to_unit
from .losses import sieve_terms

NUM_ACTIONS = 17
MOVES = (1, 2, 3, 4)  # left, right, up, down: facing 1..4 in the same order
FRESH_STEPS, FRESH_BATCH = 3000, 256
DEAD = 0.1


def _normalized(frames: np.ndarray, device) -> torch.Tensor:
    from main_hjepa import _normalize_uint8_images_on_device

    batch = {"pixels": torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).to(device)}
    _normalize_uint8_images_on_device(batch)
    return batch["pixels"]


@torch.no_grad()
def encode(jepa, frames: np.ndarray, device, chunk: int = 512) -> np.ndarray:
    return torch.cat([jepa.encoder(_normalized(frames[a : a + chunk], device)).float().cpu() for a in range(0, len(frames), chunk)]).numpy()


@torch.no_grad()
def predict(jepa, z: np.ndarray, action: np.ndarray, device, chunk: int = 2048) -> np.ndarray:
    """The predicted next latent from one latent and the action taken (history 1)."""
    from slowness.predict import action_embedding

    out = []
    for a in range(0, len(z), chunk):
        act = action_embedding(jepa, action[a : a + chunk, None, None].astype(np.float32), NUM_ACTIONS).to(device)
        ctx = torch.from_numpy(z[a : a + chunk, None]).to(device)
        out.append(jepa.predict(ctx, act)[:, -1].float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def decode_errors(decoder, z: np.ndarray, frames: np.ndarray, device, chunk: int = 1024) -> np.ndarray:
    """Patch errors (N, 81) of the decoded latents against the frames."""
    out = []
    for a in range(0, len(z), chunk):
        target = patchify(uint8_to_unit(frames[a : a + chunk]).to(device))
        out.append(patch_errors(decoder(torch.from_numpy(z[a : a + chunk]).to(device)), target).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def decoded_frames(decoder, z: np.ndarray, device) -> np.ndarray:
    return decoder(torch.from_numpy(z).to(device)).float().cpu().numpy()


def effective_rank(z: np.ndarray) -> tuple[float, int]:
    from slowness.variance import effective_rank as er

    values = np.clip(np.linalg.eigvalsh(np.cov(z.astype(np.float64), rowvar=False)), 0, None)
    return er(values), int((values < DEAD * values.mean()).sum())


def gradient_ratio(jepa, part, t, t1, sieve_cfg, device, n: int = 256, seed: int = 0) -> float:
    """|encoder gradient of the weighted pixel terms| / |encoder gradient of the next-latent MSE| on n training transitions."""
    from slowness.predict import action_embedding

    pick = np.random.default_rng(seed).choice(len(t), min(n, len(t)), replace=False)
    frames = np.stack([part.pixels[t[pick]], part.pixels[t1[pick]]], 1)  # (n, 2, 63, 63, 3)
    pixels = _normalized(frames.reshape(-1, *frames.shape[2:]), device)
    was = jepa.training
    jepa.eval()
    try:
        emb = jepa.encoder(pixels).reshape(len(pick), 2, -1)
        act = action_embedding(jepa, part.action[t[pick], None, None], NUM_ACTIONS).to(device)
        pred = jepa.predict(emb[:, :1], act)[:, -1]
        latent = F.mse_loss(pred, emb[:, 1].detach())
        pixel = sieve_terms(jepa.decoder, emb, pred, pixels.reshape(len(pick), 2, *pixels.shape[1:]), sieve_cfg, 1)["sieve_loss"]
        params = [p for p in jepa.encoder.parameters() if p.requires_grad]
        norm = lambda gs: torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None))  # noqa: E731
        g_latent = norm(torch.autograd.grad(latent, params, retain_graph=True, allow_unused=True))
        g_pixel = norm(torch.autograd.grad(pixel, params, allow_unused=True))
    finally:
        jepa.train(was)
    return float(g_pixel / g_latent.clamp(min=1e-12))


def fresh_decoder(z_tr, frames_tr, z_va, frames_va, device, steps: int = FRESH_STEPS, seed: int = 0) -> np.ndarray:
    """A new decoder on frozen latents (plain patch errors for every model), its held-out patch errors (N, 81)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    decoder = PatchDecoder(z_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    z = torch.from_numpy(z_tr)
    for _ in range(steps):
        idx = torch.randint(0, len(z), (min(FRESH_BATCH, len(z)),), generator=g)
        target = patchify(uint8_to_unit(frames_tr[idx.numpy()]).to(device))
        loss = patch_errors(decoder(z[idx].to(device)), target).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    decoder.eval()
    return decode_errors(decoder, z_va, frames_va, device)


def _moves(jepa, bank: dict, device) -> dict:
    """Squared latent distance of each edit over the mean squared distance between unrelated frames of the bank."""
    if bank is None or not len(bank.get("orig", [])):
        return {}
    z0 = encode(jepa, bank["orig"], device)
    perm = np.roll(np.random.default_rng(0).permutation(len(z0)), 1)
    unrelated = float(((z0 - z0[perm]) ** 2).sum(-1).mean())
    out = {}
    for kind in ("facing", "mob", "tile", "item"):
        valid = bank[f"valid_{kind}"]
        if valid.any():
            z1 = encode(jepa, bank[kind][valid], device)
            out[f"moves_for_{kind}"] = float(((z1 - z0[valid]) ** 2).sum(-1).mean() / max(unrelated, 1e-12))
    if len(bank["noise_orig"]):
        a, b = encode(jepa, bank["noise_orig"], device), encode(jepa, bank["noise_alt"], device)
        out["moves_for_noise"] = float(((a - b) ** 2).sum(-1).mean() / max(unrelated, 1e-12))
    return out


def evaluate(model, sample, bank=None, sieve_cfg=None, device="cpu", fresh: bool = False, probe_steps: int = pr.STEPS,
             fresh_steps: int = FRESH_STEPS, log=lambda msg: None) -> dict:  # fmt: skip
    """The metrics of one model (a saved `HJEPA`) on the sample; `fresh` adds the fresh decoder (stage 2, joint)."""
    t0 = time.time()
    model = model.to(device)
    model.eval()
    jepa = model.get_level(1)
    tr, va = sample.train, sample.val
    m: dict = {}

    log("encode")
    z_tr, z_va = encode(jepa, tr.pixels, device), encode(jepa, va.pixels, device)
    m["effective_rank"], m["dead_axes"] = effective_rank(z_va)

    log("predict")
    t_tr, t1_tr = tr.pairs()
    t_va, t1_va = va.pairs()
    a_va = va.action[t_va]
    p_va = predict(jepa, z_va[t_va], a_va, device)
    p_tr = predict(jepa, z_tr[t_tr], tr.action[t_tr], device)
    err_va = float(((p_va - z_va[t1_va]) ** 2).mean())
    m["latent_error_vs_copy"] = err_va / max(float(((z_va[t_va] - z_va[t1_va]) ** 2).mean()), 1e-12)
    m["latent_error_gap"] = err_va / max(float(((p_tr - z_tr[t1_tr]) ** 2).mean()), 1e-12)
    rng = np.random.default_rng(0)
    shuffled = predict(jepa, z_va[t_va], rng.permutation(a_va), device)
    m["action_lift"] = float(((shuffled - z_va[t1_va]) ** 2).mean()) / max(err_va, 1e-12)
    other = (a_va + rng.integers(1, NUM_ACTIONS, len(a_va))) % NUM_ACTIONS  # always another action
    p_other = predict(jepa, z_va[t_va], other.astype(np.float32), device)
    step = max(float(((z_va[t1_va] - z_va[t_va]) ** 2).mean()), 1e-12)
    m["action_follow"] = float(((p_other - p_va) ** 2).mean()) / step
    move = rng.choice(MOVES, len(a_va)).astype(np.float32)
    p_move = predict(jepa, z_va[t_va], move, device)

    decoder = getattr(jepa, "decoder", None)
    if decoder is not None:
        log("decode")
        e_va = decode_errors(decoder, z_va, va.pixels, device)
        e_tr = decode_errors(decoder, z_tr, tr.pixels, device)
        m["recon_error_mean"] = float(e_va.mean())
        m["recon_error_worst1pct"] = float(np.percentile(e_va, 99))
        m["recon_gap"] = float(e_va.mean() / max(e_tr.mean(), 1e-12))
        e_pred = decode_errors(decoder, p_va, va.pixels[t1_va], device)
        m["decoded_pred_error"] = float(e_pred.mean())
        tau = float(decoder.tau)
        m["tau"] = tau
        m["sharp_patch_share"] = float((e_pred < tau).mean()) if np.isfinite(tau) else float("nan")
        n = min(2048, len(p_va))
        frame_step = float(((uint8_to_unit(va.pixels[t1_va[:n]]) - uint8_to_unit(va.pixels[t_va[:n]])) ** 2).mean())
        follow = ((decoded_frames(decoder, p_other[:n], device) - decoded_frames(decoder, p_va[:n], device)) ** 2).mean()
        m["action_follow_frame"] = float(follow) / max(frame_step, 1e-12)
        if sieve_cfg is not None:
            m["recon_vs_pred_gradient"] = gradient_ratio(jepa, tr, t_tr, t1_tr, sieve_cfg, device)

    log("probes")
    y_tr, y_va = pr.targets(tr), pr.targets(va)
    chance = pr.chance(y_va)
    mlp = pr.Probe("mlp", z_tr, y_tr, device, steps=probe_steps)
    linear = pr.Probe("linear", z_tr, y_tr, device, steps=probe_steps)
    q_mlp, q_lin = mlp.predict(z_va), linear.predict(z_va)
    s_mlp, s_lin = pr.scores(q_mlp, y_va), pr.scores(q_lin, y_va)
    ci = pr.bootstrap(q_mlp, y_va, va.episode, pr.HEADLINE)
    ratios = []
    for name, value in s_mlp.items():
        m[f"holds_{name}"] = value
        m[f"holds_{name}_linear"] = s_lin[name]
        m[f"chance_{name}"] = chance[name]
        if name in ci:
            m[f"holds_{name}_ci"] = ci[name]
        if name in pr.HEADLINE and np.isfinite(value) and pr.skill(value, chance[name]) > 0.05:
            ratios.append(pr.skill(s_lin[name], chance[name]) / pr.skill(value, chance[name]))
    m["linear_vs_mlp"] = float(np.mean(ratios)) if ratios else float("nan")
    y_t = {k: v[t_va] for k, v in y_va.items()}
    y_1 = {k: v[t1_va] for k, v in y_va.items()}
    m.update(pr.foresees(mlp.predict(p_va), {k: v[t_va] for k, v in q_mlp.items()}, y_t, y_1))
    m["action_follow_facing"] = float((mlp.predict(p_move)["facing"] == move.astype(int) - 1).mean())

    log("counterfactuals")
    m.update(_moves(jepa, bank, device))

    if fresh:
        log("fresh decoder")
        e_fresh = fresh_decoder(z_tr, tr.pixels, z_va, va.pixels, device, steps=fresh_steps)
        m["fresh_decoder_error"] = float(e_fresh.mean())
        m["fresh_decoder_worst1pct"] = float(np.percentile(e_fresh, 99))
    m["seconds"] = time.time() - t0
    return m
