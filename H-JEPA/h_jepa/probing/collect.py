"""One pass over the clips of a split, with the trained model and an untrained one of the same architecture.

Per frame: the latent (the CLS token after the projector, what the predictor and the planner use), the CLS token before
the projector, the mean of the map's patch tokens, the untrained latent, the action, and the targets. Per view tile, a
stratified sample (each block type and each mob type capped, with weights that restore the natural frequencies): the
tile's 7x7 pixels, its patch token after some layers and after the last one, and the untrained ViT's patch token. On
every `strip_every`-th frame the inventory strip (its pixels and its 18 patch tokens). On the held-out split, bright
frames for the sprite swaps; on both, the sprite library (the tile crops of each block type and of each mob on a block).
"""

from __future__ import annotations

import torch

from . import targets as tg

SWAP_DAYLIGHT = 0.99  # frames this bright show the plain textures (below 1 the renderer blends in a night tint)
SPRITES_PER_CLASS = 8


def vit_of(jepa):
    """The level-1 ViT (a transformers ViTModel) behind the projector."""
    encoder = getattr(jepa.encoder, "encoder", None)
    if encoder is None:
        raise ValueError("the probe suite needs a pixel encoder with a projector (ProjectedEncoder): no proprio fusion")
    return encoder


class EncoderTap:
    """Captures the ViT's tokens during forwards: `final` (N, 1 + patches, H) after the last layer norm, and the
    residual stream after each of `layers` (1-based; a layer the model does not have is skipped)."""

    def __init__(self, vit, layers=()):
        self.vit, self.layers = vit, tuple(int(layer) for layer in layers)
        self.final, self.by_layer, self.handles = [], {}, []

    def __enter__(self):
        self.handles.append(self.vit.register_forward_hook(self._final))
        # the transformer blocks: `encoder.layer` in transformers 4 (the pinned version), `layers` in 5
        blocks = getattr(getattr(self.vit, "encoder", None), "layer", None) or getattr(self.vit, "layers", None)
        for layer in self.layers:
            if blocks is not None and 1 <= layer <= len(blocks):
                self.by_layer[layer] = []
                self.handles.append(blocks[layer - 1].register_forward_hook(self._layer(layer)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _final(self, module, args, output):
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        self.final.append(hidden.detach())

    def _layer(self, layer):
        def hook(module, args, output):
            self.by_layer[layer].append((output[0] if isinstance(output, tuple) else output).detach())

        return hook

    def take(self) -> tuple[torch.Tensor, dict]:
        final = torch.cat(self.final) if self.final else None
        layers = {layer: torch.cat(v) for layer, v in self.by_layer.items() if v}
        self.final, self.by_layer = [], {layer: [] for layer in self.by_layer}
        return final, layers


def tile_crops(raw: torch.Tensor) -> torch.Tensor:
    """The 7x7 pixels of each map tile: (N, 3, 63, 63) -> (N, 63, 147)."""
    n = raw.shape[0]
    top = raw[:, :, : tg.VIEW_ROWS * tg.TILE]
    tiles = top.reshape(n, 3, tg.VIEW_ROWS, tg.TILE, tg.VIEW_COLS, tg.TILE).permute(0, 2, 4, 1, 3, 5)
    return tiles.reshape(n, tg.VIEW_TILES, 3 * tg.TILE * tg.TILE)


def paste(raw: torch.Tensor, tile: int, crop: torch.Tensor) -> torch.Tensor:
    """A copy of one frame (3, 63, 63) with the 7x7 crop (3, 7, 7) at view tile `tile`."""
    out = raw.clone()
    r, c = divmod(int(tile), tg.VIEW_COLS)
    out[:, r * tg.TILE : (r + 1) * tg.TILE, c * tg.TILE : (c + 1) * tg.TILE] = crop
    return out


def crop_of(raw: torch.Tensor, tile: int) -> torch.Tensor:
    r, c = divmod(int(tile), tg.VIEW_COLS)
    return raw[:, r * tg.TILE : (r + 1) * tg.TILE, c * tg.TILE : (c + 1) * tg.TILE].clone()


def frame_of_step(level: int, steps: torch.Tensor, cfg) -> torch.Tensor:
    """Level-1 frame whose latent chunk ends the level-`level` step `steps`."""
    index = steps
    for n in range(level, 1, -1):
        lv = cfg[f"level{n}"]
        index = index * int(lv.stride) + int(lv.get("window_size", 1)) - 1
    return index


def common_frames(num_levels: int, cfg, num_frames: int) -> torch.Tensor:
    """Level-1 frames at which every level has a latent, so that all levels are probed on the same states."""
    common = None
    for n in range(1, num_levels + 1):
        frames = frame_of_step(n, torch.arange(num_frames), cfg)
        frames = frames[frames < num_frames]
        common = frames if common is None else common[torch.isin(common, frames)]
    if common is None or not len(common):
        raise ValueError("the levels have no level-1 frame in common (check stride and window_size)")
    return common


def clip_episodes(loader) -> list[int]:
    """Episode of each clip of a loader over a Subset of the HDF5 dataset (for the episode bootstrap)."""
    subset = loader.dataset
    return [int(subset.dataset.clip_indices[i][0]) for i in subset.indices]


class _Reservoir:
    """Keeps a uniform sample of at most `cap` entries per stratum (the smallest random keys) and counts what it saw."""

    def __init__(self, cap: int, seed: int):
        self.cap, self.g = int(cap), torch.Generator().manual_seed(seed)
        self.keys, self.data, self.seen = {}, {}, {}

    def offer(self, stratum: int, n: int):
        """Draws keys for `n` new entries; returns (kept old positions, kept new positions)."""
        keys = torch.rand(n, generator=self.g)
        self.seen[stratum] = self.seen.get(stratum, 0) + n
        old = self.keys.get(stratum, torch.zeros(0))
        merged = torch.cat([old, keys])
        keep = (
            torch.arange(len(merged))
            if len(merged) <= self.cap
            else torch.topk(merged, self.cap, largest=False).indices
        )
        keep_old, keep_new = keep[keep < len(old)], keep[keep >= len(old)] - len(old)
        self.keys[stratum] = torch.cat([old[keep_old], keys[keep_new]])
        return keep_old, keep_new

    def store(self, stratum: int, keep_old: torch.Tensor, new: dict):
        old = self.data.get(stratum)
        self.data[stratum] = new if old is None else {k: torch.cat([v[keep_old], new[k]]) for k, v in old.items()}

    def result(self) -> dict:
        if not self.data:
            return {}
        names = next(iter(self.data.values())).keys()
        out = {k: torch.cat([self.data[s][k] for s in sorted(self.data)]) for k in names}
        out["weight"] = torch.cat(
            [torch.full((len(self.keys[s]),), self.seen[s] / max(len(self.keys[s]), 1)) for s in sorted(self.data)]
        )
        out["seen"] = {int(s): int(v) for s, v in self.seen.items()}
        return out


@torch.no_grad()
def collect_split(
    model,
    control_model,
    loader,
    cfg,
    device,
    normalize,
    *,
    episodes,
    layers=(3, 6, 9),
    token_cap=4000,
    strip_every=4,
    swap_frames=0,
    seed=0,
) -> dict:
    """Everything the suite reads from one split (see the module docstring), on the CPU. Frames are stored clip by clip,
    so frame arrays reshape to (clips, T, ...)."""
    from hjepa_utils import _discrete_idm_target

    jepa, control = model.get_level(1), control_model.get_level(1)
    vit, control_vit = vit_of(jepa), vit_of(control)
    jepas = [model.get_level(n) for n in range(1, model.num_levels + 1)] + [control]
    saved = [j.target_length for j in jepas]
    modes = (model.training, control_model.training)
    for j in jepas:
        j.target_length = None  # keep every frame of the clip
    model.eval()
    control_model.eval()
    rings = tg.tile_rings()
    tokens = _Reservoir(token_cap, seed)
    swaps = _Reservoir(swap_frames, seed + 1) if swap_frames else None
    sprites = {"blocks": {}, "mobs": {}}
    keep = {k: [] for k in ("latent", "cls", "mean_patches", "control", "act", "action")}
    ys, sources, strip, levels = {}, [], {"index": [], "raw": [], "tokens": []}, {}
    clip_count, frame_count, T = 0, 0, None
    try:
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            raw = batch["pixels_level1"].cpu()
            B, T = raw.shape[:2]
            y = tg.frame_targets(batch)
            source = batch.get("source_level1")
            normalize(batch)
            with EncoderTap(vit, layers) as tap:
                out = model.encode_hierarchical(dict(batch))
            final, by_layer = tap.take()
            with EncoderTap(control_vit) as ctap:
                cout = control_model.encode_hierarchical(dict(batch))
            cfinal, _ = ctap.take()
            n = B * T
            assert final is not None and len(final) == n, (
                "the ViT ran on a different number of frames than the clips hold"
            )
            final, cfinal = final.float().cpu(), cfinal.float().cpu()
            by_layer = {layer: v.float().cpu() for layer, v in by_layer.items()}
            keep["latent"].append(out["embed_1"].float().flatten(0, 1).cpu())
            keep["cls"].append(final[:, 0])
            keep["mean_patches"].append(final[:, 1 : 1 + tg.VIEW_TILES].mean(1))
            keep["control"].append(cout["embed_1"].float().flatten(0, 1).cpu())
            keep["act"].append(out["action_1"].float().flatten(0, 1).cpu())
            index = _discrete_idm_target(out, 1, cfg.level1)
            keep["action"].append(index[..., 0].flatten().cpu() if index is not None else torch.full((n,), -1))
            for k, v in y.items():
                ys.setdefault(k, []).append(v.flatten(0, 1))
            if source is not None:
                sources.append(source.reshape(n, -1)[:, 0].cpu().long())
            if model.num_levels > 1:
                _collect_levels(levels, out, y, model.num_levels, cfg, T)

            # the tile sample
            flat_raw = raw.flatten(0, 1)
            crops = tile_crops(flat_raw)
            blocks, mobs = y["blocks"].flatten(0, 1), y["mobs"].flatten(0, 1)
            faced = tg.faced_index(y["facing_raw"].flatten(0, 1))
            strata = tg.tile_strata(blocks, mobs)
            strata[:, tg.PLAYER_INDEX] = -1
            frame_episode = torch.tensor(episodes[clip_count : clip_count + B]).repeat_interleave(T)
            for s in torch.unique(strata).tolist():
                if s < 0:
                    continue
                fi, ti = torch.nonzero(strata == s, as_tuple=True)
                keep_old, keep_new = tokens.offer(s, len(fi))
                fi, ti = fi[keep_new], ti[keep_new]
                entry = {
                    "raw": crops[fi, ti].half(),
                    "final": final[fi, 1 + ti].half(),
                    "control": cfinal[fi, 1 + ti].half(),
                    "block": blocks[fi, ti],
                    "mobs": mobs[fi, ti],
                    "ring": rings[ti],
                    "faced": ti == faced[fi],
                    "episode": frame_episode[fi],
                    "frame": frame_count + fi,
                }
                for layer, v in by_layer.items():
                    entry[f"layer{layer}"] = v[fi, 1 + ti].half()
                tokens.store(s, keep_old, entry)

            # the inventory strip, every strip_every-th frame
            t_index = torch.arange(T).repeat(B)
            thin = torch.nonzero(t_index % max(strip_every, 1) == 0)[:, 0]
            strip["index"].append(frame_count + thin)
            strip["raw"].append(flat_raw[thin][:, :, tg.VIEW_ROWS * tg.TILE :].reshape(len(thin), -1).half())
            strip["tokens"].append(
                final[thin][:, 1 + tg.VIEW_TILES : 1 + tg.VIEW_TILES + tg.STRIP_PATCHES].reshape(len(thin), -1).half()
            )

            # bright, awake frames: sprite library and swap frames
            bright = (y["daylight"].flatten() >= SWAP_DAYLIGHT) & (y["sleeping"].flatten() < 0.5)
            bright_frames = torch.nonzero(bright)[:, 0]
            _add_sprites(sprites, flat_raw, blocks, mobs, bright_frames)
            if swaps is not None and len(bright_frames):
                keep_old, keep_new = swaps.offer(0, len(bright_frames))
                chosen = bright_frames[keep_new]
                swaps.store(0, keep_old, {"pixels": flat_raw[chosen].clone(), "frame": frame_count + chosen})
            clip_count += B
            frame_count += n
    finally:
        for j, t in zip(jepas, saved):
            j.target_length = t
        model.train(modes[0])
        control_model.train(modes[1])

    frames = {k: torch.cat(v) for k, v in keep.items() if k not in ("act", "action")}
    swap = swaps.result() if swaps is not None else {}
    sample = tokens.result()
    return {
        "frames": frames,
        "act": torch.cat(keep["act"]),
        "action": torch.cat(keep["action"]).long(),
        "y": {k: torch.cat(v) for k, v in ys.items()},
        "source": torch.cat(sources) if sources else None,
        "clips": clip_count,
        "T": T,
        "clip": torch.arange(clip_count).repeat_interleave(T) if T else torch.zeros(0, dtype=torch.long),
        "episode": torch.tensor(episodes[:clip_count]).repeat_interleave(T) if T else torch.zeros(0, dtype=torch.long),
        "clip_episode": torch.tensor(episodes[:clip_count]),
        "tokens": sample,
        "strip": {k: torch.cat(v) for k, v in strip.items()} if strip["index"] else {},
        "swap": {"pixels": swap["pixels"], "frame": swap["frame"]} if swap else {},
        "sprites": sprites,
        "levels": {n: (torch.cat(z), {k: torch.cat(v) for k, v in yy.items()}) for n, (z, yy) in levels.items()},
        "layers": sorted(int(k[5:]) for k in sample if k.startswith("layer")),
    }


def _collect_levels(levels: dict, out: dict, y: dict, num_levels: int, cfg, T: int):
    """The latents of every level at the level-1 frames all levels share (for the state probes of the upper levels)."""
    common = common_frames(num_levels, cfg, T)
    for n in range(1, num_levels + 1):
        emb = out[f"embed_{n}"].float().cpu()
        frames = frame_of_step(n, torch.arange(emb.shape[1]), cfg)
        steps = torch.arange(emb.shape[1])[torch.isin(frames, common)]
        z, yy = levels.setdefault(n, ([], {}))
        z.append(emb[:, steps].flatten(0, 1))
        for k in tg.TARGETS:
            yy.setdefault(k, []).append(y[k][:, frame_of_step(n, steps, cfg)].flatten(0, 1))


def _add_sprites(sprites: dict, raw: torch.Tensor, blocks: torch.Tensor, mobs: torch.Tensor, frames: torch.Tensor):
    """Adds tile crops of bright frames to the library until each block type (no mob on it) and each (mob, block below)
    has SPRITES_PER_CLASS crops; the player's tile is left out (the player is drawn over it)."""
    if not len(frames):
        return
    b, m = blocks[frames], mobs[frames]
    free = ~m.any(-1)
    free[:, tg.PLAYER_INDEX] = False
    for c in torch.unique(b[free]).tolist():
        have = sprites["blocks"].setdefault(int(c), [])
        if len(have) >= SPRITES_PER_CLASS:
            continue
        fi, ti = torch.nonzero(free & (b == c), as_tuple=True)
        step = max(1, len(fi) // SPRITES_PER_CLASS)
        for f, t in list(zip(fi.tolist(), ti.tolist()))[::step][: SPRITES_PER_CLASS - len(have)]:
            have.append(crop_of(raw[frames[f]], t))
    alone = m.sum(-1) == 1
    alone[:, tg.PLAYER_INDEX] = False
    for k in range(tg.NUM_MOBS):
        here = alone & m[..., k]
        for c in torch.unique(b[here]).tolist():
            have = sprites["mobs"].setdefault((k, int(c)), [])
            if len(have) >= SPRITES_PER_CLASS:
                continue
            fi, ti = torch.nonzero(here & (b == c), as_tuple=True)
            for f, t in list(zip(fi.tolist(), ti.tolist()))[: SPRITES_PER_CLASS - len(have)]:
                have.append(crop_of(raw[frames[f]], t))
