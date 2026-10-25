"""Block A: what the encoder keeps of each entity, where along the chain it is lost, and the tests of why.

The chain of a view tile: its 7x7 pixels (the ceiling) -> its patch token after some ViT layers and after the last one
(the untrained ViT's token is the control: how much passes through without training) -> the CLS token before the
projector -> the latent. The frame-level features (mean of the patch tokens, CLS, latent, untrained latent) read the
whole 7x9 map at once (17 block classes and 4 mob flags per tile), so the same per-class recall is measured at every
stage. Inventory and vitals are drawn in the strip below the map: their chain is the strip's pixels -> its patch tokens
-> CLS -> latent.

Every trained probe comes twice: with the plain loss (favours the common classes) and class-balanced (asks whether a
rare class can be told apart at all). The chain verdicts use the balanced MLP.
"""

from __future__ import annotations

import numpy as np
import torch

from . import readout as ro
from . import targets as tg
from .collect import EncoderTap, paste, vit_of

FRAME_FEATURES = {
    "mean_patches": "mean of the map's patch tokens",
    "cls": "CLS before the projector",
    "latent": "latent (CLS after the projector)",
    "control": "untrained latent",
}
PROBES = [("linear", False), ("mlp", False), ("linear", True), ("mlp", True)]  # (kind, class-balanced)
RARE_SHARE = 0.01  # block types under this share of the view tiles count as rare (the spectrum)


def probe_name(kind: str, balanced: bool) -> str:
    return kind + (" balanced" if balanced else "")


def _clip_view(split: dict, x: torch.Tensor) -> torch.Tensor:
    return x.reshape(split["clips"], split["T"], *x.shape[1:])


# --------------------------------------------------------------------------
# state targets


def state_probes(train: dict, val: dict, *, steps: int, device: str, boot: int, seed: int = 0) -> tuple[dict, dict]:
    """Ridge probes on the latent, the CLS token and the untrained latent, an MLP on the latent and the untrained
    latent, per target and column; intervals of the latent's ridge scores. Returns (results, the latent's ridge
    probes)."""
    targets = tg.TARGETS
    ytr = {n: train["y"][n] for n in targets}
    yva = {n: val["y"][n] for n in targets}
    ridge, fitted_latent, preds_latent = {}, None, None
    for feature in ("latent", "cls", "control"):
        fitted = ro.fit_targets(train["frames"][feature], ytr, targets, seed=seed, groups=train["clip"])
        preds = {n: f["predict"](val["frames"][feature]) for n, f in fitted.items()}
        scored = ro.score_predictions(preds, yva, targets)
        for n, f in fitted.items():
            scored["targets"][n].update(**{"lambda": f["lambda"], "at_edge": f["at_edge"]})
        ridge[feature] = scored
        if feature == "latent":
            fitted_latent, preds_latent = fitted, preds
    heads = [
        ro.Head(n, s["kind"], 1 if s["kind"] == "cls" else len(s["names"]), s.get("classes", 1))
        for n, s in targets.items()
    ]
    head_targets = {n: (ytr[n].long() if s["kind"] == "cls" else ytr[n]) for n, s in targets.items()}
    mlp = {}
    for feature in ("latent", "control"):
        predict = ro.train_heads(
            "mlp", train["frames"][feature], head_targets, heads, steps=steps, device=device, seed=seed
        )
        out = predict(val["frames"][feature])
        preds = {n: (out[n][:, 0] if s["kind"] == "cls" else out[n]) for n, s in targets.items()}
        mlp[feature] = ro.score_predictions(preds, yva, targets)
    intervals = {}
    for n, spec in targets.items():

        def stat(idx, n=n, spec=spec):
            return ro.score_prediction(spec, preds_latent[n][idx], yva[n][idx])["score"]

        intervals[n] = ro.episode_bootstrap(stat, val["episode"], boot, seed)
    edge = sorted(n for n, t in ridge["latent"]["targets"].items() if t.get("at_edge"))
    return {"ridge": ridge, "mlp": mlp, "intervals": intervals, "lambda_at_edge": edge}, fitted_latent


def upper_levels(train: dict, val: dict, seed: int = 0) -> dict:
    """Ridge probes of every level on the frames all levels share (a model with one level: empty)."""
    out = {}
    for n in sorted(train.get("levels") or {}):
        (ztr, ytr), (zva, yva) = train["levels"][n], val["levels"][n]
        fitted = ro.fit_targets(ztr, ytr, tg.TARGETS, seed=seed)
        out[n] = ro.score_targets(fitted, zva, yva, tg.TARGETS)
        out[n]["n_train"], out[n]["dim"] = len(ztr), ztr.shape[1]
    return out


# --------------------------------------------------------------------------
# the tile maps from frame-level features


TILE_HEADS = [
    ro.Head("blocks", "cls", tg.VIEW_TILES, tg.NUM_BLOCKS),
    ro.Head("mobs", "bin", tg.VIEW_TILES * tg.NUM_MOBS),
]


def tile_targets(split: dict) -> dict:
    """The map targets of every frame; the player's tile is not scored (the player is drawn over it)."""
    blocks = split["y"]["blocks"].clone()
    blocks[:, tg.PLAYER_INDEX] = -1
    mobs = split["y"]["mobs"].float().clone()
    mobs[:, tg.PLAYER_INDEX] = float("nan")
    return {"blocks": blocks, "mobs": mobs.reshape(len(mobs), -1)}


def evaluate_tiles(pred_blocks: torch.Tensor, mob_scores: torch.Tensor, split: dict, min_n: int) -> dict:
    """Per-class recall (all tiles, per ring, the faced tile), accuracy, balanced accuracy, mob AUROC (all tiles and per
    ring) and the confusion matrix of a map prediction (N, 63) / (N, 63, 4) against the split's maps."""
    blocks, mobs = split["y"]["blocks"], split["y"]["mobs"]
    rings = tg.tile_rings()
    scored = torch.arange(tg.VIEW_TILES) != tg.PLAYER_INDEX
    y, p = blocks[:, scored], pred_blocks[:, scored]
    n, recall = ro.class_recalls(y, p, tg.NUM_BLOCKS)
    out = {
        "accuracy": float((y == p).float().mean()),
        "balanced": ro.balanced_accuracy(n, recall, min_n),
        "n": n.tolist(),
        "recall": recall.tolist(),
        "confusion": ro.confusion(y, p, tg.NUM_BLOCKS).tolist(),
        "rings": {},
        "mob_auroc": [
            ro.auroc(mobs[:, scored, k].flatten().numpy(), mob_scores[:, scored, k].flatten().numpy())
            for k in range(tg.NUM_MOBS)
        ],
        "mob_rings": {},
    }
    for r in tg.RINGS:
        here = rings == r
        nr, rr = ro.class_recalls(blocks[:, here], pred_blocks[:, here], tg.NUM_BLOCKS)
        out["rings"][r] = {"n": nr.tolist(), "recall": rr.tolist()}
        out["mob_rings"][r] = [
            ro.auroc(mobs[:, here, k].flatten().numpy(), mob_scores[:, here, k].flatten().numpy())
            for k in range(tg.NUM_MOBS)
        ]
    faced = tg.faced_index(split["y"]["facing_raw"])
    rows = torch.arange(len(blocks))
    nf, rf = ro.class_recalls(blocks[rows, faced], pred_blocks[rows, faced], tg.NUM_BLOCKS)
    out["faced"] = {
        "n": nf.tolist(),
        "recall": rf.tolist(),
        "accuracy": float((blocks[rows, faced] == pred_blocks[rows, faced]).float().mean()),
    }
    return out


def map_decode(train: dict, val: dict, *, steps: int, device: str, boot: int, min_n: int, seed: int = 0):
    """Every frame-level feature x {linear, MLP} x {plain, balanced} reads the whole map. Returns (results, decoders):
    the decoders are the plain MLPs of the latent and the CLS token and the plain linear probe of the latent, for the
    predictions, the swaps and the events."""
    ttr = tile_targets(train)
    results, decoders = {}, {}
    for feature in FRAME_FEATURES:
        results[feature] = {}
        for kind, balanced in PROBES:
            predict = ro.train_heads(
                kind,
                train["frames"][feature],
                ttr,
                TILE_HEADS,
                steps=steps,
                balanced=balanced,
                device=device,
                seed=seed,
            )
            out = predict(val["frames"][feature])
            pred = out["blocks"].argmax(-1)
            scores = out["mobs"].reshape(len(pred), tg.VIEW_TILES, tg.NUM_MOBS)
            entry = evaluate_tiles(pred, scores, val, min_n)
            if kind == "mlp" and balanced:
                entry["recall_interval"] = _recall_interval(pred, val, boot, seed)
            results[feature][probe_name(kind, balanced)] = entry
            if not balanced and (feature, kind) in (("latent", "mlp"), ("cls", "mlp"), ("latent", "linear")):
                decoders[f"{feature} {kind}"] = predict
    return results, decoders


def _recall_interval(pred: torch.Tensor, split: dict, boot: int, seed: int):
    scored = torch.arange(tg.VIEW_TILES) != tg.PLAYER_INDEX
    y, p = split["y"]["blocks"][:, scored], pred[:, scored]

    def stat(idx):
        return ro.class_recalls(y[idx], p[idx], tg.NUM_BLOCKS)[1]

    interval = ro.episode_bootstrap(stat, split["episode"], boot, seed)
    return None if interval is None else [interval[0].tolist(), interval[1].tolist()]


# --------------------------------------------------------------------------
# the tile chain on the token sample


def token_features(split: dict) -> dict:
    """The token-level stages that the sample holds, in chain order: pixels, layers, last layer, untrained."""
    tok = split["tokens"]
    names = ["raw"] + [f"layer{layer}" for layer in split["layers"]] + ["final", "control"]
    return {n: tok[n] for n in names if n in tok}


TOKEN_LABELS = {
    "raw": "the tile's pixels",
    "final": "patch token, last layer",
    "control": "untrained ViT's patch token",
}


def token_label(name: str) -> str:
    return TOKEN_LABELS.get(name) or (f"patch token after layer {name[5:]}" if name.startswith("layer") else name)


def token_probes(train: dict, val: dict, *, steps: int, device: str, boot: int, min_n: int, seed: int = 0) -> dict:
    """The block type and the mob flags of a tile from each token-level stage. Plain probes train on the natural
    frequencies (the sample's weights), balanced ones on the capped sample with class weights; every score is weighted
    back to the natural frequencies."""
    ttr, tva = train["tokens"], val["tokens"]
    if not ttr or not tva:
        return {}
    heads = [ro.Head("block", "cls", 1, tg.NUM_BLOCKS), ro.Head("mobs", "bin", tg.NUM_MOBS)]
    targets = {"block": ttr["block"][:, None], "mobs": ttr["mobs"].float()}
    w_va = tva["weight"]
    out = {}
    for name, xtr in token_features(train).items():
        xva = tva[name].float()
        out[name] = {}
        for kind, balanced in PROBES:
            predict = ro.train_heads(
                kind,
                xtr.float(),
                targets,
                heads,
                steps=steps,
                balanced=balanced,
                sample_weight=None if balanced else ttr["weight"],
                device=device,
                seed=seed,
            )
            pred = predict(xva)
            p = pred["block"][:, 0].argmax(-1)
            n, recall = ro.class_recalls(tva["block"], p, tg.NUM_BLOCKS, w_va)
            entry = {
                "accuracy": float((w_va * (p == tva["block"]).float()).sum() / w_va.sum()),
                "balanced": ro.balanced_accuracy(n, recall, min_n),
                "n": n.tolist(),
                "recall": recall.tolist(),
                "confusion": ro.confusion(tva["block"], p, tg.NUM_BLOCKS, w_va).tolist(),
                "mob_auroc": [
                    ro.auroc(tva["mobs"][:, k].numpy(), pred["mobs"][:, k].numpy(), w_va.numpy())
                    for k in range(tg.NUM_MOBS)
                ],
                "rings": {},
            }
            for r in tg.RINGS:
                here = tva["ring"] == r
                nr, rr = ro.class_recalls(tva["block"][here], p[here], tg.NUM_BLOCKS, w_va[here])
                entry["rings"][r] = {"n": nr.tolist(), "recall": rr.tolist()}
            faced = tva["faced"]
            nf, rf = ro.class_recalls(tva["block"][faced], p[faced], tg.NUM_BLOCKS, w_va[faced])
            entry["faced"] = {"n": nf.tolist(), "recall": rf.tolist()}
            if kind == "mlp" and balanced:

                def stat(idx, p=p):
                    return ro.class_recalls(tva["block"][idx], p[idx], tg.NUM_BLOCKS, w_va[idx])[1]

                interval = ro.episode_bootstrap(stat, tva["episode"], boot, seed)
                entry["recall_interval"] = None if interval is None else [interval[0].tolist(), interval[1].tolist()]
            out[name][probe_name(kind, balanced)] = entry
    return out


# --------------------------------------------------------------------------
# the inventory strip


STRIP_STAGES = {
    "raw": "the strip's pixels",
    "tokens": "the strip's patch tokens",
    "cls": "CLS before the projector",
    "latent": "latent",
}


def strip_chain(train: dict, val: dict, seed: int = 0) -> dict:
    """Ridge R^2 of each inventory item and vital from the strip's pixels, its patch tokens, the CLS token and the
    latent, on the thinned frames."""
    if not train.get("strip") or not val.get("strip"):
        return {}
    out = {}
    itr, iva = train["strip"]["index"], val["strip"]["index"]
    stages = {
        "raw": (train["strip"]["raw"].float(), val["strip"]["raw"].float()),
        "tokens": (train["strip"]["tokens"].float(), val["strip"]["tokens"].float()),
        "cls": (train["frames"]["cls"][itr], val["frames"]["cls"][iva]),
        "latent": (train["frames"]["latent"][itr], val["frames"]["latent"][iva]),
    }
    for stage, (xtr, xva) in stages.items():
        out[stage] = {}
        for target in ("inventory", "vitals"):
            predict, lam, edge = ro.fit_with_validation(xtr, train["y"][target][itr], seed, train["clip"][itr])
            r2 = ro.r2_columns(val["y"][target][iva], predict(xva))
            out[stage][target] = {"columns": dict(zip(tg.TARGETS[target]["names"], r2)), "lambda": lam, "at_edge": edge}
    out["n_train"], out["n_val"] = len(itr), len(iva)
    return out


# --------------------------------------------------------------------------
# why: frequency, look-alike sprites, the spectrum


def tile_shares(*splits) -> np.ndarray:
    """Share of the view tiles (the player's left out) of each block type, over the splits' frames."""
    counts = torch.zeros(tg.NUM_BLOCKS)
    scored = torch.arange(tg.VIEW_TILES) != tg.PLAYER_INDEX
    for s in splits:
        counts += torch.bincount(s["y"]["blocks"][:, scored].flatten(), minlength=tg.NUM_BLOCKS).float()
    return (counts / counts.sum().clamp_min(1)).numpy()


def sprite_distances(sprites: dict) -> np.ndarray:
    """Mean absolute pixel difference (0 to 1) between the first library crop of each pair of block types; nan where a
    type has no crop."""
    d = np.full((tg.NUM_BLOCKS, tg.NUM_BLOCKS), np.nan)
    crops = {c: v[0].float() / 255 for c, v in sprites.get("blocks", {}).items() if v}
    for i, a in crops.items():
        for j, b in crops.items():
            d[i, j] = float((a - b).abs().mean())
    return d


def causes(maps: dict, tokens: dict, shares: np.ndarray, distances: np.ndarray, min_n: int) -> dict:
    """Does a stage lose a block type because it is rare, or because its sprite looks like another one?

    Per stage (balanced MLP): the rank correlation over block types of the recall with the log share and with the
    distance to the nearest other sprite, and a standardised least-squares fit of the recall on both; over pairs of
    types, the rank correlation of the confusion rate with the sprite distance (look-alikes are confused more: negative)
    and with the share of the type confused for."""
    nearest = np.full(tg.NUM_BLOCKS, np.nan)
    nearest_to = [None] * tg.NUM_BLOCKS
    for i in range(tg.NUM_BLOCKS):
        row = distances[i].copy()
        row[i] = np.nan
        if not np.all(np.isnan(row)):
            j = int(np.nanargmin(row))
            nearest[i], nearest_to[i] = row[j], tg.BLOCK_NAMES[j]
    stages = {}
    sources = {
        "latent": maps.get("latent"),
        "cls": maps.get("cls"),
        "final": tokens.get("final"),
        "raw": tokens.get("raw"),
    }
    for stage, res in sources.items():
        if not res:
            continue
        r = res["mlp balanced"]
        n, recall = np.array(r["n"]), np.array(r["recall"], dtype=np.float64)
        ok = (n >= min_n) & ~np.isnan(recall) & (shares > 0)
        entry = {
            "classes": int(ok.sum()),
            "recall_vs_log_share": ro.spearman(recall[ok], np.log(shares[ok])),
            "recall_vs_nearest_sprite": ro.spearman(recall[ok], nearest[ok]),
            "fit": _standardised_fit(recall[ok], {"log_share": np.log(shares[ok]), "nearest_sprite": nearest[ok]}),
        }
        conf = np.array(r["confusion"], dtype=np.float64)
        rates, dist, share_to = [], [], []
        for i in np.flatnonzero(ok):
            total = conf[i].sum()
            for j in np.flatnonzero(ok):
                if i != j and total > 0 and not np.isnan(distances[i, j]):
                    rates.append(conf[i, j] / total)
                    dist.append(distances[i, j])
                    share_to.append(shares[j])
        entry["pairs"] = len(rates)
        entry["confusion_vs_sprite_distance"] = ro.spearman(rates, dist)
        entry["confusion_vs_share_of_target"] = ro.spearman(rates, share_to)
        stages[stage] = entry
    return {
        "stages": stages,
        "nearest": nearest.tolist(),
        "nearest_to": nearest_to,
        "shares": shares.tolist(),
        "distances": distances.tolist(),
    }


def _standardised_fit(y: np.ndarray, xs: dict) -> dict | None:
    """Least squares of y on the standardised xs (with an intercept): coefficients and R^2; None with too few points."""
    keep = ~np.isnan(y)
    for v in xs.values():
        keep &= ~np.isnan(v)
    if keep.sum() < len(xs) + 2:
        return None
    cols = []
    for v in xs.values():
        v = v[keep]
        sd = v.std()
        cols.append((v - v.mean()) / sd if sd > 0 else np.zeros_like(v))
    x = np.column_stack([np.ones(int(keep.sum()))] + cols)
    beta, *_ = np.linalg.lstsq(x, y[keep], rcond=None)
    resid = y[keep] - x @ beta
    r2 = 1 - resid.var() / y[keep].var() if y[keep].var() > 0 else None
    return {"coef": dict(zip(xs, beta[1:].tolist())), "r2": None if r2 is None else float(r2), "n": int(keep.sum())}


def spectrum(
    train: dict, val: dict, shares: np.ndarray, *, ks, steps: int, device: str, min_n: int, seed: int = 0
) -> dict:
    """Balanced linear map decode from the top-k principal components of the latent (and of the untrained latent):
    where in the latent's variance do the rare block types and the mobs sit?"""
    ttr = tile_targets(train)
    rare = [i for i in range(tg.NUM_BLOCKS) if 0 < shares[i] < RARE_SHARE]
    out = {"rare": [tg.BLOCK_NAMES[i] for i in rare], "features": {}}
    for feature in ("latent", "control"):
        x = train["frames"][feature].float()
        mean = x.mean(0)
        _, s, v = torch.linalg.svd(x - mean, full_matrices=False)
        explained = (s**2).cumsum(0) / (s**2).sum()
        rows = []
        for k in [k for k in ks if k <= x.shape[1]] + ([x.shape[1]] if x.shape[1] not in ks else []):
            basis = v[:k].T
            predict = ro.train_heads(
                "linear", (x - mean) @ basis, ttr, TILE_HEADS, steps=steps, balanced=True, device=device, seed=seed
            )
            pred = predict((val["frames"][feature].float() - mean) @ basis)
            e = evaluate_tiles(
                pred["blocks"].argmax(-1), pred["mobs"].reshape(-1, tg.VIEW_TILES, tg.NUM_MOBS), val, min_n
            )
            rare_recall = [e["recall"][i] for i in rare if e["n"][i] >= min_n]
            mobs = [a for a in e["mob_auroc"] if a is not None]
            rows.append(
                {
                    "k": int(k),
                    "variance": float(explained[k - 1]),
                    "balanced": e["balanced"],
                    "rare_recall": float(np.nanmean(rare_recall)) if rare_recall else None,
                    "mob_auroc": float(np.mean(mobs)) if mobs else None,
                }
            )
        out["features"][feature] = rows
    return out


# --------------------------------------------------------------------------
# causal: sprite swaps


@torch.no_grad()
def encode_pixels(
    model, pixels: torch.Tensor, normalize, device: str, chunk: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latent and CLS token of uint8 frames (N, 3, 63, 63)."""
    jepa = model.get_level(1)
    vit = vit_of(jepa)
    was = model.training
    model.eval()
    lat, cls = [], []
    try:
        for a in range(0, len(pixels), chunk):
            batch = {"pixels": pixels[a : a + chunk].to(device)}
            normalize(batch)
            with EncoderTap(vit) as tap:
                z = jepa.encoder(batch["pixels"])
            final, _ = tap.take()
            lat.append(z.float().cpu())
            cls.append(final[:, 0].float().cpu())
    finally:
        model.train(was)
    return torch.cat(lat), torch.cat(cls)


def step_scale(split: dict, feature: str) -> float:
    """Median squared distance between consecutive awake frames: the unit of the latent shifts (the planner's cost is
    the squared distance)."""
    z = _clip_view(split, split["frames"][feature])
    awake = _clip_view(split, split["y"]["sleeping"][:, 0] < 0.5)
    d = (z[:, 1:] - z[:, :-1]).pow(2).sum(-1)[awake[:, :-1] & awake[:, 1:]]
    return float(d.median()) if len(d) else float("nan")


def swap_sensitivity(
    model, val: dict, sprites: dict, decoders: dict, *, normalize, device: str, boot: int, seed: int = 0
) -> dict:
    """Paste one tile of a bright held-out frame with another block type's sprite (or a mob on the tile's block) and
    measure how far the latent and the CLS token move, in units of an ordinary step; and whether the map decoders of the
    latent and the CLS token read the pasted block (or mob) at that tile afterwards. The null swap pastes a crop of the
    same block type from another frame: what lighting and rounding alone move."""
    swap = val.get("swap") or {}
    if not swap or not len(swap.get("frame", [])):
        return {}
    pixels, frames = swap["pixels"], swap["frame"]
    blocks, mobs = val["y"]["blocks"][frames], val["y"]["mobs"][frames]
    episodes = val["episode"][frames]
    scale = {"latent": step_scale(val, "latent"), "cls": step_scale(val, "cls")}
    base_lat, base_cls = encode_pixels(model, pixels, normalize, device)
    g = torch.Generator().manual_seed(seed)
    rings = tg.tile_rings()
    libs = {c: v for c, v in sprites.get("blocks", {}).items() if v}
    plans = [("null", None)] + [
        (f"block:{tg.BLOCK_NAMES[c]}", c) for c in sorted(libs) if tg.BLOCK_NAMES[c] != "invalid"
    ]
    plans += [
        (f"mob:{tg.MOB_NAMES[m]}", m) for m in range(tg.NUM_MOBS) if any(k[0] == m for k in sprites.get("mobs", {}))
    ]
    out = {"scale": scale, "frames": len(frames), "targets": {}}
    for name, c in plans:
        edited, tiles, rows, kinds = [], [], [], name.split(":")[0]
        for j in range(len(frames)):
            free = ~mobs[j].any(-1)
            free[tg.PLAYER_INDEX] = False
            if kinds == "block":
                ok = free & (blocks[j] != c)
            elif kinds == "mob":
                ok = free & torch.tensor([(c, int(b)) in sprites["mobs"] for b in blocks[j]])
            else:
                ok = free & torch.tensor([int(b) in libs for b in blocks[j]])
            cand = torch.nonzero(ok)[:, 0]
            if not len(cand):
                continue
            tile = int(cand[torch.randint(len(cand), (1,), generator=g)])
            lib = (
                libs[c]
                if kinds == "block"
                else sprites["mobs"][(c, int(blocks[j, tile]))]
                if kinds == "mob"
                else libs[int(blocks[j, tile])]
            )
            crop = lib[int(torch.randint(len(lib), (1,), generator=g))]
            edited.append(paste(pixels[j], tile, crop))
            tiles.append(tile)
            rows.append(j)
        if not rows:
            continue
        rows_t, tiles_t = torch.tensor(rows), torch.tensor(tiles)
        lat, cls = encode_pixels(model, torch.stack(edited), normalize, device)
        d_lat = (lat - base_lat[rows_t]).pow(2).sum(-1) / scale["latent"]
        d_cls = (cls - base_cls[rows_t]).pow(2).sum(-1) / scale["cls"]
        entry = {
            "n": len(rows),
            "latent_shift": float(d_lat.median()),
            "cls_shift": float(d_cls.median()),
            # over the swaps that move the CLS token at all (the null swap moves nothing)
            "projector_ratio": float((d_lat / d_cls)[moved].median()) if (moved := d_cls > 1e-6).any() else None,
            "rings": {r: float(d_lat[rings[tiles_t] == r].median()) for r in tg.RINGS if (rings[tiles_t] == r).any()},
        }

        def stat(idx, d=d_lat):
            return float(d[idx].median()) if len(idx) else None

        entry["latent_interval"] = ro.episode_bootstrap(stat, episodes[rows_t].numpy(), boot, seed)
        if kinds != "null":
            for source, (after_z, before_z) in (("latent", (lat, base_lat[rows_t])), ("cls", (cls, base_cls[rows_t]))):
                decoder = decoders.get(f"{source} mlp")
                if decoder is None:
                    continue
                after, before = decoder(after_z), decoder(before_z)
                at = torch.arange(len(rows))
                if kinds == "block":
                    entry[f"{source}_reads_after"] = float(
                        (after["blocks"][at, tiles_t].argmax(-1) == c).float().mean()
                    )
                    entry[f"{source}_reads_before"] = float(
                        (before["blocks"][at, tiles_t].argmax(-1) == c).float().mean()
                    )
                else:
                    col = tiles_t * tg.NUM_MOBS + c
                    entry[f"{source}_reads_after"] = float((after["mobs"][at, col] > 0).float().mean())
                    entry[f"{source}_reads_before"] = float((before["mobs"][at, col] > 0).float().mean())
        out["targets"][name] = entry
    return out


# --------------------------------------------------------------------------
# what the latent distance (the planner's cost) sees


def distance_sensitivity(val: dict, scale: float, gaps=(1, 8)) -> dict:
    """Least squares of the squared latent distance between frames t and t + gap (in units of an ordinary step) on what
    changed between them. Frames of a clip are correlated, so treat the fit as descriptive."""
    y = val["y"]
    view = lambda x: _clip_view(val, x)  # noqa: E731
    z = view(val["frames"]["latent"])
    pos, facing, inv, vit_, light = (
        view(y["position"]).float(),
        view(y["facing_raw"]),
        view(y["inventory"]),
        view(y["vitals"]),
        view(y["daylight"]),
    )
    sleep, mobs, blocks, ach = view(y["sleeping"]), view(y["mobs"]), view(y["blocks"]), view(y["achievements"])
    out = {}
    for gap in gaps:
        if gap >= val["T"]:
            continue
        a, b = slice(0, -gap), slice(gap, None)
        dist = (z[:, b] - z[:, a]).pow(2).sum(-1) / scale
        feats = {
            "moved (tiles)": (pos[:, b] - pos[:, a]).abs().sum(-1),
            "turned": (facing[:, b] != facing[:, a]).float(),
        }
        for i, name in enumerate(tg.INVENTORY_NAMES):
            feats[f"inventory {name}"] = (inv[:, b, i] - inv[:, a, i]).abs()
        for i, name in enumerate(tg.VITAL_NAMES):
            feats[f"vitals {name}"] = (vit_[:, b, i] - vit_[:, a, i]).abs()
        feats["daylight"] = (light[:, b, 0] - light[:, a, 0]).abs()
        feats["sleeping"] = (sleep[:, b, 0] != sleep[:, a, 0]).float()
        for k, name in enumerate(tg.MOB_NAMES):
            feats[f"{name}s in view"] = (mobs[:, b, :, k].sum(-1) - mobs[:, a, :, k].sum(-1)).abs().float()
        feats["view tiles changed"] = (blocks[:, b] != blocks[:, a]).sum(-1).float()
        feats["achievements unlocked"] = (ach[:, b] - ach[:, a]).clamp_min(0).sum(-1)
        feats = {k: v.flatten().double() for k, v in feats.items() if (v != 0).any()}
        target = dist.flatten().double()
        x = torch.stack([torch.ones_like(target)] + list(feats.values()), 1)
        beta = torch.linalg.lstsq(x, target[:, None]).solution[:, 0]
        resid = target - x @ beta
        rows = {}
        for i, (name, v) in enumerate(feats.items()):
            nonzero = v != 0
            typical = float(v[nonzero].mean())
            rows[name] = {
                "coef": float(beta[i + 1]),
                "share_changed": float(nonzero.double().mean()),
                "typical_change": typical,
                "effect": float(beta[i + 1]) * typical,
            }
        out[gap] = {
            "pairs": len(target),
            "intercept": float(beta[0]),
            "r2": float(1 - resid.var() / target.var()),
            "mean_distance": float(target.mean()),
            "features": rows,
        }
    return out
