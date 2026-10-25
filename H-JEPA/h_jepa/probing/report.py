"""The markdown report of the probe suite (`health/probe_report.md`), in the order of the chain."""

from __future__ import annotations

import math

import health

from . import targets as tg
from .encoder import FRAME_FEATURES, STRIP_STAGES, token_label

PREVIEW_K = 8  # the rollout step of the per-column change tables


def _f(x, digits=2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return "-" if math.isnan(x) else f"{x:.{digits}f}"
    return str(x)


def _interval(iv, i=None) -> str:
    if iv is None:
        return "-"
    lo, hi = (iv[0], iv[1]) if i is None else (iv[0][i], iv[1][i])
    return f"{_f(lo)} to {_f(hi)}"


def _table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + [
        "| " + " | ".join(_f(c) for c in r) + " |" for r in rows
    ]


def render(summary: dict, findings) -> str:
    out = [
        "# Probe suite",
        "",
        "Every entity is followed along the chain pixels -> ViT patch tokens -> CLS -> latent -> one-step prediction "
        "-> "
        "rollout; the verdict is the first link that fails. Probes are fitted on the training worlds and scored on the "
        "held-out worlds; intervals come from resampling whole episodes.",
        "",
        "## Findings",
        "",
        health.findings_block(findings),
    ]
    out += _settings(summary)
    out += _verdicts(summary)
    out += _facts(summary)
    out += ["## A. Encoder", ""]
    out += _state(summary)
    out += _tiles(summary)
    out += _mobs(summary)
    out += _strip(summary)
    out += _causes(summary)
    out += _swaps(summary)
    out += _distance(summary)
    out += ["## B. Predictor", ""]
    out += _loss(summary)
    out += _change(summary)
    out += _events(summary)
    out += _mob_dynamics(summary)
    out += _monotonicity(summary)
    out += _idm(summary)
    return "\n".join(out) + "\n"


def _settings(summary: dict) -> list[str]:
    s = summary.get("settings") or {}
    if not s:
        return []
    return ["## Data", "", ", ".join(f"{k}: {v}" for k, v in s.items()), ""]


def _verdicts(summary: dict) -> list[str]:
    v = summary.get("verdicts") or {}
    if not v:
        return []
    out = [
        "## Verdicts",
        "",
        "Encoder stages: balanced-MLP recall (block types), 2*AUROC-1 (mobs), ridge R^2 (inventory, vitals, other "
        "state). "
        "`lost at` a stage: it keeps less than 80% of the best stage before it; past the encoder a link fails under "
        "0.5 "
        "(swap: share of pastes the latent's map decoder newly reads, after minus before; one step: `shown` of the "
        "event, or the mob motion "
        f"recovered over copying; rollout: `recovered` of the change test at k = {v.get('k')}, planning history). "
        "Heuristic thresholds: read the scores.",
        "",
    ]
    for kind, title, extra in (
        ("blocks", "Block types", ["untrained token", "latent linear", "swap shift"]),
        ("mobs", "Mobs", ["untrained token", "swap shift"]),
        ("items", "Inventory and vitals", ["latent MLP", "event"]),
        ("other", "Other state", ["untrained latent"]),
    ):
        rows = v.get(kind) or []
        if not rows:
            continue
        stages = [name for name, _ in rows[0]["stages"]]
        links = [name for name, _ in rows[0]["links"]]
        keys = {
            "untrained token": "control",
            "latent linear": "latent_linear",
            "swap shift": "swap_shift",
            "latent MLP": "latent_mlp",
            "event": "event",
            "untrained latent": "control",
        }
        body = []
        for r in rows:
            body.append(
                [
                    r["entity"],
                    *[s for _, s in r["stages"]],
                    *[s for _, s in r["links"]],
                    r["verdict"],
                    *[r.get(keys[e]) for e in extra],
                ]
            )
        out += [f"### {title}", ""] + _table(["entity", *stages, *links, "verdict", *extra], body) + [""]
    return out


def _facts(summary: dict) -> list[str]:
    f = summary.get("facts") or {}
    if not f:
        return []
    out = ["## Facts of the probe clips", ""]
    shares = f.get("tile_share") or []
    rows = [[tg.BLOCK_NAMES[c], shares[c]] for c in range(len(shares)) if shares[c] > 0]
    out += (
        ["Share of the view tiles (the player's left out), both splits:", ""] + _table(["block", "share"], rows) + [""]
    )
    mobs = f.get("mobs") or {}
    if mobs:
        out += ["Mobs: share of frames with the mob in view, share of view tiles it stands on:", ""]
        out += _table(["mob", "frames", "tiles"], [[m, e["frames"], e["tiles"]] for m, e in mobs.items()]) + [""]
    events = f.get("events") or {}
    if events:
        out += ["Events per 1000 awake steps (successful / failed presses):", ""]
        out += _table(
            ["event", "successful", "failed"], [[n, e["succeeded"], e["failed"]] for n, e in events.items()]
        ) + [""]
    return out


def _state(summary: dict) -> list[str]:
    st = summary.get("state") or {}
    if not st:
        return []
    ridge, mlp, iv = st["ridge"], st.get("mlp") or {}, st.get("intervals") or {}
    out = [
        "### State targets",
        "",
        "Ridge (linear) and MLP probes from frame features. Score: R^2 for counts and levels, 2*AUROC-1 for flags, "
        "(accuracy-majority)/(1-majority) for classes. Position is not a target (an egocentric frame does not show "
        "it); "
        "mobs are scored as presence.",
        "",
    ]
    rows = []
    for g in tg.GROUPS:
        rows.append(
            [
                g,
                ridge["latent"]["groups"].get(g),
                ridge["cls"]["groups"].get(g),
                ridge["control"]["groups"].get(g),
                (mlp.get("latent") or {}).get("groups", {}).get(g),
                (mlp.get("control") or {}).get("groups", {}).get(g),
            ]
        )
    out += _table(["group", "latent", "CLS", "untrained latent", "latent MLP", "untrained MLP"], rows) + [""]
    rows = []
    for n, t in ridge["latent"]["targets"].items():
        rows.append(
            [
                n,
                t["group"],
                t["score"],
                _interval(iv.get(n)),
                ridge["cls"]["targets"][n]["score"],
                ridge["control"]["targets"][n]["score"],
                (mlp.get("latent") or {}).get("targets", {}).get(n, {}).get("score"),
                (mlp.get("control") or {}).get("targets", {}).get(n, {}).get("score"),
                t.get("lambda"),
            ]
        )
    out += _table(
        ["target", "group", "latent", "latent 95%", "CLS", "untrained", "latent MLP", "untrained MLP", "ridge lambda"],
        rows,
    ) + [""]
    if st.get("lambda_at_edge"):
        out += [f"Ridge strength at the edge of the grid for: {', '.join(st['lambda_at_edge'])}.", ""]
    rows = []
    for n, t in ridge["latent"]["targets"].items():
        for col, s in t["columns"].items():
            rows.append(
                [
                    f"{n} {col}",
                    s,
                    ridge["cls"]["targets"][n]["columns"].get(col),
                    ridge["control"]["targets"][n]["columns"].get(col),
                    (mlp.get("latent") or {}).get("targets", {}).get(n, {}).get("columns", {}).get(col),
                ]
            )
    out += ["Per column:", ""] + _table(["column", "latent", "CLS", "untrained", "latent MLP"], rows) + [""]
    levels = summary.get("upper_levels") or {}
    if levels:
        rows = [
            [n, *[e["groups"].get(g) for g in tg.GROUPS]] for n, e in sorted(levels.items(), key=lambda kv: int(kv[0]))
        ]
        out += ["Every level on the frames the levels share (ridge):", ""] + _table(["level", *tg.GROUPS], rows) + [""]
    return out


def _stage_results(summary: dict) -> list[tuple[str, dict]]:
    """(label, results) of every tile stage in chain order: token stages, then frame-level features."""
    tokens, maps = summary.get("tokens") or {}, summary.get("maps") or {}
    order = [
        k
        for k in [
            "raw",
            *sorted((k for k in tokens if k.startswith("layer")), key=lambda k: int(k[5:])),
            "final",
            "control",
        ]
        if k in tokens
    ]
    return [(token_label(k), tokens[k]) for k in order] + [
        (FRAME_FEATURES[k], maps[k]) for k in FRAME_FEATURES if k in maps
    ]


def _tiles(summary: dict) -> list[str]:
    stages = _stage_results(summary)
    if not stages:
        return []
    shares = (summary.get("facts") or {}).get("tile_share") or [None] * tg.NUM_BLOCKS
    maps = summary.get("maps") or {}
    latent = (maps.get("latent") or {}).get("mlp balanced") or {}
    out = [
        "### Block type of a view tile, stage by stage",
        "",
        "Balanced MLP recall per block type (the player's tile left out; tokens are a stratified sample weighted "
        "back to "
        "the natural frequencies). The interval is the latent's.",
        "",
    ]
    rows = []
    min_n = summary.get("settings", {}).get("min_samples", 20)
    for c, name in enumerate(tg.BLOCK_NAMES):
        n = latent.get("n", [0] * tg.NUM_BLOCKS)[c] if latent else 0
        if n < min_n:
            continue
        cells = []
        for _, res in stages:
            r = res.get("mlp balanced") or {}
            cells.append(r["recall"][c] if r and r["n"][c] >= min_n else None)
        rows.append([name, shares[c], n, *cells, _interval(latent.get("recall_interval"), c)])
    out += _table(["block", "tile share", "held-out tiles", *[s for s, _ in stages], "latent 95%"], rows) + [""]
    rows = []
    for label, res in stages:
        for probe, r in res.items():
            mobs = [a for a in r.get("mob_auroc", []) if a is not None]
            rows.append([label, probe, r.get("accuracy"), r.get("balanced"), sum(mobs) / len(mobs) if mobs else None])
    out += ["Every probe (accuracy over all tiles, balanced accuracy, mean mob AUROC):", ""]
    out += _table(["stage", "probe", "accuracy", "balanced accuracy", "mob AUROC"], rows) + [""]
    for key in ("final", "cls", "latent"):
        res = (summary.get("tokens") or {}).get(key) if key == "final" else maps.get(key)
        r = (res or {}).get("mlp balanced")
        if not r:
            continue
        label = token_label(key) if key == "final" else FRAME_FEATURES[key]
        rows = []
        for c, name in enumerate(tg.BLOCK_NAMES):
            cells = [
                r["rings"][ring]["recall"][c] if r["rings"].get(ring) and r["rings"][ring]["n"][c] >= min_n else None
                for ring in _rings(r)
            ]
            if any(x is not None for x in cells):
                rows.append([name, *cells])
        out += (
            [f"Recall by distance from the player (tiles), {label}:", ""]
            + _table(["block", *[f"ring {x}" for x in _rings(r)]], rows)
            + [""]
        )
    rows = []
    for label, res in stages:
        r = res.get("mlp") or {}
        f = r.get("faced") or {}
        if f:
            rows.append([label] + [f["recall"][c] if f["n"][c] >= min_n else None for c in range(tg.NUM_BLOCKS)])
    cols = [c for c in range(tg.NUM_BLOCKS) if any(r[1 + c] is not None for r in rows)]
    if rows and cols:
        out += ["The faced tile (plain MLP recall), as the earlier tile probe:", ""]
        out += _table(
            ["stage", *[tg.BLOCK_NAMES[c] for c in cols]], [[r[0], *[r[1 + c] for c in cols]] for r in rows]
        ) + [""]
    return out


def _rings(r: dict) -> list:
    return sorted(r.get("rings", {}), key=int)


def _mobs(summary: dict) -> list[str]:
    stages = _stage_results(summary)
    if not stages:
        return []
    rows = []
    for label, res in stages:
        r = res.get("mlp balanced") or {}
        if r:
            rows.append([label, *r.get("mob_auroc", [None] * tg.NUM_MOBS)])
    out = [
        "### Mobs on a view tile, stage by stage",
        "",
        "Balanced MLP AUROC of a mob standing on the tile (0.5 = chance):",
        "",
    ]
    out += _table(["stage", *tg.MOB_NAMES], rows) + [""]
    r = ((summary.get("maps") or {}).get("latent") or {}).get("mlp balanced") or {}
    if r.get("mob_rings"):
        rows = [[f"ring {ring}", *r["mob_rings"][ring]] for ring in sorted(r["mob_rings"], key=int)]
        out += ["By distance from the player, the latent:", ""] + _table(["ring", *tg.MOB_NAMES], rows) + [""]
    return out


def _strip(summary: dict) -> list[str]:
    s = summary.get("strip") or {}
    if not s:
        return []
    out = [
        "### Inventory and vitals: the strip's pixels to the latent",
        "",
        f"Ridge R^2 on every 4th frame ({s.get('n_train')} / {s.get('n_val')} frames).",
        "",
    ]
    rows = []
    for target in ("inventory", "vitals"):
        for col in tg.TARGETS[target]["names"]:
            rows.append(
                [
                    f"{target} {col}",
                    *[((s.get(st) or {}).get(target) or {}).get("columns", {}).get(col) for st in STRIP_STAGES],
                ]
            )
    return out + _table(["column", *STRIP_STAGES.values()], rows) + [""]


def _causes(summary: dict) -> list[str]:
    c = summary.get("causes") or {}
    out = []
    if c.get("stages"):
        out += [
            "### Why: rarity or look-alike sprites",
            "",
            "Over block types (balanced MLP): rank correlation of the recall with the log tile share and with the "
            "pixel "
            "distance to the nearest other sprite, and a standardised fit of the recall on both. Over pairs of types: "
            "rank correlation of the confusion rate with the sprite distance (look-alikes confused more: negative) and "
            "with the share of the type confused for. Few types: read as a direction, not a test.",
            "",
        ]
        rows = []
        for stage, e in c["stages"].items():
            fit = e.get("fit") or {}
            coef = fit.get("coef") or {}
            rows.append(
                [
                    token_label(stage) if stage in ("raw", "final") else FRAME_FEATURES.get(stage, stage),
                    e["classes"],
                    e["recall_vs_log_share"],
                    e["recall_vs_nearest_sprite"],
                    coef.get("log_share"),
                    coef.get("nearest_sprite"),
                    fit.get("r2"),
                    e["pairs"],
                    e["confusion_vs_sprite_distance"],
                    e["confusion_vs_share_of_target"],
                ]
            )
        out += _table(
            [
                "stage",
                "types",
                "recall~share",
                "recall~sprite distance",
                "fit: share",
                "fit: sprite distance",
                "fit R^2",
                "pairs",
                "confusion~sprite distance",
                "confusion~target share",
            ],
            rows,
        ) + [""]
        rows = [
            [tg.BLOCK_NAMES[i], c["nearest_to"][i], c["nearest"][i]] for i in range(tg.NUM_BLOCKS) if c["nearest_to"][i]
        ]
        out += ["Nearest sprite of each block type (mean absolute pixel difference, 0 to 1):", ""]
        out += _table(["block", "nearest", "distance"], rows) + [""]
    sp = summary.get("spectrum") or {}
    if sp.get("features"):
        out += [
            "### Where in the latent's variance",
            "",
            f"Balanced linear map decode from the top-k principal components. Rare types (under 1% of tiles): "
            f"{', '.join(sp.get('rare') or []) or '-'}.",
            "",
        ]
        rows = []
        for feature, entries in sp["features"].items():
            for e in entries:
                rows.append(
                    [
                        FRAME_FEATURES.get(feature, feature),
                        e["k"],
                        e["variance"],
                        e["balanced"],
                        e["rare_recall"],
                        e["mob_auroc"],
                    ]
                )
        out += _table(["feature", "k", "variance kept", "balanced accuracy", "rare recall", "mob AUROC"], rows) + [""]
    return out


def _swaps(summary: dict) -> list[str]:
    s = summary.get("swaps") or {}
    if not s.get("targets"):
        return []
    out = [
        "### Sprite swaps",
        "",
        f"One tile of {s['frames']} bright held-out frames pasted with another sprite. Shift: squared distance "
        "moved, in "
        "units of the median step between consecutive frames (the planner's cost is the squared latent distance). "
        "Projector ratio: latent shift over CLS shift (below 1, the projector shrinks the change). Reads: share of "
        "swaps after which the map decoder reads the pasted block or mob at that tile (before: the same tile "
        "unedited). Null: a crop of the same block type from another frame.",
        "",
    ]
    rows = []
    for name, e in s["targets"].items():
        rows.append(
            [
                name,
                e["n"],
                e["latent_shift"],
                _interval(e.get("latent_interval")),
                e["cls_shift"],
                e["projector_ratio"],
                e.get("latent_reads_before"),
                e.get("latent_reads_after"),
                e.get("cls_reads_before"),
                e.get("cls_reads_after"),
            ]
        )
    out += _table(
        [
            "swap",
            "n",
            "latent shift",
            "latent 95%",
            "CLS shift",
            "projector ratio",
            "latent reads before",
            "latent reads after",
            "CLS reads before",
            "CLS reads after",
        ],
        rows,
    ) + [""]
    rings = sorted({r for e in s["targets"].values() for r in e.get("rings", {})}, key=int)
    if rings:
        rows = [[name, *[e.get("rings", {}).get(r) for r in rings]] for name, e in s["targets"].items()]
        out += (
            ["Latent shift by distance of the pasted tile from the player:", ""]
            + _table(["swap", *[f"ring {r}" for r in rings]], rows)
            + [""]
        )
    return out


def _distance(summary: dict) -> list[str]:
    d = summary.get("distance") or {}
    out = []
    for gap, e in sorted(d.items(), key=lambda kv: int(kv[0])):
        out += [
            f"### What the latent distance sees, frames {gap} apart",
            "",
            f"Least squares of the squared latent distance (in median steps) on what changed; {e['pairs']} pairs, R^2 "
            f"{_f(e['r2'])}, mean distance {_f(e['mean_distance'])}, intercept {_f(e['intercept'])}. Effect: the "
            "coefficient times the typical nonzero change. Correlated pairs and collinear changes: descriptive.",
            "",
        ]
        rows = sorted(
            ([n, v["share_changed"], v["typical_change"], v["coef"], v["effect"]] for n, v in e["features"].items()),
            key=lambda r: -abs(r[4]),
        )
        out += _table(["change", "share of pairs", "typical change", "coefficient", "effect"], rows) + [""]
    return out


def _loss(summary: dict) -> list[str]:
    a = summary.get("loss") or {}
    if not a:
        return []
    out = [
        "### Which transitions carry the prediction loss",
        "",
        f"One-step prediction error (teacher-forced, the training's history) of {a['transitions']} held-out "
        "transitions; "
        f"mean {_f(a['mean_loss'], 4)} (copying the last latent: {_f(a['mean_copy'], 4)}).",
        "",
    ]
    rows = [
        [n, e["transitions"], e["share"], e["loss_share"], e["relative_loss"], e["pred_over_copy"]]
        for n, e in a["rows"].items()
        if e
    ]
    rows += [
        [f"{n} (overlaps)", e["transitions"], e["share"], e["loss_share"], e["relative_loss"], e["pred_over_copy"]]
        for n, e in (a.get("overlapping") or {}).items()
        if e
    ]
    return out + _table(["transition", "count", "share", "loss share", "loss / mean", "pred / copy"], rows) + [""]


def _change(summary: dict) -> list[str]:
    out = []
    for hist, ch in sorted((summary.get("change") or {}).items(), key=lambda kv: int(kv[0])):
        by_k = ch.get("by_k") or {}
        if not by_k:
            continue
        ks = sorted(by_k, key=int)
        names = list(by_k[ks[0]])
        out += [
            f"### Change test, history {hist}",
            "",
            f"Probes fitted on encoded latents, applied to open-loop predictions k steps ahead ({ch.get('windows')} "
            "windows, true actions). Where the entity changed: accuracy of the predicted / copied / encoded readout "
            "(copying is wrong there by construction, up to probe noise); where it did not: predicted / copied "
            "(invented changes). Recovered: (predicted - copy) / (encoded - copy) on changed windows.",
            "",
        ]
        rows = [[n, *[_cell(by_k[k][n], ("pred_changed", "copy_changed", "enc_changed")) for k in ks]] for n in names]
        out += ["Changed windows:", ""] + _table(["target", *[f"k={k}" for k in ks]], rows) + [""]
        rows = [[n, *[_cell(by_k[k][n], ("pred_same", "copy_same")) for k in ks]] for n in names]
        out += ["Unchanged windows:", ""] + _table(["target", *[f"k={k}" for k in ks]], rows) + [""]
        rows = [
            [n, *[by_k[k][n].get("recovered") for k in ks], *[by_k[k][n].get("changed") for k in ks[-1:]]]
            for n in names
        ]
        out += (
            ["Recovered (and changed windows at the last k):", ""]
            + _table(["target", *[f"k={k}" for k in ks], "changed"], rows)
            + [""]
        )
        k = max((x for x in ks if int(x) <= PREVIEW_K), key=int, default=ks[0])
        rows = []
        for n in names:
            for col, e in by_k[k][n]["columns"].items():
                if e["changed"]:
                    rows.append(
                        [
                            f"{n} {col}",
                            e["changed"],
                            e["pred_changed"],
                            e["copy_changed"],
                            e["enc_changed"],
                            e["recovered"],
                            e.get("shown"),
                            e["pred_same"],
                            e["copy_same"],
                        ]
                    )
        out += [f"Per column at k = {k}:", ""]
        out += _table(
            ["column", "changed", "pred", "copy", "encoded", "recovered", "shown", "pred (same)", "copy (same)"], rows
        ) + [""]
    return out


def _cell(e: dict, keys) -> str:
    return " / ".join(_f(e.get(k)) for k in keys)


def _events(summary: dict) -> list[str]:
    ev = summary.get("events") or {}
    if not ev:
        return []
    out = [
        "### Events, one step",
        "",
        "Signed change of the latent's probe for the column the event changes, from the latent before the step: to the "
        "encoded next latent and to the predicted one, after successful and failed presses. Shown: the predicted "
        "success-minus-failure gap over the encoded one (1 as a real frame, 0 no change predicted). Tile: where the "
        "faced tile changes, the share of predicted / encoded / copied latents whose map decoder reads the new block.",
        "",
    ]
    rows = [
        [
            n,
            e["succeeded"],
            e["failed"],
            e["encoded"],
            e.get("encoded_failed"),
            e["pred"],
            e["pred_failed"],
            e["shown"],
            e.get("tile_changes"),
            _cell(e, ("tile_pred", "tile_enc", "tile_copy")),
        ]
        for n, e in ev.items()
    ]
    return (
        out
        + _table(
            [
                "event",
                "successful",
                "failed",
                "encoded",
                "encoded (failed)",
                "predicted",
                "predicted (failed)",
                "shown",
                "tile changes",
                "tile pred / enc / copy",
            ],
            rows,
        )
        + [""]
    )


def _mob_dynamics(summary: dict) -> list[str]:
    m = summary.get("mobs") or {}
    if not m:
        return []
    out = [
        "### Mobs, one step",
        "",
        "AUROC of the tiles where the mob stands at t + 1 against the empty tiles, from the map decoder of the "
        "predicted "
        "latent / the encoded next latent / the copied last latent. Continuing: within one tile of the same mob type "
        "at t.",
        "",
    ]
    rows = [
        [
            n,
            e["continuing"],
            _cell(e, ("continuing_pred", "continuing_enc", "continuing_copy")),
            e["new"],
            _cell(e, ("new_pred", "new_enc", "new_copy")),
        ]
        for n, e in m.items()
    ]
    return out + _table(["mob", "continuing", "pred / enc / copy", "new", "pred / enc / copy"], rows) + [""]


def _monotonicity(summary: dict) -> list[str]:
    mo = summary.get("monotonicity") or {}
    if not mo.get("offsets"):
        return []
    out = [
        "### Distance to a goal along the expert's path",
        "",
        f"Goal: the latent of frame t + K ({'expert clips only' if mo.get('expert_only') else 'all clips'}). rho: rank "
        "correlation of the squared distance with the steps left (1 = falls all the way); falls: share of steps where "
        "it falls; final/start: the predicted end state's distance over the start distance (predictions with the "
        "expert's actions, planning history).",
        "",
    ]
    rows = []
    for K, e in sorted(mo["offsets"].items(), key=lambda kv: int(kv[0])):
        for part in ("all", "inventory goal", "other goal"):
            p = e.get(part)
            if not p:
                continue
            enc, pred = p.get("encoded") or {}, p.get("predicted") or {}
            rows.append(
                [
                    K,
                    part,
                    enc.get("rho_median"),
                    enc.get("rho_above_half"),
                    enc.get("falls"),
                    pred.get("rho_median"),
                    pred.get("rho_above_half"),
                    pred.get("falls"),
                    pred.get("final_over_start"),
                ]
            )
    return (
        out
        + _table(
            [
                "K",
                "goal",
                "encoded rho",
                "encoded rho > 0.5",
                "encoded falls",
                "predicted rho",
                "predicted rho > 0.5",
                "predicted falls",
                "predicted final/start",
            ],
            rows,
        )
        + [""]
    )


def _idm(summary: dict) -> list[str]:
    idm = summary.get("idm") or {}
    if not idm:
        return []
    awake, asleep = idm["awake"], idm["asleep"]
    out = [
        "### Inverse dynamics per action",
        "",
        f"Awake steps: {awake['steps']}, accuracy {_f(awake['accuracy'])} (majority {_f(awake.get('majority'))}); "
        "asleep "
        f"steps (the game ignores the action): {asleep['steps']}, accuracy {_f(asleep['accuracy'])}.",
        "",
    ]
    rows = []
    for name, e in idm["actions"].items():
        ok, bad = e.get("succeeded"), e.get("failed")
        rows.append(
            [name, e["steps"], e["accuracy"], *[f"{_f(x['accuracy'])} ({x['steps']})" if x else "" for x in (ok, bad)]]
        )
    return out + _table(["action", "steps", "accuracy", "successful", "failed"], rows) + [""]
