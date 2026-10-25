"""One verdict per entity: walk its chain until a link fails.

Encoder links (a score per stage, the balanced MLP's recall for block types, 2*AUROC-1 for mobs, ridge R^2 for the
inventory and vitals): the first stage that keeps less than KEEP of the best stage before it is where the entity is
lost; a ceiling (the pixels) under MIN_SCORE means the target itself is not readable. Past the encoder: the sprite swap
(does the latent's map decoder read a pasted block or mob), the one-step event or mob motion, the rollout's change test.
The thresholds are heuristics, set before real numbers; every score stays in the table next to the verdict.
"""

from __future__ import annotations

from . import targets as tg

KEEP = 0.8  # a stage keeps an entity if it reaches this share of the best stage before it
FLOOR = 0.5  # a swap read (after minus before), an event's `shown` or a rollout's `recovered` below this: lost
MIN_SCORE = 0.2  # a pixel ceiling below this: the target is not readable from the frame
VERDICT_K = 8  # the rollout step of the change test in the verdicts


def chain(stages: list[tuple[str, float | None]]) -> str:
    """'kept', 'lost at <stage>' or 'not readable from <stage>' along the encoder stages (None skips a stage)."""
    best = None
    for name, s in stages:
        if s is None:
            continue
        if best is None:
            if s < MIN_SCORE:
                return f"not readable from {name}"
            best = s
            continue
        if s < KEEP * best:
            return f"lost at {name}"
        best = max(best, s)
    return "kept" if best is not None else "no data"


def _later(verdict: str, links: list[tuple[str, float | None]]) -> str:
    """After a kept encoder chain: the first later link under FLOOR."""
    if verdict != "kept":
        return verdict
    for name, s in links:
        if s is not None and s < FLOOR:
            return f"lost at {name}"
    return verdict


def _rec(entry) -> float | None:
    return None if entry is None else entry.get("recovered")


def _change(change: dict, target: str, column: str | None = None):
    """The change test entry of a target (or one of its columns) at VERDICT_K (or the largest k scored)."""
    by_k = (change or {}).get("by_k") or {}
    if not by_k:
        return None
    k = max((int(k) for k in by_k if int(k) <= VERDICT_K), default=None)
    if k is None:
        return None
    entry = (by_k.get(k) or by_k.get(str(k)) or {}).get(target)
    if entry is None or column is None:
        return entry
    return entry["columns"].get(column)


def _swap_read(swap: dict) -> float | None:
    """Share of swaps after which the latent's map decoder newly reads the pasted block or mob (after minus before)."""
    after, before = swap.get("latent_reads_after"), swap.get("latent_reads_before")
    return None if after is None or before is None else after - before


def _recovered_auroc(pred, copy, enc) -> float | None:
    if None in (pred, copy, enc) or enc - copy < 0.05:
        return None
    return (pred - copy) / (enc - copy)


def entity_verdicts(summary: dict, min_n: int) -> dict:
    """Rows per entity type: block types, mobs, inventory and vitals, the other state."""
    tokens, maps = summary.get("tokens") or {}, summary.get("maps") or {}
    swaps = (summary.get("swaps") or {}).get("targets") or {}
    events, mobs_dyn = summary.get("events") or {}, summary.get("mobs") or {}
    change = summary.get("change_plan") or {}
    layers = [k for k in tokens if k.startswith("layer")]
    token_order = [k for k in ["raw", *layers, "final"] if k in tokens]
    out = {"blocks": [], "mobs": [], "items": [], "other": [], "k": VERDICT_K}

    def recall(res, c):
        r = (res or {}).get("mlp balanced")
        if not r or r["n"][c] < min_n:
            return None
        return r["recall"][c]

    collect_of = {tg.BLOCK_NAMES.index(b): v[0] for b, v in tg.COLLECT.items()}
    for c, name in enumerate(tg.BLOCK_NAMES):
        stages = [(s, recall(tokens.get(s), c)) for s in token_order] + [
            ("cls", recall(maps.get("cls"), c)),
            ("latent", recall(maps.get("latent"), c)),
        ]
        if all(s is None for _, s in stages):
            continue
        swap = swaps.get(f"block:{name}") or {}
        event = events.get(collect_of.get(c)) if c in collect_of else None
        links = [
            ("swap", _swap_read(swap)),
            ("one step", (event or {}).get("shown")),
            ("rollout", _rec(_change(change, "view_blocks", name))),
        ]
        linear = (maps.get("latent") or {}).get("linear balanced") or {}
        row = {
            "entity": name,
            "stages": stages,
            "verdict": _later(chain(stages), links),
            "links": links,
            "control": recall(tokens.get("control"), c),
            "latent_linear": None if not linear or linear["n"][c] < min_n else linear["recall"][c],
            "swap_shift": swap.get("latent_shift"),
        }
        out["blocks"].append(row)

    def mob_score(res, k, key="mob_auroc"):
        r = (res or {}).get("mlp balanced")
        a = None if not r else r[key][k]
        return None if a is None else 2 * a - 1

    for k, name in enumerate(tg.MOB_NAMES):
        stages = [(s, mob_score(tokens.get(s), k)) for s in token_order] + [
            ("cls", mob_score(maps.get("cls"), k)),
            ("latent", mob_score(maps.get("latent"), k)),
        ]
        if all(s is None for _, s in stages):
            continue
        swap = swaps.get(f"mob:{name}") or {}
        dyn = mobs_dyn.get(name) or {}
        links = [
            ("swap", _swap_read(swap)),
            (
                "one step",
                _recovered_auroc(dyn.get("continuing_pred"), dyn.get("continuing_copy"), dyn.get("continuing_enc")),
            ),
            ("rollout", _rec(_change(change, "mobs_present", name))),
        ]
        out["mobs"].append(
            {
                "entity": name,
                "stages": stages,
                "verdict": _later(chain(stages), links),
                "links": links,
                "control": mob_score(tokens.get("control"), k),
                "swap_shift": swap.get("latent_shift"),
            }
        )

    strip = summary.get("strip") or {}
    mlp_cols = {}
    for target in ("inventory", "vitals"):
        mlp_cols.update(
            (((summary.get("state") or {}).get("mlp") or {}).get("latent") or {})
            .get("targets", {})
            .get(target, {})
            .get("columns", {})
        )
    event_of = {}
    for name, (group, col, _) in {**tg.PLACE_MAKE, **{v[0]: v[1:] for v in tg.COLLECT.values()}}.items():
        e = events.get(name)
        if e and e.get("shown") is not None:
            key = (group, col)
            if key not in event_of or e["succeeded"] > event_of[key][1]["succeeded"]:
                event_of[key] = (name, e)
    for target in ("inventory", "vitals"):
        for col, name in enumerate(tg.TARGETS[target]["names"]):
            stages = [
                (s, ((strip.get(s) or {}).get(target) or {}).get("columns", {}).get(name))
                for s in ("raw", "tokens", "cls", "latent")
            ]
            event = event_of.get((target, col))
            links = [
                ("one step", None if event is None else event[1]["shown"]),
                ("rollout", _rec(_change(change, target, name))),
            ]
            out["items"].append(
                {
                    "entity": f"{target} {name}",
                    "stages": stages,
                    "verdict": _later(chain(stages), links),
                    "links": links,
                    "latent_mlp": mlp_cols.get(name),
                    "event": None if event is None else event[0],
                }
            )

    ridge = (summary.get("state") or {}).get("ridge") or {}
    for target in ("facing", "facing_block", "sleeping", "daylight", "achievements"):
        stages = [
            (f, ((ridge.get(f) or {}).get("targets") or {}).get(target, {}).get("score")) for f in ("cls", "latent")
        ]
        links = [("rollout", _rec(_change(change, target)))]
        out["other"].append(
            {
                "entity": target,
                "stages": stages,
                "verdict": _later(chain(stages), links),
                "links": links,
                "control": ((ridge.get("control") or {}).get("targets") or {}).get(target, {}).get("score"),
            }
        )
    return out
