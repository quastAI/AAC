"""Side-by-side tables of runs: planning against a baseline, compute scaling, training, probing.

    python compare_runs.py \
        --planning random=eval/random flat=eval/lewm hier=eval/hjepa_l2 --baseline random \
        --scaling s1=eval/hjepa_x0.25 s2=eval/hjepa_x1 s3=eval/hjepa_x4 \
        --training lewm=ckpts/lewm hier=ckpts/hjepa_l2 \
        --probing lewm=ckpts/lewm hier=ckpts/hjepa_l2

Reads what the runs wrote: `planning_health.json` (planning), `health/summary.json`
(training) and `health/probe_summary.json` (probing). Every section is optional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _f(x, digits=3):
    if x is None:
        return "-"
    return f"{x:.{digits}g}" if isinstance(x, float) else str(x)


def _runs(items):
    out = {}
    for item in items or []:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected label=path, got {item!r}")
        out[label] = Path(path)
    return out


def planning_table(runs: dict[str, Path], baseline: str | None) -> list[str]:
    data = {label: _load(p / "planning_health.json") for label, p in runs.items()}
    data = {k: v["summary"] for k, v in data.items() if v}
    if not data:
        return []
    base = data.get(baseline) if baseline else None
    out = ["## Planning", "", "| run | tasks | success | progress | progress over baseline | TFLOPs per task | critical | warnings |",
           "|---|---|---|---|---|---|---|---|"]
    findings = {label: (_load(runs[label] / "planning_health.json") or {}).get("findings", []) for label in data}
    for label, s in data.items():
        gain = None if base is None or label == baseline else s["progress_mean"] - base["progress_mean"]
        f = findings[label]
        out.append(
            f"| {label} | {s['num_tasks']} | {_f(s['success_rate'])} | {_f(s['progress_mean'])} | {_f(gain)} "
            f"| {_f((s.get('compute') or {}).get('tflops_per_task'))} "
            f"| {sum(x['severity'] == 'critical' for x in f)} | {sum(x['severity'] == 'warn' for x in f)} |"
        )
    groups = sorted({g for s in data.values() for g in s.get("by_group", {})})
    if groups:
        out += ["", "Success by task group" + (" (baseline in brackets)" if base else "") + ":", "",
                "| run | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for label, s in data.items():
            cells = []
            for g in groups:
                v = s.get("by_group", {}).get(g)
                ref = (base or {}).get("by_group", {}).get(g)
                cells.append("-" if v is None else _f(v["success"], 2) + (f" ({_f(ref['success'], 2)})" if ref and label != baseline else ""))
            out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out + [""]


def scaling_table(runs: dict[str, Path]) -> list[str]:
    rows = []
    for label, p in runs.items():
        doc = _load(p / "planning_health.json")
        if doc:
            s = doc["summary"]
            rows.append((label, (s.get("compute") or {}).get("tflops_per_task"), s, doc["findings"]))
    if not rows:
        return []
    rows.sort(key=lambda r: (r[1] is None, r[1]))
    out = ["## Planner compute scaling", "", "| run | TFLOPs per task | success | progress | chosen/random plan error | optimism | rank correlation |",
           "|---|---|---|---|---|---|---|"]
    for label, tf, s, _ in rows:
        pl = s.get("plans") or {}
        out.append(f"| {label} | {_f(tf)} | {_f(s['success_rate'])} | {_f(s['progress_mean'])} "
                   f"| {_f(pl.get('err_ratio_vs_random'))} | {_f(pl.get('optimism_chosen'))} | {_f(pl.get('rank_corr'))} |")
    progress = [r[2]["progress_mean"] for r in rows if r[1] is not None]
    if len(progress) >= 3 and all(b < a - 1e-9 for a, b in zip(progress, progress[1:])):
        out += ["", "**Progress falls as the planner gets more compute**: the signature of a planner that exploits the model."]
    elif len(progress) >= 2 and progress[-1] < progress[0] - 0.05:
        out += ["", "More compute gave less progress than the smallest budget: check the exploitation findings."]
    return out + [""]


def training_table(runs: dict[str, Path]) -> list[str]:
    docs = {label: _load(p / "health" / "summary.json") for label, p in runs.items()}
    docs = {k: v for k, v in docs.items() if v}
    if not docs:
        return []
    out = ["## Training", "", "| run | level | erank / dim | pred / copy | action lift | useful horizon | VQ perplexity | IDM accuracy | critical | warnings |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for label, doc in docs.items():
        crit = sum(f["severity"] == "critical" for f in doc["findings"])
        warn = sum(f["severity"] == "warn" for f in doc["findings"])
        plan_history = str(doc["summary"].get("plan_history", 1))
        for level, m in sorted(doc["summary"]["levels"].items(), key=lambda kv: int(kv[0])):
            rep, dyn, vq, idm = (m.get(k) or {} for k in ("rep", "dyn", "vq", "idm"))
            drift = (m.get("drift") or {}).get(plan_history) or {}
            out.append(f"| {label} | {level} | {_f(rep.get('erank'))} / {rep.get('dim')} | {_f(dyn.get('pred_over_copy'))} "
                       f"| {_f(dyn.get('action_lift'))} | {_f(drift.get('useful_horizon'))} | {_f(vq.get('perplexity'))} "
                       f"| {_f(idm.get('accuracy'))} | {crit} | {warn} |")
    return out + [""]


def probing_table(runs: dict[str, Path]) -> list[str]:
    docs = {label: _load(p / "health" / "probe_summary.json") for label, p in runs.items()}
    docs = {k: v for k, v in docs.items() if v}
    if not docs:
        return []
    out = ["## Probing (score of fast / medium / slow state; untrained encoder in brackets)", "",
           "| run | level | fast | medium | slow |", "|---|---|---|---|---|"]
    for label, doc in docs.items():
        for level, e in sorted(doc["summary"]["levels"].items(), key=lambda kv: int(kv[0])):
            cells = []
            for g in ("fast", "medium", "slow"):
                c = (e.get("control_groups") or {}).get(g)
                cells.append(_f(e["groups"].get(g), 2) + (f" ({_f(c, 2)})" if c is not None else ""))
            out.append(f"| {label} | {level} | " + " | ".join(cells) + " |")
    return out + [""]


def build(args) -> str:
    out = ["# Run comparison", ""]
    out += planning_table(_runs(args.planning), args.baseline)
    out += scaling_table(_runs(args.scaling))
    out += training_table(_runs(args.training))
    out += probing_table(_runs(args.probing))
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("planning", "scaling", "training", "probing"):
        parser.add_argument(f"--{name}", nargs="*", help="label=path of result or run directories")
    parser.add_argument("--baseline", help="label of the random-policy planning run")
    parser.add_argument("--out", help="write the markdown here")
    args = parser.parse_args()
    text = build(args)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
