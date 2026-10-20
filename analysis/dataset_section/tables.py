"""LaTeX tables and the change-matrix csv of the dataset section, from `dataset_numbers.json` (a dict, no data files)."""

from __future__ import annotations

import csv
from pathlib import Path

from .data import ACHIEVEMENTS

TABCOLSEP = "3.5pt"  # tighter than the default 6pt: the 9 and 10 column tables must fit the 5.5 in text width of ICLR


def thousands(n) -> str:
    return f"{int(round(n)):,}".replace(",", "\\,")


def tex_table(label: str, caption: str, columns: str, header: str, rows: list[str], note: str | None = None) -> str:
    lines = ["\\begin{table}[H]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:{label}}}", "\\small", f"\\setlength{{\\tabcolsep}}{{{TABCOLSEP}}}",
             f"\\begin{{tabular}}{{{columns}}}", "\\toprule", header + " \\\\", "\\midrule"]  # fmt: skip
    lines += [r if r.startswith("\\midrule") else r + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["", "\\vspace{4pt}", "\\begin{minipage}{\\linewidth}", f"\\footnotesize\\raggedright {note}", "\\end{minipage}"]
    return "\n".join(lines + ["\\end{table}"]) + "\n"


def length_table(rec: dict) -> str:
    train, test = rec["splits"]["train"], rec["splits"]["test"]
    cols = ["n", "mean", "min", "p10", "p25", "p50", "p75", "p90", "max"]
    row = lambda name, q: " & ".join([name, thousands(q["n"])] + [thousands(q[c]) for c in cols[1:]])  # noqa: E731
    rows = [row("Train, episode length", train["length"]["quantiles"]), row("Test, episode length", test["length"]["quantiles"])]
    ttd = test["time_to_diamond"]
    if ttd["episodes"]:
        rows += ["\\midrule", row("Test, time to diamond", ttd["quantiles"])]
    return tex_table(
        "data-length", "Episode length and time to the diamond, in frames.", "lrrrrrrrrr",
        "& $n$ & Mean & Min & p10 & p25 & Median & p75 & p90 & Max", rows,
        "Time to diamond: steps from the reset to the first frame with the diamond, over the episodes that reach it. Episodes that reach the "
        "diamond end 500 steps after it (or earlier on death).",
    )  # fmt: skip


def spectral_table(rec: dict) -> str | None:
    s = rec["splits"]["test"].get("spectral")
    if not s:
        return None
    w = s["primary_window"]
    rows = []
    for name, e in sorted(s["entities"].items(), key=lambda kv: kv[1]["period"]):
        rows.append(f"{name.replace('_', ' ')} & {e['probe_group']} & {e['period']:.1f} & {'yes' if e['limited'] else 'no'}")
    gap = s["gap"]["all"]
    ci = gap.get("ci") or {}
    note = (f"Centroid period in frames (1 / spectral centroid, Welch window {w}, hop {w // 2}, Hann; each dimension standardised per episode), {s['episodes_used']} test "
            f"episodes. Limited: more than half of the power in the two lowest bins, so the period is a lower bound. Frequency gap (largest over smallest "
            f"period): {gap['ratio']:.3g}" + (" or more" if gap["lower_bound"] else "") +
            (f", bootstrap p5 to p95 {ci['p5']:.3g} to {ci['p95']:.3g}" if ci else "") +
            "; at windows " + ", ".join(f"{k}: {v['gap']['all']['ratio']:.3g}" for k, v in sorted(s["sensitivity"].items(), key=lambda kv: int(kv[0]))) + ".")  # fmt: skip
    return tex_table("data-spectral", "Spectral centroid period of the probe entities of the test split.", "llrl",
                     "Entity & Group & Period & Limited", rows, note)  # fmt: skip


def ladder_order(rec: dict) -> list[str]:
    ladder = rec["splits"]["train"]["achievements"]["ladder"]
    return sorted(ACHIEVEMENTS, key=lambda n: (ladder[n].get("p50") is None, ladder[n].get("p50", 0.0)))


def achievement_table(rec: dict) -> str:
    train, test = (rec["splits"][s]["achievements"]["ladder"] for s in ("train", "test"))

    def cells(e):
        if "p50" not in e:
            return ["---", "---"]
        return [f"{100 * e['share']:.1f}\\%", f"{e['p50']:.0f} [{e['p10']:.0f}, {e['p90']:.0f}]"]

    rows = [" & ".join([n.replace("_", " ")] + cells(train[n]) + cells(test[n])) for n in ladder_order(rec)]
    return tex_table(
        "data-achievements", "Achievements: share of episodes that unlock each one and the frame of its first unlock.", "lrrrr",
        "& \\multicolumn{2}{c}{Train} & \\multicolumn{2}{c}{Test} \\\\\n Achievement & Share & First unlock & Share & First unlock", rows,
        "First unlock: median frame, with p10 and p90 in brackets, over the episodes that unlock it. Sorted by the train median.",
    )  # fmt: skip


def chunk_table(rec: dict) -> str:
    train, test = (rec["splits"][s].get("chunks", {}) for s in ("train", "test"))
    rows = []
    for k in sorted(train, key=int):
        cells = [k]
        for c in (train, test):
            e = c.get(k)
            info = e and e.get("information_with_motion")
            if not e:
                cells += ["---"] * 4
                continue
            cells += [thousands(e["distinct_chunks"]), f"{e['chunk_entropy_bits']:.1f}"]
            cells += [f"{info['mi_bits']:.2f}", f"{100 * info['coverage']:.0f}\\%"] if info else ["---", "---"]
        rows.append(" & ".join(cells))
    return tex_table(
        "data-chunks", "Action chunks of $k$ steps: distinct chunks, chunk entropy and information about the effect.", "lrrrrrrrr",
        "& \\multicolumn{4}{c}{Train} & \\multicolumn{4}{c}{Test} \\\\\n $k$ & Distinct & $H$ & $I$ & Cov. & Distinct & $H$ & $I$ & Cov.", rows,
        "Windows of $k$ steps without overlap (at most 500\\,000 per $k$). Distinct: chunks seen. $H$: entropy of the chunk (bits). $I$: mutual information of the chunk "
        "with the effect (bits; change of inventory, achievements, vitals and position over the window; Miller--Madow corrected). Cov.: share of windows whose chunk "
        "was seen at least 20 times, the only windows $I$ is computed on; where it is low, most chunks occur once and $I$ is not reliable. "
        "Reference scales for a codebook, not settings.",
    )  # fmt: skip


def change_csv(rec: dict, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "group", "component", "half_lag"] + [f"lag_{k}" for k in rec["lags"]])
        for label in ("train", "val", "test"):
            if label not in rec["splits"]:
                continue
            for r in rec["splits"][label]["change"]["rows"]:
                w.writerow([label, r["group"], r["name"], r["half_lag"]] + ["" if s is None else f"{s:.6f}" for s in r["share"]])
    return path
