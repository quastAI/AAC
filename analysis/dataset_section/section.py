"""The LaTeX of the whole dataset subsection, ready to paste into the appendix (ICLR: text width 5.5 in).

The tables are inlined from the files that `plot` and `splits` wrote, the figures are `\\includegraphics` of the PDFs of the
same folder (upload them to the Overleaf project; set `\\graphicspath` if they are in a folder). Needs `graphicx`,
`booktabs` and `float` (the [H] placement).
"""

from __future__ import annotations

from pathlib import Path

NAME = "dataset_section.tex"

INTRO = r"""\subsection{Dataset}
\label{app:dataset}

The data are split into train, validation and test. Every episode uses its
own world seed, and the seed ranges of the splits are disjoint (train from 0,
validation from 1M, test from 2M), so validation and test worlds are never
seen during training.

\emph{Train} (Table~\ref{tab:data-train}) has 2.0M frames (3\,529 episodes)
from a mix of the final PQN-RNN expert (70\% of the frames) and four earlier
snapshots of its training run (7.5\% each). Expert episodes are kept only if
they reach the diamond, and they end 500 steps after it, or earlier on death.
Snapshot episodes need no diamond and run until the agent dies.
\emph{Validation} (Table~\ref{tab:data-val}) has 151K frames (262 episodes)
and follows the same mix. \emph{Test} (Table~\ref{tab:data-test}) has 151K
frames (185 episodes) from the final expert only. It is the task set, used
only for planning and evaluation, never for training.
"""  # the author's text, verbatim

FIGURES = {  # file name -> (label, caption)
    "fig_length_train": ("fig:data-length-train", "Episode length in the training split (bins of 50 frames). The test figure uses the same axes."),
    "fig_length_test": ("fig:data-length-test", "Episode length in the test split (bins of 50 frames), on the same axes as the training split."),
    "fig_time_to_diamond": ("fig:data-time-to-diamond", "Steps from the reset to the first diamond in the test split (bins of 25 steps). Dashed lines: 25th, 50th and 75th percentile."),
    "fig_achievements_train": ("fig:data-achievements-train", "Distinct achievements (of 22) unlocked per episode in the training split."),
    "fig_achievements_test": ("fig:data-achievements-test", "Distinct achievements (of 22) unlocked per episode in the test split, on the same axes as the training split."),
    "fig_achievement_ladder": ("fig:data-ladder", "Frame of the first unlock of each achievement, over the episodes that unlock it: median and 10th to 90th percentile, sorted by the training median."),
    "fig_change": ("fig:data-change", "Share of frame pairs $(t, t+k)$ of one episode in which a state component has a different value, per lag $k$. The view rows are pooled over the 7$\\times$9 tiles; the world-frame rows compare terrain and mobs in world coordinates on the tiles that both views show (lags up to 5). The second line of each tick is the percentage of episodes long enough to contain a pair at that lag. Daylight counts as changed above a difference of 0.05."),
    "fig_clips": ("fig:data-clips", "Training clips per stride (all overlapping clips of the training split, with death padding as in training) and the share of episodes long enough to contain one, for clips of 2, 6 and 10 frames (context plus rollout)."),
    "fig_actions": ("fig:data-actions", "Steps per action in the training split."),
}
# the order of the section: (kind, name); a table is a file written by `plot` or `splits`
ORDER = [
    ("table", "table_train"), ("table", "table_val"), ("table", "table_test"),
    ("figure", "fig_length_train"), ("figure", "fig_length_test"), ("table", "table_length"), ("figure", "fig_time_to_diamond"),
    ("table", "table_spectral"),
    ("figure", "fig_achievements_train"), ("figure", "fig_achievements_test"), ("figure", "fig_achievement_ladder"), ("table", "table_achievements"),
    ("figure", "fig_change"), ("figure", "fig_clips"), ("figure", "fig_actions"), ("table", "table_chunks"),
]  # fmt: skip


def figure_block(name: str) -> str:
    label, caption = FIGURES[name]
    return "\n".join(["\\begin{figure}[H]", "\\centering", f"\\includegraphics[width=\\linewidth]{{{name}.pdf}}", f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}", "\\end{figure}"]) + "\n"  # fmt: skip


def write_section(out: Path) -> Path | None:
    """`out / dataset_section.tex` from the tables and figures of `out`; tables that were not written are left out."""
    parts = [INTRO]
    for kind, name in ORDER:
        if kind == "figure":
            parts.append(figure_block(name))
        elif (out / f"{name}.tex").exists():
            parts.append((out / f"{name}.tex").read_text())
    path = out / NAME
    path.write_text("\n".join(parts))
    return path
