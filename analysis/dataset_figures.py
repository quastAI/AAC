"""The dataset figures and tables of the appendix that the model design depends on, from the episode files.

The code is the package `dataset_section` (see its modules); this is the command line.

Two steps, so that a figure can be restyled without the data and every number has one source:

* `compute` reads the collector's HDF5 files (numpy and h5py, no pixels; the view of the `state` column for the change
  matrix) and writes `dataset_numbers.json`: every number of every figure and table, the files' hashes, the collector's
  `runs` records and the git commit (as `dataset_tables.json` does);
* `plot` reads only that JSON and writes the figures (`fig_*.pdf`, matplotlib) and tables (`table_*.tex`, `*.csv`).

Each split is taken as one set, the way the model sees it: clips are drawn at random from all of train. Test is the
final expert only. Validation is computed like the others and only compared with train (`val_vs_train`: distances, no
figures).

* length: frames per episode (`ep_len`, L + 1 frames for L actions), per split, as a histogram and quantiles; the time
  to the diamond (first frame with `collect_diamond`) of the episodes that reach it, per split (shown for test);
* spectral: the entity frequency gap of the paper (`dataset_section.spectral`, the 7 probe entities, Welch window
  256, window sensitivity, episode bootstrap) on the test split. The numbers are not strides: the slowest entities are
  limited by the window (the gap is a lower bound then) and the centroid also rises with fast flicker;
* achievements: achievements per episode (final frame), and per achievement the share of episodes that unlock it and the
  quantiles of the frame of its first unlock (among the episodes that do), the ladder of the expert's structure;
* change matrix: per component of the state (and the view, pooled) and per lag k, the share of frame pairs (t, t + k)
  of one episode in which it differs (net change, not "anything changed in between"). Daylight is a float and counts as
  changed above `DAYLIGHT_TOL` (a heuristic; it is also periodic with a day of 300 steps, so the share falls back near
  lag 300). The view is egocentric, so walking changes every tile: besides the egocentric rows the terrain and the mobs
  are compared in world coordinates on the tiles that both views show (needs the `position` column; only for lags up
  to `WORLD_LAG_MAX`, where two views always overlap). `half_lag` is the first lag of the grid with a share of 50%;
* clip coverage: clips (all overlapping, not independent ones) per stride, for context 1 and 5 and rollout 1 and 5, as
  in `dataset_section.episodes.clip_counts` (a clip is `context + rollout` frames, death padding as in training);
* actions: steps per action, and per chunk length k the distinct chunks, the entropy of the chunk and its mutual
  information with the effect (`dataset_section.actions`), reference scales for a codebook, not settings.

    python dataset_figures.py compute --data expert_train.h5 --label train --data expert_val.h5 --label val \
        --data expert_test.h5 --label test --out ../reports/dataset/ [--sha256]
    python dataset_figures.py plot --out ../reports/dataset/

Pairs, lags and strides never cross an episode. The numbers are deterministic (seeded bootstraps and subsamples).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_section.compute import compute
from dataset_section.figures import plot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compute", help="episode files -> dataset_numbers.json")
    c.add_argument("--data", action="append", required=True, help="episode file (repeatable, one --label each)")
    c.add_argument("--label", action="append", required=True, help="split name: train, val or test")
    c.add_argument("--out", default="reports/dataset")
    c.add_argument("--sha256", action="store_true", help="hash the files (slow for the 8 GB train file)")
    p = sub.add_parser("plot", help="dataset_numbers.json -> figures and tables")
    p.add_argument("--out", default="reports/dataset")
    args = ap.parse_args()
    if args.command == "compute":
        if len(args.data) != len(args.label):
            raise SystemExit("--data and --label must be given the same number of times")
        if not {"train", "test"} <= set(args.label):
            raise SystemExit("the labels must include train and test")
        compute(args.data, args.label, Path(args.out), args.sha256)
    else:
        for path in plot(Path(args.out)):
            print("wrote", path)


if __name__ == "__main__":
    main()
