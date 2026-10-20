"""Command line of the dataset section: `python -m dataset_section <command>`, run from the `analysis/` folder.

    compute  episode files -> dataset_numbers.json (every number of the figures and tables, with provenance)
    plot     dataset_numbers.json -> fig_*.pdf, table_*.tex, change_matrix.csv (needs matplotlib, no data files)
    splits   episode files -> table_<split>.tex, dataset_tables.csv and .json (composition by policy)

    python -m dataset_section compute --data expert_train.h5 --label train --data expert_val.h5 --label val \
        --data expert_test.h5 --label test --out ../reports/dataset/ [--sha256]
    python -m dataset_section plot --out ../reports/dataset/
    python -m dataset_section splits --data ... --label ... --checkpoints <hf>/checkpoints --out ../reports/dataset/ [--sha256]

Every split is read as one set, with every frame; the labels must be train, val or test (compute and plot need train and test).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .compute import compute
from .figures import plot
from .splits import write_split_tables


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dataset_section", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, text in (("compute", "episode files -> dataset_numbers.json"), ("splits", "episode files -> the split tables")):
        c = sub.add_parser(name, help=text)
        c.add_argument("--data", action="append", required=True, help="episode file (repeatable, one --label each)")
        c.add_argument("--label", action="append", required=True, help="split name: train, val or test")
        c.add_argument("--out", default="reports/dataset")
        c.add_argument("--sha256", action="store_true", help="hash the files (slow for the 8 GB train file)")
        if name == "splits":
            c.add_argument("--checkpoints", help="expert checkpoint folder with its *_config.yaml (for the train steps)")
    p = sub.add_parser("plot", help="dataset_numbers.json -> figures and tables")
    p.add_argument("--out", default="reports/dataset")
    args = ap.parse_args(argv)
    if args.command == "plot":
        for path in plot(Path(args.out)):
            print("wrote", path)
        return
    if len(args.data) != len(args.label):
        raise SystemExit("--data and --label must be given the same number of times")
    if args.command == "compute":
        if not {"train", "test"} <= set(args.label):
            raise SystemExit("the labels must include train and test")
        compute(args.data, args.label, Path(args.out), args.sha256)
    else:
        write_split_tables(args.data, args.label, Path(args.out), args.checkpoints, args.sha256)


if __name__ == "__main__":
    main()
