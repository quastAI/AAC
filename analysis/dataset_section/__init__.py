"""The figures and tables of the dataset section (appendix) of the paper, from the episode files.

`compute` reads each split as one set (every frame, no sampling) and writes one JSON with every number; `plot` reads
only that JSON. Entry point: `analysis/dataset_figures.py`. Modules:

* `data`: constants (achievements, actions, `state` layout), `Split` (the small columns of one file), provenance;
* `episodes`: episode length, time to the diamond, achievements, clip coverage;
* `actions`: action coverage and the information of action chunks about their effect;
* `spectral`: the entity frequency gap of the paper;
* `change`: the change matrix (share of frame pairs at a lag in which each state component differs);
* `compute`: one split's numbers, the comparison of validation with train, the JSON;
* `tables`, `figures`: LaTeX tables, csv and matplotlib figures from the JSON.
"""
