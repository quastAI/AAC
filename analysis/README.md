# analysis

`dataset_stats.py` (D0), `timescales.py`, `compare_runs.py`: numbers from the episode files. This file decides the dataset figures and tables for the paper (env + dataset section, appendix first). Nothing is built yet; status per item below.

Data (local copy): `data/hf/craftax-classic-pqn-rnn-v1/data/craftax_expert_{train,val,test}.h5`, D0 report in `reports/dataset_stats/`.

## Rules

- A figure earns its place if it answers one of the reader's five questions: (1) what is the environment, (2) where do the episodes come from, (3) how big and how split, (4) what is in them, (5) which time scales are in them. Trust checks are for us, not the reader.
- **Train is one distribution** (the mix the model samples from): plot it pooled. Expert and early show as faint lines behind the pooled one. Separate groups only where the split is the point: F2 (expert and each snapshot: what the early snapshots add) and F3 (expert and early end differently, a pooled share hides both). Pooled numbers inherit the 70 / 30 frame mix, a design choice; say so in the captions.
- **Val and test have other purposes** (expert only, unseen worlds): describe them as task sets, in their own panels or rows. They are not compared with train, except by T2.
- Colours: one for the expert, a light-to-dark ramp for the snapshots (by training updates), the expert colour dashed for test.
- Use all episodes. The D0 state sections use every second train episode.

## Trust checks (before any figure is quoted)

| id | check | status |
|---|---|---|
| T1 | How episodes end: death (cause), horizon or cap; what the expert does after the diamond (47% of expert episodes die, max length 3.9k) | open |
| T2 | Same expert in all splits: expert checkpoint hash in the `runs` attribute of train / val / test, plus the expert's achievements per episode in train and test | open |
| T3 | `check_replay.py` passed on the final files | open, no record found |

Seed disjointness between the files is already checked by D0 (`seed_overlap`).

## Appendix figures

| id | figure | question | data | status |
|---|---|---|---|---|
| F1 | Episode length ECDF (log x): train pooled (expert and early faint) and test in a separate panel; death marked | 3, 4 | `ep_len`, `source`, `dead` | decided, to build |
| F2 | Achievements, two panels: share of episodes that unlock each (expert and each snapshot; train, test as expert only) and first-unlock frame; diamond p50 / p75 / p90 marked | 4 | `achievements`, `source`, `policy_updates` | decided |
| F3 | Episode ends per source: death by exhaustion, damage, lava, still alive at the cut; time to death; for the expert, steps after the diamond | 4 | `dead`, `vitals`, `achievements` | decided, needs T1 |
| F4 | Time scales. (b) inter-event time ECDF per entity, vitals by event type with the clocks 26 / 21 / 31; (a, optional) spectral period per entity as in H-JEPA appendix F (limited entities hatched with "≥", window sensitivity 64 to 512) | 5 | `timescales.py` | decided; (a) optional |
| F5 | Actions: frequency and share that changes the state, coverage per source, share of consecutive pixel-identical frames; chunk entropy and information (chunk to effect) against chunk length k; optional: run length of repeated actions (not in D0) | 4 | `dataset_stats.action_*`, `chunk_diversity`, `pixels` | decided |
| F6 | Clips per window length, 1 to 512 (all and independent, expert and early lines); caption: history 3, rollout 1, death padding | 3 | `dataset_stats.clip_counts` | decided |

## Appendix tables

| id | table | data |
|---|---|---|
| S1 | Splits and sources: seeds, episodes, frames, share per source and snapshot, diamond rate, death rate, median length (also the main-text table) | D0 composition, `runs` |
| S2 | Columns of the episode file (shape, dtype, content) | README episode file |
| S3 | Probe label balance: distribution and majority baseline per probe target (position, facing, mobs, vitals, inventory, achievements, daylight), the reference for probe scores | `probe_eval.py` targets |
| S4 | Val and test as task sets: episodes, lengths, time to the diamond, per offset the episodes with progress and the size of the required set (one task per episode: test 185, val 128) | D0 `task_feasibility`, `required_top_step` |

Main text (later, written by hand): split table, example frames from one expert episode, achievement ladder.

Dropped for now: vitals example episode, distance-horizon curves.

## Notes

- Expert lengths are selected: only episodes that reach the diamond are kept, so very short expert episodes cannot exist. Say so next to F1.
- Spectral period (F4a): a Welch window of 256 frames sees at most one cycle per 256. Position, vitals, inventory, achievements and daylight put more than half of their power in the two lowest bins, so their period is a lower bound set by the window (facing 9.7 and mobs 15.5 are measured). The gap grows with the window (8.2x, 13.8x, 22.5x, 27.5x for 64 to 512), so its value means little; it clears the paper's 2x either way. Inventory and achievements are staircases (cumulative), slow spectrally but frequent by events (median gap 2 and 7 steps): F4b is the evidence, F4a the comparison with the paper. Check the paper's own figure before building F4a.
- Plan: one script per figure in `analysis/figures/`, shared loader on `dataset_stats.Data`, PDF and PNG out, numbers cached in a small npz. Reads without `pixels` and `full_state` (the train file is 8 GB; only F5 and the example frames need pixels). Needs `matplotlib` (optional, not in the test environment).

## Open

- F1: log x, two panels (train, test).
- F5: whether to add the action run length.
