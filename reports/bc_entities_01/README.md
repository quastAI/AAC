# BC entities, first run (state analysis)

`bc.py` on the ground-truth state, the final expert's episodes (`source` 0), run on a RunPod pod (`notebooks/runpod_bc_entities.ipynb`, paths `/root/data/...`) and copied here on 2026-10-31. What an MLP on the `state` column of frame t needs of each entity to predict the expert's next 4, 8 and 16 actions, and to tell how many steps lie between two frames (temporal distance). Simulator labels: an analysis, the world models never see them. How it is measured: `bc.py` docstring and `CLAUDE.md` (2026-10-29).

Settings (all four runs): executed actions (`action`), 8 epochs, hidden 512, depth 3, lr 1e-3, batch 4096, 3 partner frames per batch and entity, 200 bootstrap resamples, the top 6 entities and the coarse groups trained again without, seed 0, 128 validation episodes. The commit of the run is not recorded in the files.

| task | validation samples | cross-entropy (marginal) | skill | accuracy, first / last step |
|---|---|---|---|---|
| chunk 4 | 104,811 | 1.209 (2.130) | 0.92 | 0.70 / 0.51 |
| chunk 8 | 104,299 | 1.375 (2.130) | 0.76 | 0.69 / 0.45 |
| chunk 16 | 103,275 | 1.531 (2.128) | 0.60 | 0.67 / 0.40 |
| temporal distance | 102,985 | 1.541 (2.485) | 0.94 | 0.37 |

## Contents

| File | What |
|---|---|
| `bc_entities_chunk4.md`, `chunk8`, `chunk16`, `distance` (`.md` and `.json`) | per task: coarse parts, block types, mobs, distance rings, inventory, vitals, facing / daylight / sleeping, skill lost by tile, the model trained again without an entity; the `.json` has every number and the arguments |
| `bc_entities.png` | the notebook's plot of the four tasks: the top entities and the skill lost per tile |

## Not in this run

The multi-step inverse dynamics (`bc.py --chunk --inverse 1 4 8`, files `bc_entities_inverse<k>.*`): the task was added after this run and has not been run on the state yet. The 20-episode smoke run of the same pod (`bc_smoke/`, its cross-entropy is above the marginal) was left out.
