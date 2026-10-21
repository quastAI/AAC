# Notebooks

| Notebook | What it does |
|---|---|
| [runpod_dataset.ipynb](runpod_dataset.ipynb) | One A100 pod with the network volume at `/workspace`: Hugging Face login and the JAX GPU environment, a smoke test and the full training of the expert (detached), the checkpoints to the Hub, the choice of early snapshots, the data collection (detached), the replay check, the dataset numbers (job D0), and the data and numbers to the same Hub repo |
| [relabel_dataset.ipynb](relabel_dataset.ipynb) | Once, on any machine with about 20 GB of disk: downloads the three files from the Hub, relabels the actions taken while asleep as noop (`action_policy` keeps the policy's choice), checks that every file replays and follows the rule, recomputes the dataset numbers, and replaces the files on the Hub (the old ones stay in its history) |
| [runpod_flat.ipynb](runpod_flat.ipynb) | The flat model (level 1), the complete cycle, on one A100 pod: the PyTorch environment and the dataset from the Hub, a smoke run of the training on the real files (with the probes), the timing of a step, of the real data loader and of a planning solve (optionally the training file without compression), F1 (training, probes; detached, resumable), P1-P3 (F3-F5) after the planning settings are frozen, the variant V1, and the runs and reports to the Hub |

The notebook is stored without outputs. Edit the configuration cell (paths, repo id, run length, frame counts, shares; for the flat model the epochs from the timing) and run the cells from the top; every long step runs detached and a step that finished is skipped when its cell is run again. Notes on the pod: [environment/README.md](../environment/README.md).
