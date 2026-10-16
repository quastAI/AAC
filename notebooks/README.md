# Notebooks

| Notebook | What it does |
|---|---|
| [d1_sampling.ipynb](d1_sampling.ipynb) | RunPod A100 with the network volume at `/workspace`: install the JAX GPU environment, collect the D1 training data (train, val, test) in the background, check that it replays on the CPU, write the D0 report, upload the files to a Hugging Face dataset repo |

The notebooks are stored without outputs. Edit the configuration cell (paths, checkpoints, frame counts, shares) and run the cells from the top; a restarted pod skips the collection runs that finished. Notes on the pod: [environment/README.md](../environment/README.md).
