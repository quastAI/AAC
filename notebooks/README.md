# Notebooks

| Notebook | What it does |
|---|---|
| [runpod.ipynb](runpod.ipynb) | One A100 pod with the network volume at `/workspace`: Hugging Face login and the JAX GPU environment, a smoke test and the full training of the expert (detached), the checkpoints to the Hub, the choice of early snapshots, the data collection (detached), the replay check, the dataset numbers (job D0), and the data and numbers to the same Hub repo |

The notebook is stored without outputs. Edit the configuration cell (paths, repo id, run length, frame counts, shares) and run the cells from the top; every long step runs detached and a step that finished is skipped when its cell is run again. Notes on the pod: [environment/README.md](../environment/README.md).
