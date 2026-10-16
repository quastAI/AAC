# Environments

Two Python projects with separate environments (JAX and PyTorch do not share one).

| Side | Install | Tests |
|---|---|---|
| JAX: expert, collector, analysis, experiments | `pip install -r environment/jax-requirements.txt` | `pytest purejaxql/tests analysis/tests experiments/tests` |
| PyTorch: H-JEPA | `pip install -r environment/hjepa-requirements.txt` (CPU torch: see the header of the file) | `cd H-JEPA && PYTHONPATH=. pytest tests` |

The `reference-freeze-*.txt` files list the exact versions of the environments the tests ran in (macOS arm64, Python 3.12). They are a record of what was known to work, not a cross-platform lock file: write a real lock for the cluster (`pip freeze` or `uv pip compile` on the target machine) when the first GPU runs start, and commit it here.

Pins that matter: `jax<=0.4.38` needs `flax<=0.10.3` (newer flax crashes with it), `craftax==1.6.1`, `torch==2.7.1`, `transformers==4.51.3` (a fresh resolve can pick a release that does not import with torch 2.7.1; the 153 tests passed with 4.51.3 on Python 3.11), `gymnasium==1.2.3`, `stable-pretraining==0.1.7`.

If `pip` points at a private index with expired credentials, use `PIP_INDEX_URL=https://pypi.org/simple` for a throwaway environment instead of changing the configuration.

## On a RunPod pod (A100, network volume at `/workspace`)

Nothing here has run on a GPU yet (written on macOS): check the first notebook run. The notebook [notebooks/runpod.ipynb](../notebooks/runpod.ipynb) does the JAX side of this (expert training, data collection) in one go.

- **JAX (collector, expert):** `pip install -r environment/jax-gpu-requirements.txt` from the repository root, in its own venv. Without the `jax[cuda12]` pin the collector runs on the CPU. Check with `python -c "import jax; print(jax.devices())"`.
- **PyTorch (H-JEPA):** its own venv, `pip install -r environment/hjepa-requirements.txt` (torch 2.7.1 from PyPI is a CUDA 12 build; the image's own torch, probably 2.8, is not used). JAX in this venv is the CPU build on purpose: the planning environment pins `JAX_PLATFORMS=cpu` and must not compete with torch for the GPU. Python 3.11 is the project's pin (the RunPod PyTorch images ship 3.11).
- **Keep on the volume:** the repository clone, the venvs (a restarted pod reuses them), `HJEPA_HOME` (data, checkpoints, reports), the expert snapshots. Set `HJEPA_HOME=/workspace/data` and `HDF5_USE_FILE_LOCKING=FALSE` (HDF5 can fail to lock files on network volumes).
- **Dataloader:** the training config asks for `num_workers: 12`, `prefetch_factor: 8`, `persistent_workers`. Check `nproc` and `df -h /dev/shm` on the pod; with a small `/dev/shm` or few cores lower `loader.num_workers` and `loader.prefetch_factor`.
- **Cross-backend replay.** Episodes collected on the GPU must replay identically on the CPU planning environment (floating point differs between backends). `JAX_PLATFORMS=cpu python purejaxql/scripts/check_replay.py --data <file.h5>` replays a sample on the CPU and compares every column; the notebook runs it after the collection.
- **A restarted collection** skips the runs that finished (`$HJEPA_HOME/collect_done/`); a run that was killed starts again and its partial episodes stay in the file.
