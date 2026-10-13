# Environments

Two Python projects with separate environments (JAX and PyTorch do not share one).

| Side | Install | Tests |
|---|---|---|
| JAX: expert, collector, analysis, experiments | `pip install -r environment/jax-requirements.txt` | `pytest purejaxql/tests analysis/tests experiments/tests` |
| PyTorch: H-JEPA | `pip install -r environment/hjepa-requirements.txt` (CPU torch: see the header of the file) | `cd H-JEPA && PYTHONPATH=. pytest tests` |

The `reference-freeze-*.txt` files list the exact versions of the environments the tests ran in (macOS arm64, Python 3.12). They are a record of what was known to work, not a cross-platform lock file: write a real lock for the cluster (`pip freeze` or `uv pip compile` on the target machine) when the first GPU runs start, and commit it here.

Pins that matter: `jax<=0.4.38` needs `flax<=0.10.3` (newer flax crashes with it), `craftax==1.6.1`, `torch==2.7.1`, `gymnasium==1.2.3`, `stable-pretraining==0.1.7`.

If `pip` points at a private index with expired credentials, use `PIP_INDEX_URL=https://pypi.org/simple` for a throwaway environment instead of changing the configuration.
