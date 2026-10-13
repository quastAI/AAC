# CLAUDE.md

Hierarchical JEPA world models (LeWM, HWM, H-JEPA) on Craftax-Classic, planning to diamond. `README.md` has the research questions, method, experiments and usage; this file has what is needed to work in the repo.

## Layout

- `H-JEPA/` PyTorch: world models (`h_jepa/`), planning and environments (`stable_worldmodel/`), tests (`tests/`). Imported from a fork with `git subtree`.
- `purejaxql/` JAX: PQN-RNN expert and the episode collector (`scripts/collect_craftax_episodes.py`). Also a subtree.
- `analysis/` (numpy, h5py): `dataset_stats.py`, `compare_runs.py`. `experiments/`: `experiments.yaml` (E1-E8, D0, D1, C1) and `run.py`. `environment/`: install lists, reference freezes. `.github/workflows/tests.yml` is untested (written without runner access).
- `papers/` PDFs are gitignored. `Craftax/` and `stable-worldmodel-craftax/` are read-only reference clones, gitignored, never edit them.
- The two code folders only share the HDF5 episode file (column table in the README).

## Environments (nothing persistent is checked in)

Test venvs are throwaway: recreate them with `python3 -m venv`. The default pip index is a private registry with expired credentials, so for a throwaway venv use `PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL=` and do not change pip config. The `python3` on PATH may belong to `Craftax/.venv` (jax 0.11): do not use it for tests.

- JAX side: `pip install -e "purejaxql[jax_envs]"` (jax 0.4.38 needs flax <= 0.10.3, craftax 1.6.1, h5py). Tests: `pytest purejaxql/tests analysis/tests experiments/tests` (the last two only need numpy, h5py, pyyaml). `environment/` has the install lists.
- H-JEPA side: torch 2.7.1, torchvision 0.22.1, gymnasium 1.2.3, numpy, loguru, tabulate, einops, h5py, scipy, hdf5plugin, tqdm, typer, rich, pillow, pytest, plus `craftax==1.6.1 jax==0.4.38 jaxlib==0.4.38 flax<=0.10.3`, `omegaconf hydra-core` and the training extras (`transformers`, `stable-pretraining==0.1.7`, `lightning`, `torchmetrics`, `scikit-learn`, `matplotlib`, `datasets`). The project pins Python 3.11; the tests were run on 3.12. Tests: `cd H-JEPA && PYTHONPATH=. pytest tests`.
- `h_jepa/*.py` import each other as top-level modules: run scripts from `H-JEPA/h_jepa` with `PYTHONPATH=..:.` and `HJEPA_HOME` set.
- CPU smoke run of the training script: `trainer.accelerator=cpu trainer.precision=32 trainer.devices=1 loader.batch_size=8 loader.num_workers=0 loader.persistent_workers=false loader.prefetch_factor=null loader.pin_memory=false trainer.max_epochs=1 +trainer.limit_train_batches=6 +trainer.limit_val_batches=2 planning_eval.enabled=false resume_every_n_steps=0 num_workers=0`.

## Conventions

- Style follows the surrounding file. `H-JEPA/stable_worldmodel/` and `H-JEPA/tests/`: single quotes, 79 columns (`ruff check --isolated --select E,F --line-length 79`; format with `--config "format.quote-style='single'"`). `h_jepa/` and `purejaxql/`: double quotes, longer lines. Only format files you created: `gd.py`, `solver.py` and others already exceed 79 columns.
- Commits: small and logical, message explains why. End with the attribution trailer given in the session. Push only when asked.
- Never commit data, checkpoints, PDFs or the reference clones (all gitignored).
- Keep README and this file in step with the code; the README stays minimal and note-style.

## Contracts that must not drift

- **Episodes** are replayable from `seed` + actions: `reset_key, step_key, _ = split(PRNGKey(seed), 3)`, step at `state.timestep == t` uses `fold_in(step_key, t)`. The collector (`purejaxql`) and the env (`H-JEPA/stable_worldmodel/envs/craftax/env.py`) both follow it; `H-JEPA/tests/data/craftax_classic_episode.npz` guards it. `World` resets envs with the dataset's `seed` column.
- `full_state` is a flat uint8 snapshot of the Craftax state pytree (leaf order), made by `pack_state` in the collector and the env. Craftax stays pinned at 1.6.1.
- Env: `terminated` = goal success, death/horizon = `truncated`; the env freezes after death or the horizon (after goal success it keeps stepping; `World` ORs the success flags).
- Discrete actions: level-1 one-hot (`level1.wm.num_actions`); the normalizer artifact has no action stats and records `num_actions`. Upper levels: `VQActionEncoder` (groups x codes, optional duration k); planners read `model.get_action_spec()`.
- `CategoricalCEMSolver` mirrors `GradientSolver` (`configure(..., level=)`, `solve`, `solve_records`). Its duration slot is pinned to the default stride unless `allowed_durations` says otherwise.

## Decisions taken

Settled with the user: goal test `new_achievements` with all achievements counting (random baseline as reference); protocol A main task set = offsets with progress, per offset; data = expert (diamond filter) + 25% non-expert (half noised expert eps 0.3, half random), seed ranges 0 / 1M / 2M for train / val / test, headline runs use the test worlds; level-1 step = 1 environment step. Earlier: Discrete CEM at every level; VQ latent actions with a factorized codebook above level 1; IDM (cross-entropy) on at every level (E5 ablates per model); native 63x63 frames with a ViT of patch 7; actors are learned separately later (frozen, after the world model); the band of horizons (E7) is meant to become the default. Planner design (proposals, trust region, actors) is deferred; README "Open decisions" lists the rest.

Settled on 2026-10-13 (review session): tasks and all planning evaluation come from diamond-reaching expert episodes only (`source == 0`); all achievements count, no per-group extras for now; horizons are fixed over the episode (`horizon_schedule: fixed`); strides and depth come from the dataset analysis (D0) so that the top level comfortably spans the expert's typical time to diamond (a quantile, for example the 75th); LeWM is just the hierarchy with one level, so the depth is a setting (1 to 4), the first trained model is the level-1-only one and a random baseline runs on the same tasks; the data mix (shares of diamond, noised-expert and uniform random episodes, counted in frames, and the noise values) is decided after the data exists and its behaviour has been analysed; uniform random episodes train level 1 only, noised-expert episodes (which may die) stay in the loss of every level, without an ablation; upper-level CEM starts from code sequences of the training data (actors replace that later); the terminal (death) frame is padded for the level steps. Death and starvation beyond that (a learned survival head, longer top-level clips) are left open until the data is analysed.

## Diagnostics (all in `H-JEPA/h_jepa`)

`health.py` holds the metric functions, the `Thresholds` (heuristics, to be tuned on real runs) and `diagnose_training/planning/probing` returning `Finding`s (critical/warn/info with a hint). `training_monitor.py` (callback: live, per epoch, end report, optional probing), `planning_metrics.py` (`RecordingPolicy`, `RecordingSolver`, `RandomRecordingPolicy`, `summarize_planning`), `probe_eval.py`, `eval_tasks.py`, `time_run.py`. The goal test is `new_achievements` (progress-defined); tasks have one world seed each. A random-policy baseline on the same tasks is needed to read any planning number.

## Not built yet

Depth as a setting 1-4 with strides (one config instead of one per depth), the per-source loss mask for uniform random episodes, frame-based data shares in the collector and the runner, code-sequence proposals for CEM, terminal-frame padding with a per-frame `dead` flag from the collector, a coverage table (levels x strides x horizon against the time to diamond) in D0, E7 code (irregular-spacing sampler, `action_duration` through the hierarchical encode, horizon coupling in the hierarchical solver), E8 (learned jumps), protocol B with key states (needs goal switching in the policy), planner proposals/trust region/actors (decision deferred). Open: the final goal test and offsets, data mix, level-1 step size, a real lock file and the SLURM settings for the cluster. The expert is trained by the user.

## Pitfalls seen

- The expert checkpoint must contain BatchRenorm `batch_stats` (the training script saves them now); older checkpoints cannot be used.
- A level that follows the plan of the one above must have a horizon in [stride, 2 x stride) with one subgoal (`check_subgoal_horizon` raises otherwise); change a stride and the lower horizons with it. The Craftax eval configs use `horizon_schedule: fixed` (the Push-T `decreasing` default shrinks the horizon over the budget and skips upper levels late). `receding_horizon` counts level-1 steps.
- Evaluation needs a task file (`eval_tasks.py`) passed as `load_eval_trajs_path`; `eval.chunk_size` must be null without it. `output.dir` is relative to the parent of the policy directory (or absolute for `policy=random`).
- Tasks are one per world seed (the evaluation asserts distinct seeds). Random agents solve some tasks (wake_up, drink, sapling): always compare with the random baseline. Smoke runs on untrained models only check plumbing.
- Probe scores below zero mean too few samples for the latent dimension (`PROBE_FEW_SAMPLES`).
- Shell: zsh does not split unquoted variables; `timeout` is missing; `sleep` loops in the foreground are blocked, use `until ...; do sleep 5; done` as a background command; the safety check rejects `zsh -c "..."` wrappers, write a script file instead.
- `hjepa_forward` logs only metrics whose key contains `loss`, which is why some statistics are named `*_loss_levelN`.
