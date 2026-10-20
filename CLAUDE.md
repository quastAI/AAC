# CLAUDE.md

Hierarchical JEPA world models (LeWM, HWM, H-JEPA) on Craftax-Classic, planning to diamond. `README.md` has the research questions, method, experiments and usage; this file has what is needed to work in the repo.

## Layout

- `H-JEPA/` PyTorch: world models (`h_jepa/`), planning and environments (`stable_worldmodel/`), tests (`tests/`). Imported from a fork with `git subtree`.
- `purejaxql/` JAX: PQN-RNN expert and the episode collector (`scripts/collect_craftax_episodes.py`). Also a subtree.
- `analysis/` (numpy, h5py): the package `dataset_section/` (the appendix figures and tables of the dataset section; run from `analysis/`: `python -m dataset_section compute` reads each split as one set, every frame, and writes `dataset_numbers.json`, `plot` makes the figures and tables from that JSON alone (needs matplotlib), `splits` writes the per-split composition tables with a provenance JSON; modules: `data`, `episodes`, `actions`, `spectral`, `change`, `compute`, `tables`, `figures`, `splits`) and `compare_runs.py`. `experiments/`: `experiments.yaml` (job specs: drafts of the training and planning jobs, D0 = `dataset_section compute`, D1) and `run.py`. `environment/`: install lists, reference freezes. `notebooks/`: the RunPod notebook (expert, D1, D0, Hub upload) (see `environment/README.md`, section RunPod). `.github/workflows/tests.yml` is untested (written without runner access).
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

Settled with the user: goal test `new_achievements` with all achievements counting (random baseline as reference); protocol A main task set = offsets with progress, per offset; data = final expert (diamond filter) + rollouts of earlier snapshots of the expert's training (see 2026-10-16 below), seed ranges 0 / 1M / 2M for train / val / test, headline runs use the test worlds; level-1 step = 1 environment step. Earlier: Discrete CEM at every level; VQ latent actions with a factorized codebook above level 1; IDM (cross-entropy) on at every level (ablated per model later); native 63x63 frames with a ViT of patch 7; actors are learned separately later (frozen, after the world model); the band of horizons is meant to become the default. Planner design (proposals, trust region, actors) is deferred; README "Open decisions" lists the rest.

Settled on 2026-10-13 (review session): tasks and all planning evaluation come from diamond-reaching expert episodes only (`source == 0`); all achievements count, no per-group extras for now; horizons are fixed over the episode (`horizon_schedule: fixed`); strides and depth come from the dataset section so that the top level comfortably spans the expert's typical time to diamond (a quantile, for example the 75th); LeWM is just the hierarchy with one level, so the depth is a setting (1 to 4), the first trained model is the level-1-only one and a random baseline runs on the same tasks; the data mix (shares counted in frames) is decided after the data exists and its behaviour has been analysed; episodes of sources noised (1) and early (3), which may die, stay in the loss of every level, without an ablation, and uniform random episodes (2), if any, train level 1 only; upper-level CEM starts from code sequences of the training data (actors replace that later); the terminal (death) frame is padded for the level steps. Death and starvation beyond that (a learned survival head, longer top-level clips) are left open until the data is analysed.

Settled on 2026-10-16 (D0 redesign, the script was removed on 2026-10-20, see below): D0 reported facts and makes no stride or depth choice. Headline: the entity frequency gap of the paper on the 7 probe entities of `probe_eval.py` (Welch window 256, sensitivity 64/128/512, an episode bootstrap, the same estimator for the view and terrain from the `state` column as extras); an entity whose power sits in the two lowest bins is `limited` by the window (staircases, walks, the day) and the gap is then a lower bound. The threshold 2x is the paper's boundary between six environments, a heuristic; a large gap is near certain here, so the information is in the ladder. The ladder clusters entities by the median time between their events (the spectral period cannot order the slow end), reports a stride candidate (median) and a band candidate (p10 to p90, CV as irregularity) per cluster, and the clips left at the band's top; later experiments decide the strides. Spectra, events and tasks use the expert episodes; the gap is also given per source and snapshot and for the placeholder mix. Thresholds that are heuristics: gap 2x, cluster ratio 2x, CV 0.5 / 1, critical frames (food, drink or energy at most 2, or health at most 3). Checks against the environment's own clocks (food 26, drink 21, energy 31, the day 300) print WARNING when the columns or events are wrong. Dropped: the stride-tuple enumeration and the level-1 step item.

Settled on 2026-10-19 (dataset first): the experiment table (E1-E8) was removed from the README. Experiments are not planned yet: first the dataset is understood (descriptive figures and tables, without experiment framing; the plan was in `analysis/README.md`, removed on the same day, see git history), then the LeWM baseline, then step by step. The E-ids and C1 in `experiments/experiments.yaml` and its tests are drafts, not a plan.

Settled on 2026-10-20 (dataset section): the appendix figures and tables are the package `analysis/dataset_section/`, independent of the D0 script, which was removed with its ladder, task, survival and per-source sections (git history has them). Each split is one set with every frame (train as a whole, not chunks; the old `Data` class sampled every second episode above 2M frames). Content: episode length (train and test) and time to the diamond (test), the entity frequency gap (test only, the pure expert), achievements per episode and the first-unlock ladder, the change matrix (net change per state component and lag 1 to 800, view pooled, world-frame terrain and mobs up to lag 6), clip coverage per stride (clips depend on context + rollout only), action coverage and chunk information. Validation is only compared with train (distances in the JSON). The `D0` experiment id stays and runs `dataset_section compute`.

## Diagnostics (all in `H-JEPA/h_jepa`)

`health.py` holds the metric functions, the `Thresholds` (heuristics, to be tuned on real runs) and `diagnose_training/planning/probing` returning `Finding`s (critical/warn/info with a hint). `training_monitor.py` (callback: live, per epoch, end report, optional probing), `planning_metrics.py` (`RecordingPolicy`, `RecordingSolver`, `RandomRecordingPolicy`, `summarize_planning`), `probe_eval.py`, `eval_tasks.py`, `time_run.py`. The goal test is `new_achievements` (progress-defined); tasks have one world seed each. A random-policy baseline on the same tasks is needed to read any planning number.

## Not built yet

Variable-horizon code (irregular-spacing sampler, `action_duration` through the hierarchical encode, horizon coupling in the hierarchical solver), learned jumps, protocol B with key states (needs goal switching in the policy), planner proposals/trust region/actors (decision deferred). Open: the final goal test and offsets, data mix, level-1 step size, a real lock file and the SLURM settings for the cluster. The expert is trained by the user.

## Configs

One training config per model type (`craftax_hjepa`, `craftax_hwm`) and one planning config (`craftax_plan`) serve every depth: `num_levels=1..4` and `levelN.stride=..` are settings (level 1 alone is the flat model, LeWM). The blocks of levels above `num_levels` stay in the config and are ignored (`data.dataset_kwargs` for the dataset, `_drop_levels_above_model` for planning). In `craftax_plan` the top level plans its own `horizon`; a level below plans one upper step, so its horizon is the stride of the level above (`horizon_schedule: fixed`).

## Pitfalls seen

- The expert checkpoint must contain BatchRenorm `batch_stats` (the training script saves them now); older checkpoints cannot be used.
- A level that follows the plan of the one above must have a horizon in [stride, 2 x stride) with one subgoal (`check_subgoal_horizon` raises otherwise); `craftax_plan` derives it from the stride. The Craftax eval configs use `horizon_schedule: fixed` (the Push-T `decreasing` default shrinks the horizon over the budget and skips upper levels late). `receding_horizon` counts level-1 steps.
- Evaluation needs a task file (`eval_tasks.py`) passed as `load_eval_trajs_path`; `eval.chunk_size` must be null without it. `output.dir` is relative to the parent of the policy directory (or absolute for `policy=random`).
- Tasks are one per world seed (the evaluation asserts distinct seeds). Random agents solve some tasks (wake_up, drink, sapling): always compare with the random baseline. Smoke runs on untrained models only check plumbing.
- Probe scores below zero mean too few samples for the latent dimension (`PROBE_FEW_SAMPLES`).
- Shell: zsh does not split unquoted variables; `timeout` is missing; `sleep` loops in the foreground are blocked, use `until ...; do sleep 5; done` as a background command; the safety check rejects `zsh -c "..."` wrappers, write a script file instead.
- Collection: the collector appends and `--target-frames` counts the frames of one run, so a repeated run adds a full share on new seeds. `run.py` writes a marker `$HJEPA_HOME/collect_done/<job>` after each finished collect job (local and SLURM) and skips those; a killed run has none and starts again (its partial episodes stay).
- Expert training overrides need the `alg.` prefix on the command line (`alg.SAVE_PATH=...`, `alg.RESUME=true`): the group file's own value beats a top-level `SAVE_PATH=...`, which is silently ignored.
- Expert training: the chunked loop (`chunk_fn` in `pqn_rnn_craftax.py`) must keep the greedy test a real `lax.cond`: under `vmap` the state arrives batched, the cond becomes a select and the 10000-step test runs in every update (fixed for one seed, several seeds still do it; a test guards it). Time the expert with `purejaxql/scripts/time_expert.py` before choosing `TOTAL_TIMESTEPS`: the run length also sets the learning-rate and epsilon schedules and cannot be extended later.
- RunPod: the image's `LD_LIBRARY_PATH=/usr/local/cuda/lib64` (CUDA 12.8) makes JAX segfault at the first matmul next to the pip CUDA 12.9 libraries; run JAX without it (the notebook strips it). Seen on a pod with Python 3.12.
- GPU: `purejaxql[jax_envs]` alone installs the CPU jax on Linux; use `environment/jax-gpu-requirements.txt` (`jax[cuda12]==0.4.38`). Episodes collected on a GPU must pass `purejaxql/scripts/check_replay.py` on the CPU before anything is trained or planned on them. Never run the JAX tests with the python of `Craftax/.venv` (jax 0.11).
- `hjepa_forward` logs only metrics whose key contains `loss`, which is why some statistics are named `*_loss_levelN`.
