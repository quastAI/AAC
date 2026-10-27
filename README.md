# Hierarchical JEPA World Models on Craftax-Classic

Learning hierarchical latent world models from pixels and planning with them all the way to **diamond**.

## Introduction

Planning with a learned world model gets harder as the goal gets further away. Prediction errors compound over long rollouts, and in the latent space of a flat model the distance to a far-away goal carries almost no signal about progress. H-JEPA ([Zhang, Terver et al., 2026](https://arxiv.org/abs/2610.06805)) addresses this with a stack of action-conditioned JEPA world models trained end-to-end. Each level predicts further ahead in its own learned latent space, and planning proceeds top-down: the top level optimises progress toward the goal, and its predictions become subgoals for the level below. The paper demonstrates this on continuous-control and manipulation tasks.

This project transfers H-JEPA to [Craftax-Classic](https://github.com/MichaelTMatthews/Craftax), a procedurally generated, discrete-action, long-horizon benchmark. Reaching diamond requires a long chain of dependent achievements (table, wood and stone pickaxes, furnace, coal, iron, iron pickaxe), which makes it a natural test for hierarchical planning. We compare H-JEPA against a flat world model and a shared-latent hierarchy (HWM), analyse what each level learns to represent, and study two extensions: levels that predict over a band of horizons instead of a fixed stride, and levels that learn their own time-independent jumps (in the spirit of THICK, Gumbsch et al., ICLR 2024).

## Research questions

| | Question |
|---|---|
| **RQ1** Hierarchy | How much does a hierarchy of learned latent spaces (H-JEPA) improve planning success over a flat world model and a shared-latent hierarchy (HWM) on Craftax-Classic, and how does this change with goal distance, up to diamond? |
| **RQ2** Representation | What information does each level of H-JEPA retain or discard, and how does this relate to the temporal frequency of the underlying entities? |
| **RQ3** Variable horizons | To what extent does letting each level predict over a band of horizons, with the planner choosing the horizon, improve planning over a fixed stride per level? |
| **RQ4** Learned jumps | Can a level learn time-independent, meaningful jumps from observation-action data alone, and how does planning with them compare to fixed and band strides? |

The work goes step by step. The dataset is collected and described (dataset section). The current phase is the flat model.

## Current phase: flat model

The first deliverable is a strong flat model (level 1 alone, `num_levels=1`): the baseline for the hierarchy, and the place to learn what carries over to it. Every variant goes through the same cycle:

1. **Train.** Loss curves and the training diagnostics.
2. **Probe suite** (after training, before planning; `probe_suite.py`). Every entity of the game along one chain: the pixels (ceiling), the ViT's patch tokens (untrained ViT as control), the CLS token, the latent (linear and MLP; untrained latent as control), a pasted sprite, one step (collect, place, make, mob motion), the rollout (scored only where the entity changed, against copying the last latent). The verdict is the first link that fails; rarity, look-alike sprites, the latent's variance spectrum and loss attribution say why. Also the questions the game asks of the view (a resource in view and how near, walkable neighbours, table or furnace adjacent, lava, zombies, arrows in line), the tiles ahead and around the player, the faced tile by what changed it (with an oracle bound and a context swap on the predictor), and the displacement between two latents. Every R^2 comes with the exact accuracy, the error, the variance and the majority baseline. Probes are fitted on the training worlds, scored on the held-out worlds, intervals from an episode bootstrap.
3. **Planning settings** (default model only). Goal offset, planning horizon and CEM settings, chosen on the validation worlds, then frozen for every variant. Designed when the default model exists.
4. **Planning**, each against a random policy on the same tasks, with example videos. The flat variants run only a cheap P1 (set A: offset 8, the probe-readable horizon, which is also the planning horizon, 300 tasks, a budget of 16 steps); P2 and P3 are for the final flat-against-hierarchy comparison:
   - **P1** fixed-offset goals (protocol A);
   - **P2** diamond through oracle subgoals: the expert's frames at multiples of the goal offset in the same world, switched after a fixed budget per subgoal (placeholder 2x the offset);
   - **P3** diamond from the final goal only (protocol B).

**Default model.** `craftax_hjepa num_levels=1`: history 3, `rollout_n` 1, the CLS latent of the ViT, IDM on (not ablated: previous work says it is needed), the Push-T coefficients.

**Variants**, one at a time against the current best (interactions are not tested). The flat model stays until it is sound (the latent holds what the game needs at level 1, and P1 clearly beats the random baseline at offset 8).

- **A1, first (an ablation, not a model candidate):** F1 with balanced clip sampling (`balance.py`, `data.balance.enabled=true`; SIGReg stays 0.09). Clips with rare faced blocks (iron, coal, diamond, lava, plants) or rare events (crafts, collecting iron, coal, diamond) are drawn up to 8 times as often, lifting an item toward 5% of the draws; items from fewer than 500 scenes are left alone (few transitions: a diamond is collected about once per episode, 1,800 times in the training data), so the model does not memorise them. The weights use simulator labels (`state`) a real agent would not have, so the result says what the latent keeps when rare things are seen often (is the loss of the faced rare tiles a matter of frequency or of the objective?), not what we would train. On the training split: faced iron 1.8% to 5.1% of the draws, coal 2.0% to 5.0%, faced diamond 0.42% to 2.3%, collecting diamond 0.27% to 1.5%, making the iron sword 0.26% to 1.4%; 0.99M effective clips of 1.99M, and each diamond-collect scene is drawn about 240 times in 10 epochs against 30 now. A cap of 4 gave diamond only about 3 times its share (1.3%) and was too timid; 16 would leave a third of the effective clips. Overfitting shows in the probes (fitted on train clips, scored on validation clips) and in the IDM and validation curves against F1. The cap and the scene minimum are heuristics; the run's `balance.json` has every item before and after.
- **V1** (full runs, F1 settings otherwise): the rollout model (history 1, `rollout_n` 4: does the planner's history 1 suffice, do 4-step rollouts bring back view content), and SIGReg weight 0.03 (F1: 0.09; it whitens the latent, and the untrained latent held more of the view);
- **later, designed when V1 is read:** chunked behaviour cloning as an encoder regulariser (the next 4 expert actions from z_t, labels `action_policy`; needs what lies ahead), and a temporal-distance head (latent distance tracks the steps between two frames; offsets of 16 or fewer, hard negatives against the clock, one-way);
- **revisit only:** encoder without projector. LeWorldModel keeps the projector because the ViT's final LayerNorm keeps SIGReg from being optimised, so the variant would drop that LayerNorm and use an identity projector. Only if after V1 and the heads the rare faced tiles and the vitals stay weak while the CLS token still holds them;
- **deferred:** multi-step IDM (overlaps with the temporal-distance head, its target is ambiguous), the IDM weight, delta prediction (only if the default barely beats copying), CLS vs patch tokens in the predictor (about 81 times the tokens per planning step, the levels above take one vector), the auxiliary map decoder (a diagnostic: capacity or objective).

Everything in the cycle is built except the search for the planning settings.

### F1 result (default model, 2026-10-23)

10 epochs (155,510 steps, batch 128), seed 42, A100. Run files: `data/runs/F1_flat_seed42/` (local, not in git; its README has the provenance and gaps). All numbers on held-out worlds.

- **Health.** No collapse: effective rank 101 of 192 (57 after epoch 1), no dead dimensions, no loss spikes. One-step prediction error 0.20 of copying the last latent (0.29 after epoch 1); open-loop error at the planning horizon (10 steps) 0.41 of copy (0.60); action lift 0.86; IDM accuracy 0.93 (chance 0.20). Validation loss still falling at the end. The `OVERFIT` warning (validation loss 1.9x training) compares total losses: after epoch 1 the gap was all SIGReg (1.74 vs 1.0, prediction and IDM terms equal); not checked for the last epoch.
- **Probes** (512 clips). Fast / medium / slow 0.33 / 0.90 / 0.91 (untrained 0.40 / 0.66 / 0.77); inventory 0.82, vitals 0.79, block faced 0.53. Mobs in view (0.08) and blocks in view (0.32) stay below the untrained encoder (0.39, 0.60) at every epoch. Predictions beat copying the last latent up to 32 steps (fast, medium) and 16 (slow); at 8 steps fast 0.23 vs 0.09, medium 0.86 vs 0.75, slow 0.90 vs 0.89. Crafts are predicted almost fully (0.97 of the real change), placements 0.78, `place_stone` worst (0.42: a failed press looks like half a success). The probe scores settled by epoch 2 to 3; later epochs improved the rollouts.
- **Where the tile is lost** (the earlier `tile_probe.py`, block faced, MLP probe): the ViT's patch token at the faced tile reads it at 1.00, the CLS token before the projector 0.89, the latent 0.78 (linear: 0.85 vs 0.70). Rare tiles lose most at the projector (CLS → latent): coal 0.68 → 0.27, iron 0.67 → 0.20, diamond 0.76 → 0.26, crafting table 0.71 → 0.43. A class-balanced probe does not recover them (balanced accuracy: latent 0.51, CLS 0.69).
- **Probe suite** (2026-10-27, 512 clips per split, 781 s on an A100; `health/probe_report.md`, the earlier reports in `health/probe_first_version/`). The latent is a progress tracker plus a near-field model. Kept: the resources in the inventory (R^2 0.91 to 0.97; exact 87 to 99%, stone 67%), the tools (exact 99% or more), facing 0.85, sleeping, daylight, achievements, a table or furnace next to the agent (a pasted table moves the latent 3.45 median steps when adjacent, 0.11 at 4 tiles). Weak: vitals (exact 32 to 66%), the faced block (accuracy 0.68 linear, 0.79 MLP; by the earlier tile probe iron 0.20, coal 0.27, diamond 0.26 against 0.67 to 0.76 in the CLS token, and recall follows how often a block is faced, rank correlation 0.92), mobs in view (AUROC 0.64 to 0.76 against 0.85 to 0.94 untrained), ores and lava in view (count R^2 0.03 to 0.14 against 0.37 to 0.74 untrained), arrows (a pasted arrow moves the latent 0.01 steps). The CLS token holds more than the latent on almost every target. Crafts are predicted at 0.96 to 1.0 of the real change, collecting iron 0.24 and coal 0.40; inventory changes are recovered 0.94 at one step, 0.59 at 8; for goals without an inventory change the predicted distance to the goal does not fall (rank correlation 0.25 at 8 steps, 0.01 at 16). History 1 and 3 roll out alike. The lower R^2 of the wood tools against the iron tools is the variance of the target (held in 93% vs 42% of the frames), not the encoding.
- **Next.** The probe suite again on F1 (the additions of 2026-10-27) and F3 (F1 on the cheap set A), then A1 (the balanced ablation) and V1 (rollout model, SIGReg 0.03), one pod and notebook per run, each with the probe suite and the cheap P1 against F1.

## Method

```
pixels o_t ──ViT, patch 7──► z_t (CLS) ──► predictor ──► z_t+1        level 1: step = 1 env step
                                              ▲
                  a_t, one-hot(17) ───────────┘
```

- **Encoder.** A ViT on the native 63×63 frame (patch 7, one patch per tile); the latent is its CLS token, projected.
- **Predictor.** A causal transformer over a history of 3 latents, conditioned on the embedded action, predicts the next latent.
- **Losses.** Next-latent MSE, SIGReg on the latents, and an inverse-dynamics cross-entropy on the action index.
- **Planning.** CEM over the 17 discrete actions, forward passes only, minimising the latent distance to the goal frame; receding horizon.
- **Hierarchy (later phases).** Each level above encodes one latent of the level below every `stride` steps (a latent MLP in H-JEPA, the identity in HWM); its actions are VQ codes of the pooled chunk of lower-level actions, and planning runs top-down, the predictions of a level being the subgoals of the level below. The code exists (`num_levels` 2 to 4, [loss routing](docs/loss_routing.svg)); CLAUDE.md has its contracts.

## Setup

**Environment.** Craftax-Classic (JAX). World models use pixel observations. The symbolic state is used only by the expert and for probing and evaluation. Diamond is achievement 19 and requires the iron pickaxe.

**Expert.** PQN-RNN from [purejaxql](purejaxql), trained on `Craftax-Classic-Symbolic-v1`. The collector renders pixels from the same state.

**Data.** The final expert's episodes that reach the diamond (`source 0`; all tasks come from them) and rollouts of earlier snapshots of the expert's training (`source 3`, `early`; they may die). The mix is counted in frames (`collection.share` in `experiments.yaml`); the test split holds only the final expert. Seeds are the split: training from 0, validation from 1,000,000, test from 2,000,000. Exploration takes its tasks from the validation worlds, headline results from the test worlds. Episodes that ended in death are extended by copies of their dead frame (`data.dataset.terminal_pad`). Training clips are drawn uniformly over every start position (uniform over frames). The figures and tables are in `analysis/dataset_section` (see Usage).

**Goal test.** A task's required set is the achievements the goal frame has and the start frame lacks. Success means all of them are unlocked, in any order and by any route; progress is the share unlocked. The planner never sees this: it minimises the latent distance to the goal frame; the test only judges the result. All achievements count, including those a random agent gets by itself (wake up, drink, sapling): the random baseline on the same tasks is the reference. Every reported success number comes with the planner compute used. The planner is goal-conditioned and reward-free, so results are phrased as "reaches X% of the expert's achievement rate given goal frames", not "beats the expert".

## Repository layout

One repository holds all code. [H-JEPA](H-JEPA) and [purejaxql](purejaxql) were imported from our forks with `git subtree` (history kept, no submodules), so a result is traceable to one commit and the paper release is one tag.

```
AAC/
  H-JEPA/        world models, planning, Craftax environment (PyTorch)
  purejaxql/     PQN-RNN expert and episode collector (JAX)
  papers/        reference papers (PDFs are local, see papers/README.md)
  analysis/      dataset section figures and tables (`dataset_section`) and run comparison (numpy and h5py; matplotlib to plot)
  notebooks/     RunPod notebooks: expert and data (D0), relabeling, the flat model (F1, F3-F5), one per variant; helpers in pod.py
  experiments/   job specs and a runner (print, local, SLURM)
  environment/   install lists and reference freezes of the two environments
  CLAUDE.md      working notes for Claude Code
```

The two code folders are separate Python projects (JAX and PyTorch do not share an environment). They only meet in the HDF5 episode file below.

**Upstream bases** (diff against these to see what changed): H-JEPA `97054ec` (github.com/quastAI/H-JEPA), purejaxql `47af6d7` (github.com/quastAI/purejaxql). **External:** [Craftax](https://github.com/MichaelTMatthews/Craftax) is an unmodified pinned dependency (v1.6.1, it already renders pixels from a state); the environment wrapper is adapted from [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). Both clones are kept outside this repository.

## Usage

Pipeline: train expert → collect episodes → dataset section → train the world model → probe → plan and evaluate.

### Expert and episodes (purejaxql)

```bash
cd purejaxql && pip install -e ".[jax_envs]"      # pins: jax<=0.4.38 needs flax<=0.10.3
python scripts/time_expert.py                                      # seconds per update, steps per second, hours per run length (size TOTAL_TIMESTEPS from it)
python purejaxql/pqn_rnn_craftax.py +alg=pqn_rnn_craftax_classic   # saves params + BatchRenorm stats + config
python scripts/collect_craftax_episodes.py --out data/expert.h5 --checkpoint models/.../..._vmap0.safetensors \
    --num-episodes 1000 --seed-start 0
```

- **Expert checkpoints.** Training runs in chunks of `CHECKPOINT_INTERVAL` (0.05) of the updates. After each chunk it saves a policy snapshot `..._vmap0_u<updates>.safetensors` with a `.json` (updates, env steps, latest greedy-test metrics) and the full state `..._state.safetensors` (overwritten). After a kill, `RESUME=true` with the same configuration continues and ends with the same policy as an uninterrupted run (tested). The final policy is also saved as `..._vmap0.safetensors`, the path the collector takes.
- The collector keeps episodes that reach diamond and ends them `--tail-steps` (500) after it, or earlier on death. `--source early` labels the rollouts of a snapshot (`--checkpoint ..._vmap0_u<updates>.safetensors`, usually `--no-require-diamond`); `--eps` gives noised-expert episodes. A second run on an existing file appends and skips seeds already in it. `experiments/run.py plan D1 --expert-checkpoint <ckpt> --early-checkpoint <snapshot> ...` prints the collection commands of the data plan.
- **Provenance.** Every collector run adds a record to the file attribute `runs` (JSON: flags, git commit and dirty flag, craftax/jax versions, the checkpoint's sha256 and, for a snapshot, its training state, and the outcome). A killed run stays marked `finished: false` and is safe to repeat.
- **Cross-backend replay.** Collected on a GPU, replayed by the planning environment on the CPU: `JAX_PLATFORMS=cpu python scripts/check_replay.py --data <file.h5>` replays a sample and compares every column (discrete ones exactly, float leaves of the full state to 1e-6). The RunPod notebook [notebooks/runpod_dataset.ipynb](notebooks/runpod_dataset.ipynb) runs the whole chain on one pod, [notebooks/runpod_flat.ipynb](notebooks/runpod_flat.ipynb) the flat-model cycle; pod notes in [environment/README.md](environment/README.md).

**Episode file.** One flat HDF5 in the layout of H-JEPA's `HDF5Dataset` (`ep_len`, `ep_offset`, one array per column). Per episode there are L+1 frames and L actions; `action[t]` belongs to frame `t`, the last one is NaN.

| Column | Shape, dtype | Content |
|---|---|---|
| `pixels` | (N, 63, 63, 3) uint8 | rendered frame |
| `state` | (N, 1345) float32 | symbolic observation |
| `action` | (N, 1) float32 | the action the game executed, index 0-16: the policy's action, noop (0) while the player is asleep (Craftax ignores every action then) |
| `action_policy` | (N, 1) float32 | the action the policy chose |
| `reward` | (N,) float32 | reward of the transition into this frame |
| `achievements` | (N, 22) uint8 | cumulative flags, Craftax-Classic order |
| `full_state` | (N, 20952) uint8 | snapshot of the complete env state (about 1.5 KB per frame on disk) |
| `inventory` (12), `vitals` (4), `facing`, `daylight`, `position` (2), `mobs_in_view` (4) | (N, k) small ints, `daylight` float32 | symbolic state for probing and data analysis: item counts, health/food/drink/energy, direction 1-4, light, player row/col, zombies/cows/skeletons/arrows in the 7x9 view |
| `dead` | (N,) uint8 | 1 on the last frame of an episode that ended in death (health 0, or walked into lava), else 0 |
| `step_idx`, `ep_idx`, `seed`, `source`, `policy_updates` | (N,) int | `seed`: world seed; `source`: 0 expert, 1 noised expert, 2 random, 3 early checkpoint; `policy_updates` (int32, -1 unknown): training updates of the snapshot that acted |

- **Executed actions.** About 17% of the steps are taken while asleep, when the game replaces every action by a noop; `action` holds that noop and `action_policy` the policy's choice. The collector writes both; files collected before were relabeled with `purejaxql/scripts/relabel_asleep.py` ([notebooks/relabel_dataset.ipynb](notebooks/relabel_dataset.ipynb) for the Hub files; `check_replay.py` checks the rule).
- **Replayable.** An episode is determined by `seed` and its actions (`action` or `action_policy`, they replay identically): `reset_key, step_key, _ = split(PRNGKey(seed), 3)`; the reset uses `reset_key`, the step at `state.timestep == t` uses `fold_in(step_key, t)`. The planning environment follows the same convention.

### Dataset section (analysis)

Run from `analysis/`: `python -m dataset_section compute --data expert_train.h5 --label train --data expert_val.h5 --label val --data expert_test.h5 --label test --out ../reports/dataset/ [--sha256]` writes `dataset_numbers.json`: episode length (weighted by frames) and time to the diamond, the entity frequency gap of the test split, the first-unlock ladder, the change matrix (share of frame pairs at lags 1 to 800 in which each state component differs), clips per stride, action coverage of the executed actions split into succeeded and failed steps (a step taken asleep is a noop), chunk information, file hashes, collector runs and the git commit. `plot --out ../reports/dataset/` makes `fig_*.pdf`, `table_*.tex` and `change_matrix.csv` from that JSON alone; `splits --checkpoints <hf>/checkpoints ...` writes one LaTeX table per split (share, length, diamond rate, return, achievements and death rate per policy and snapshot). Job D0 in `experiments.yaml` runs `compute`.

### Planning environment (H-JEPA)

`swm/CraftaxClassic-v1` ([H-JEPA/stable_worldmodel/envs/craftax](H-JEPA/stable_worldmodel/envs/craftax)): pixel frames, symbolic observation, snapshot state API. Needs `craftax==1.6.1` with `jax==0.4.38` (both pinned in the `env` extra; jax 0.5+ splits keys differently and would generate other worlds, so the env and the collector also set `jax_threefry_partitionable=False`). JAX runs on the CPU unless `JAX_PLATFORMS` is set.

- Use `image_shape=(63, 63)` in `World` (native resolution). `reset(seed=s)` generates the world of dataset episode `s`.
- `terminated` = goal test succeeded (`World` counts it as success). Death and the 10000-step horizon are `truncated`.
- `World.evaluate_from_dataset` starts from dataset states through the callables `_set_state` (`full_state`) and `_set_goal_state` (`goal_full_state`).
- Goal test `new_achievements` (default): `info` has `goal_required` and `goal_progress`. A goal without progress never succeeds.

### Training, probing and planning (H-JEPA)

```bash
cd H-JEPA/h_jepa
python main_hjepa.py --config-name craftax_hjepa num_levels=1                                    # the flat model (2 to 4 add levels)
python probe_suite.py --run <run dir> [--checkpoint <ckpt>] [--out <dir>]                        # the probe suite (also run by the training at its end)
python eval_tasks.py --dataset craftax_expert_val --mode offset --offsets 25 50 100 --num-tasks 200 --eval-budget 400 --out tasks.pt
python eval.py --config-name craftax_plan policy=random load_eval_trajs_path=tasks.pt eval.num_eval=200 eval.eval_budget=400 eval.goal_offset_steps=400 output.dir=<abs dir>   # baseline
python eval.py --config-name craftax_plan policy=ckpts/craftax/<run>/seed42 load_eval_trajs_path=tasks.pt ...
python time_run.py --config-name craftax_hjepa --eval-config-name craftax_plan                   # step and solve timing
python ../../analysis/plot_training.py --run <run dir>                                           # loss curves and health metrics of a run (health/loss_curves.png)
python ../../experiments/run.py plan F1                                                          # print the commands of one job group
```

- **Task modes** (`eval_tasks.py`): `offset` (goal = start + offset, pairs with progress; P1), `chain` (the tasks of `episode`, the planner's goal switching through the expert's frames every offset, `--subgoal-budget` steps each; P2), `episode` (from reset to the expert's diamond frame; P3), `achievements` (goal = the d-th next unlock). In `chain` only the goal frame the planner sees switches (the payload's `goal_schedule`); the env's goal test stays the final goal, so P2 and P3 are scored alike. One task per world seed, from expert episodes only (`source == 0`).
- **Configs** in `h_jepa/config/{train,eval}`: one training config per model type (`craftax_hjepa`, `craftax_hwm`) and one planning config (`craftax_plan`) for every depth; with `num_levels=1` both training configs are the same flat model. Hyperparameters are the Push-T ones until tuned on Craftax. Dataset names `craftax_expert_train` / `_val` / `_test` are the collector output in `$HJEPA_HOME`.
- **Jobs.** `experiments/experiments.yaml` holds the data plan (D1), the dataset numbers (D0) and the jobs of the flat phase (F1 default model, F2 planning settings, F3-F5 planning P1-P3 on task sets A, C, B, V1 the rollout model and SIGReg 0.03 against F1); `experiments/run.py` prints the commands, runs them locally, or writes SLURM scripts with dependencies (`slurm.yaml`). Jobs whose output exists are skipped. `--run NAME` keeps one trained run of an experiment with its reference runs (one pod per variant; the reports are named after the run).
- **Actions.** One-hot (`data.OneHotColumn`, `level1.wm.num_actions: 17`); NaN becomes zeros. The normalizer artifact holds no action statistics and records `num_actions`, so evaluation does not de-normalize action indices.
- **IDM.** `idm_coeff: 0.1` (placeholder). Cross-entropy over the action index; logs `idm_accuracy_loss_level1` (reference: the best constant guess, at least 1/17).
- **Planner.** `stable_worldmodel.solver.CategoricalCEMSolver`, same interface as `GradientSolver`, logs `solve_records`. Level 1 plans `action_block` slots of 17 actions. Horizons are fixed over the episode (`horizon_schedule: fixed`); `receding_horizon` counts environment steps.

### Diagnostics

Training, planning and probing each write a report that names collapses and other failures. The thresholds are heuristics set before the first run (`health.Thresholds`); adjust them with real numbers.

| Stage | When | What | Where |
|---|---|---|---|
| Training | live every 2000 steps | effective rank, dead dimensions, latent scale (SIGReg holds it near std 1), latent change per step, predictor vs copy-last-latent and vs shuffled actions, IDM accuracy, raw gradient norms, loss spikes (per level) | `health/*` in the logger, findings printed when they first appear |
| Training | every epoch, long validation clips | adds open-loop drift (planning history 1 and training history), cost geometry (latent distance vs time gap), IDM on held-out worlds; two epochs in a row with a critical finding stop the training (`health.stop_after_critical_epochs`) | `health_val/*`, `health/epoch_*.json` |
| Training | at the end | all of it, findings, the probe suite | `health/report.md`, `summary.json`, `probe_report.md` |
| Planning | per task and replanning | progress, latent distance to the goal, CEM improvement and entropy, FLOPs, executed-action diversity; plan diagnostics replay the chosen plan in a cloned simulator (prediction error vs random plans, optimism, cost ranking, distance to the data) | `planning_report.md`, `planning_health.json`, `planning_records.json`, `videos/task_*.gif` (the first `eval.videos` tasks: agent next to goal, step, progress, unlocks) |
| Probing | at the end of training, or `probe_suite.py` | per entity (block types, mobs, inventory, vitals, facing, sleeping, daylight, achievements) the chain pixels -> ViT tokens (layers 3, 6, 9, last) -> CLS -> latent -> sprite swap -> one step -> rollout, with untrained controls and a verdict; the 7x9 tile map from every stage (per class, per distance from the player); why (rarity vs look-alike sprites, spectrum, loss attribution); the change test at k = 1 to 64 (history 1 and 3); events incl. collecting; mob motion; what the latent distance sees; the distance to a goal along the expert path; IDM per action; view concepts (a block in view, the nearest tile of a type, walkable neighbours, table or furnace adjacent, lava, zombies, arrows in line); the 6 tiles around the player in its frame; the faced tile one step ahead by cause, with an oracle and a context swap; the displacement between two latents; exact accuracy, error, variance and majority baseline next to every R^2 | `probe_report.md`, `probe_summary.json` |

Findings are `critical` (collapse or a failure that makes the model unusable), `warn`, or `info`, each with a hint. Failure signals beyond collapse include: predictor not better than copying the last latent, actions ignored, rollout drift beyond copy, latent distance that saturates after a few steps, IDM at chance, plans predicted worse than random plans (exploitation), predicted progress that reality does not deliver, and probes no better than an untrained encoder. `analysis/compare_runs.py` compares planning against the random baseline, compute scaling, training and probing across runs.

### Tests

`pytest purejaxql/tests analysis/tests experiments/tests` (JAX environment, see `environment/`) and `cd H-JEPA && PYTHONPATH=. pytest tests` (H-JEPA environment). They include a frame-exact comparison of the environment with a collector episode (`tests/data/craftax_classic_episode.npz`, regenerate with `tests/data/make_fixture.py`, its docstring has the collector command) and a CPU smoke run of the full chain (training, task dump, planning on the real environment) whose success rates carry no meaning.

## Compute

About 15 A100-hours per month (~180 in total). One seed per variant; three seeds only for headline results. A training run and a planning evaluation are timed first (`time_run.py`), and the runs are sized from those numbers.

Measured on an A100 (RunPod, 2026-10-23): the flat model trains at 0.064 s per step (batch 128, clips of 4 frames, 11.2 M parameters, 4.7 GB), 15,551 steps and 16.6 min per epoch, so 10 epochs take 2.8 GPU-hours; the loader gives 3,500 clips/s for the 2,000 the step uses. A planning solve of 8 environments takes 0.71 s at about 1% GPU use (the solver processes 5 environments at a time, `solver.solvers.level1.batch_size`): an upper bound of 18 s per task at a budget of 400 steps and 133 s at 3000, that is about 1 h for 200 tasks of set A and 3.7 h for 100 tasks of set B or C, per model, with the simulator not counted. Planning costs more than training until the batch size and the task sets are set (F2).

The F1 run itself: 12.4 to 12.8 it/s while only training, 11.4 on average with the epoch-end evaluation and the probes alongside, so about 3.8 A100-hours for 10 epochs (the 2.8 above counts the step alone), plus the restarts. The probe suite on F1 took 781 s (512 clips per split, before the view concepts, the tiles around the player, the faced-tile analyses and the displacement probe were added; those add trained probes, not timed yet).

## Open decisions

- Oversampling rare transitions: successful crafts are about 0.1% of the steps and clips are drawn uniformly over frames. The event split of F1 exists: crafts are shown at 0.96 to 1.0 of the real change, rare collects are not (iron 0.24, coal 0.40). Still open.
- The planning settings (goal offset, horizon, CEM) and the P2 budget per subgoal: from the default model.
- The planner beyond CEM (trust region, learned actors): deferred to the hierarchy.

## References

1. Zhang, Terver, Rabbat, LeCun, Balestriero. H-JEPA. [arXiv:2610.06805](https://arxiv.org/abs/2610.06805), 2026.
2. Zhang, Terver et al. Hierarchical planning with latent world models (HWM). [arXiv:2604.03208](https://arxiv.org/abs/2604.03208), 2026.
3. Maes et al. LeWorldModel. [arXiv:2603.19312](https://arxiv.org/abs/2603.19312), 2026.
4. Balestriero, LeCun. LeJEPA (SIGReg). [arXiv:2511.08544](https://arxiv.org/abs/2511.08544), 2025.
5. Gumbsch et al. THICK: hierarchical world models with adaptive temporal abstractions. ICLR 2024.
6. Matthews et al. Craftax. ICML 2024.
7. Gallici et al. Simplifying deep temporal difference learning (PQN). 2024.
