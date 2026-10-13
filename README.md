# Hierarchical JEPA World Models on Craftax-Classic

Learning hierarchical latent world models from pixels and planning with them all the way to **diamond**.

## Introduction

Planning with a learned world model gets harder as the goal gets further away. Prediction errors compound over long rollouts, and in the latent space of a flat model the distance to a far-away goal carries almost no signal about progress. H-JEPA ([Zhang, Terver et al., 2026](https://arxiv.org/abs/2610.06805)) addresses this with a stack of action-conditioned JEPA world models trained end-to-end. Each level predicts further ahead in its own learned latent space, and planning proceeds top-down: the top level optimises progress toward the goal, and its predictions become subgoals for the level below. The paper demonstrates this on continuous-control and manipulation tasks.

This project transfers H-JEPA to [Craftax-Classic](https://github.com/MichaelTMatthews/Craftax), a procedurally generated, discrete-action, long-horizon benchmark. Reaching diamond requires a long chain of dependent achievements (table, wood and stone pickaxes, furnace, coal, iron, iron pickaxe), which makes it a natural test for hierarchical planning. We compare H-JEPA against a flat world model (LeWM) and a shared-latent hierarchy (HWM), analyse what each level learns to represent, and study two extensions: levels that predict over a band of horizons instead of a fixed stride, and levels that learn their own time-independent jumps (in the spirit of THICK, Gumbsch et al., ICLR 2024).

## Research questions

| | Question |
|---|---|
| **RQ1** Hierarchy | How much does a hierarchy of learned latent spaces (H-JEPA) improve planning success over a flat world model and a shared-latent hierarchy (HWM) on Craftax-Classic, and how does this change with goal distance, up to diamond? |
| **RQ2** Representation | What information does each level of H-JEPA retain or discard, and how does this relate to the temporal frequency of the underlying entities? |
| **RQ3** Variable horizons | To what extent does letting each level predict over a band of horizons, with the planner choosing the horizon, improve planning over a fixed stride per level? |
| **RQ4** Learned jumps | Can a level learn time-independent, meaningful jumps from observation-action data alone, and how does planning with them compare to fixed and band strides? |

## Setup

**Environment.** Craftax-Classic (JAX). World models use pixel observations. The symbolic state is used only by the expert and for probing and evaluation. Diamond is achievement 19 and requires the iron pickaxe.

**Expert.** PQN-RNN from [purejaxql](purejaxql), trained on `Craftax-Classic-Symbolic-v1`. Rolled-out episodes are filtered to the successful ones (diamond collected), and the collector renders pixels from the same state.

**Data.** Mostly expert episodes, plus a share of random or noised-expert episodes for level-1 dynamics coverage. The ratio is open. Validation, probing and planning tasks use held-out world seeds. We report a per-level clip count for each depth, to see early whether deeper levels still have enough data.

## Models

| Model | Hierarchy | Role |
|---|---|---|
| **LeWM** | Single level (JEPA + SIGReg) | Flat baseline |
| **HWM** | Several temporal levels, identity encoders above level 1, one shared latent space | Isolates temporal decomposition |
| **H-JEPA** | A distinct learned latent space per level, trained end-to-end | Main model |

Implemented on top of the [H-JEPA](H-JEPA) code. Craftax-specific changes:

- Discrete level-1 actions with CEM planning at level 1. Upper levels use continuous macro-actions.
- Small encoder suited to Craftax's small frames.
- Inverse dynamics (IDM) per model, as decided by the ablation in E5.

## Evaluation

| Protocol | Definition | Metric |
|---|---|---|
| **A. Offset goals** | Start and goal frames from held-out expert trajectories at fixed offsets (values to be fixed) | Success under a goal test defined in advance on the symbolic state |
| **B. Full episodes** | From reset, with goal frames taken from an expert rollout in the same world seed | Achievement rates by achievement depth, and diamond rate |

Every reported success number comes with the planner compute used. The planner is goal-conditioned and reward-free, so results are phrased as "reaches X% of the expert's achievement rate given goal frames", not "beats the expert".

## Experiments

| ID | RQ | Experiment | Status |
|---|---|---|---|
| E1 | 1 | LeWM vs HWM vs H-JEPA at matched depth, protocols A and B | planned |
| E2 | 1 | Depth: 2 vs 3 levels, strides derived from the data (event durations, frequency gap) | planned |
| E3 | 1, 2 | Rollout loss on/off, set per model (not per level), on H-JEPA and LeWM | planned |
| E4 | 2 | Per-level probes, frequency gap, cost geometry | planned |
| E5 | 1, 2 | IDM on/off, set per model (not per level), reported with collapse diagnostics (within-episode variance share, effective rank, action sensitivity) and planning. Run on LeWM first, then on H-JEPA at the chosen depth. HWM follows the H-JEPA outcome (run it too if budget allows). One seed for the ablation, three seeds only for the setting used in the headline results. | planned |
| E6 | 1 | Cost-only: level-1 planning with native vs projected upper-level cost (evaluation only, optional) | optional |
| E7 | 3 | Variable band vs fixed stride, including band width | planned |
| E8 | 4 | Learned jumps vs fixed vs band, including the jump constraint type | planned |

Not doing: compute-Pareto sweeps, stagewise data specialisation, detachment grids, DROID, multi-GPU SIGReg sharding, a hand-crafted symbolic oracle, clustering and obs-change boundary baselines, and boundary-gradient and training-schedule ablations.

## Repository layout

All project code lives in this repository. [H-JEPA](H-JEPA) and [purejaxql](purejaxql) were imported from our forks with `git subtree`, so their history is kept and there are no submodules. A result is traceable to one commit of this repository, and the paper release is a single tag.

```
AAC/
  H-JEPA/        world-model training and planning (PyTorch)
  purejaxql/     PQN-RNN expert and episode collection (JAX)
  experiments/   one launcher per experiment E1-E8 (hydra overrides, seeds)
  analysis/      clip-count tables, probes, plots
  papers/        reference papers (PDFs are local, see papers/README.md)
```

The two code folders are separate Python projects with their own `pyproject.toml` and lockfile, because JAX and PyTorch do not share an environment. The only interface between them is a flat HDF5 episode file in the format of H-JEPA's `HDF5Dataset`.

| Folder | Contents | Planned changes |
|---|---|---|
| [purejaxql](purejaxql) | PQN-RNN expert | Classic env config, checkpoint saving, achievement logging, episode collector (pixels rendered from the same state) |
| [H-JEPA](H-JEPA) | World-model training and planning | Craftax env wrapper, discrete actions, CEM solver, small encoder, variable-stride dataset and planner |

**Upstream bases.** Diff against these to see exactly what changed relative to the forks as imported:

| Folder | Source | Commit |
|---|---|---|
| H-JEPA | github.com/quastAI/H-JEPA | `97054ec` |
| purejaxql | github.com/quastAI/purejaxql | `47af6d7` |

**External references (not in this repository).** [Craftax](https://github.com/MichaelTMatthews/Craftax) is used unmodified as a pinned dependency (v1.6.1, which already provides state-to-pixel rendering). The Craftax Gymnasium wrapper in H-JEPA is adapted from [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel). Both are kept as read-only clones outside this repository.

Pipeline: train expert → collect and filter episodes → train world models → plan and evaluate → probe. Run commands are added here as each stage is implemented.

## Stage 1: expert and episode collection

Environment for the JAX side (the pins matter: `jax<=0.4.38` needs `flax<=0.10.3`):

```bash
cd purejaxql && pip install -e ".[jax_envs]"
```

Train the expert on `Craftax-Classic-Symbolic-v1`. This writes `<SAVE_PATH>/Craftax-Classic-Symbolic-v1/pqn_rnn_..._vmap0.safetensors` (params and BatchRenorm statistics) and the matching `_config.yaml`:

```bash
python purejaxql/pqn_rnn_craftax.py +alg=pqn_rnn_craftax_classic
```

Collect episodes. The collector rolls out the expert, keeps the episodes that reach diamond (achievement 19), and renders pixels from the same states by replaying them:

```bash
python scripts/collect_craftax_episodes.py --out data/expert.h5 --checkpoint models/.../..._vmap0.safetensors \
    --num-episodes 1000 --seed-start 0
python scripts/collect_craftax_episodes.py --out data/random.h5 --eps 1.0 --no-require-diamond \
    --num-episodes 1000 --seed-start 5000000        # uniform random, no checkpoint
```

Defaults: an episode ends `--tail-steps` (500) steps after the diamond, or earlier if the agent dies, as a margin for planning horizons that extend past it. `--no-stop-at-diamond` keeps full episodes. `--eps` between 0 and 1 gives noised-expert episodes. Running again on an existing file appends and skips world seeds already in it.

**Data format.** One flat HDF5 file in the layout of H-JEPA's `HDF5Dataset` (`ep_len`, `ep_offset`, one array per column). Per episode there are L+1 frames and L actions, and `action[t]` belongs to frame `t` (the last one is NaN).

| Column | Shape, dtype | Content |
|---|---|---|
| `pixels` | (N, 63, 63, 3) uint8 | rendered frame |
| `state` | (N, 1345) float32 | symbolic observation |
| `action` | (N, 1) float32 | action index 0-16 |
| `reward` | (N,) float32 | reward of the transition into this frame |
| `achievements` | (N, 22) uint8 | cumulative flags in Craftax-Classic order |
| `full_state` | (N, 20952) uint8 | snapshot of the complete env state, to start the planning env from any frame (about 1.5 KB per frame on disk) |
| `step_idx`, `ep_idx`, `seed`, `source` | (N,) int | `seed`: world seed of the episode; `source`: 0 expert, 1 noised expert, 2 random |

**Replayable episodes.** An episode is fully determined by `seed` and its actions: with `reset_key, step_key, _ = split(PRNGKey(seed), 3)` the reset uses `reset_key` and the step taken at `state.timestep == t` uses `fold_in(step_key, t)`. The collector checks this on every kept episode, and the planning environment follows the same convention, so a state loaded from `full_state` continues exactly as the dataset episode would under the same actions.

**Seed ranges.** World seeds are the split: use disjoint ranges for training, validation and test data, so that probing and planning tasks use held-out worlds. The ranges are not fixed yet.

Tests: `pytest purejaxql/tests`.

## Stage 2: planning environment

`swm/CraftaxClassic-v1` ([H-JEPA/stable_worldmodel/envs/craftax](H-JEPA/stable_worldmodel/envs/craftax)) is Craftax-Classic with pixel frames (63x63), the symbolic observation, and a snapshot-based state API. It is registered next to the other environments and needs `pip install craftax==1.6.1` (the `env` extra of H-JEPA). Use `image_shape=(63, 63)` in `World` to keep frames at native resolution. JAX runs on the CPU unless `JAX_PLATFORMS` is set.

- `reset(seed=s)` generates the world of dataset episode `s`, with the dynamics of the collector.
- For `World.evaluate_from_dataset`, the evaluation config passes these callables:

```yaml
callables:
- method: _set_state
  args: {full_state: {value: full_state}}
- method: _set_goal_state
  args: {full_state: {value: goal_full_state}}
```

- `terminated` means the goal test succeeded (`World` counts it as success). Death and the 10000-step horizon are `truncated`.
- The goal test is a placeholder (`achievements_inventory_test`: all achievements and inventory counts of the goal frame are met). A goal without progress over the start state is met after one step, so evaluation tasks must be chosen with progress between start and goal. The final goal test for protocol A is still an open decision.

Tests: `pytest H-JEPA/tests` (needs the H-JEPA environment). They compare the environment frame by frame with an episode written by the collector (`tests/data/craftax_classic_episode.npz`, regenerate it with `collect_craftax_episodes.py --eps 1.0 --seed-start 0 --max-steps 60 --num-envs 1 --no-require-diamond --no-stop-at-diamond`), and run `World.evaluate_from_dataset` on a dataset built from it.

## Stage 3: training and planning with discrete actions

Craftax actions are the 17 discrete indices. They enter the world model as one-hot vectors (`data.OneHotColumn`), switched on by `level1.wm.num_actions: 17`. The model, the losses and the action embedder are unchanged, only the embedder input size follows `num_actions`. A missing action (NaN, last frame of an episode) becomes an all-zero vector. The training normalizer artifact then holds no action statistics and records `num_actions`, so the evaluation does not de-normalize the planner's action indices.

Frames are used at their native 63x63 (`img_size: 63`, ViT with `patch_size: 7`, one patch per tile), so no new encoder was needed.

Configs (Push-T hyperparameters until timed on Craftax): `h_jepa/config/train/base/craftax.yaml`, `craftax_lewm.yaml`, and `config/eval/craftax_flat.yaml`. The dataset names `craftax_expert_train` and `craftax_expert_val` are placeholders for the collector's output files in `$HJEPA_HOME`.

```bash
cd H-JEPA/h_jepa
python main_hjepa.py --config-name craftax_lewm
# evaluation tasks are sampled once from a dataset, then reused:
python eval.py --config-name craftax_flat policy=random dump_eval_trajs_path=<tasks.pt>
python eval.py --config-name craftax_flat policy=<run> load_eval_trajs_path=<tasks.pt>
```

Planning at level 1 uses `stable_worldmodel.solver.CategoricalCEMSolver`: CEM over one categorical distribution per planned step, forward passes only, candidates scored with `model.get_cost`. It has the same `configure` / `solve` interface as `GradientSolver` (levels above 1 keep the gradient solver) and writes the same `solve_records`, marked `backward: false`.

### Upper levels: discrete codes, and the IDM at every level

The latent actions of the upper levels are vector-quantized (`levelN.action_encoder.vq`, `h_jepa/models/vq.py`). The sequence encoder that pools a chunk of lower-level actions is followed by a codebook (EMA updates, commitment loss, dead codes re-seeded from the batch, straight-through gradient), so an upper-level action is a code index. The planner then searches discrete actions at every level: `CategoricalCEMSolver` at level 1 plans the 17 actions, and at levels above it plans codes and passes the code vectors to the level's model (`get_action_codebook`). Removing the `vq:` block gives the continuous latent actions again (planned with the gradient solver), so the quantization is one config switch.

Codebook collapse is the main risk, so training logs `vq_perplexity_loss_levelN` (effective number of codes in use, 1 = collapsed) and `vq_usage_loss_levelN` (fraction of codes used in the batch). The first validation pass before training shows perplexity 1 because the codebook is only initialised on the first training forward.

The IDM is on at every level (`idm_coeff: 0.1`, a placeholder weight; E5 switches it per model). It is a cross-entropy classifier of the action taken between two consecutive latent states: the action index at level 1 (one slot per frameskip step) and the code index of each chunk above. It logs `idm_accuracy_loss_levelN`; chance is 1/17 at level 1 and 1/128 above. The same head could later score the plausibility of planned transitions.

Configs: `craftax_hjepa_l2/l3` and `craftax_hwm_l2/l3` (HWM: identity encoders and no SIGReg above level 1, same VQ), with `eval/craftax_l2/l3` using CEM at every level. The strides (4), the codebook (128 codes of dimension 16), the horizons and the sample counts are placeholders. The stride is to be derived from the data (E2), and the level-1 horizon must lie in [stride, 2 * stride).

Checked on CPU with a small random-policy dataset, for the single-level model and for the 2-level H-JEPA (a few batches): training for an epoch, the task dump, and planning on the real environment all run end to end. The resulting success rate is meaningless there, because of the goal-test caveat above and an untrained model.

## Compute and roadmap

About 15 A100-hours per month (~180 in total). One seed for exploration, three seeds only for headline results. We time one training run and one planning evaluation first, then size the experiments from those numbers.

1. Expert, data collection, clip-count table, frequency gap
2. Flat LeWM, probes, planning protocols
3. HWM and H-JEPA (2 and 3 levels), E1–E5
4. Variable band and learned jumps (E7, E8)
5. Final three-seed runs and write-up

## Open decisions

- Expert / random data ratio
- Offset values and goal test for protocol A
- Goal sequence for protocol B (final goal only vs key states)
- Level-1 step size (1 env step vs a few with concatenated actions)
- Encoder architecture
- Band ranges for E7

## References

1. Zhang, Terver, Rabbat, LeCun, Balestriero. H-JEPA. [arXiv:2610.06805](https://arxiv.org/abs/2610.06805), 2026.
2. Zhang, Terver et al. Hierarchical planning with latent world models (HWM). [arXiv:2604.03208](https://arxiv.org/abs/2604.03208), 2026.
3. Maes et al. LeWorldModel. [arXiv:2603.19312](https://arxiv.org/abs/2603.19312), 2026.
4. Balestriero, LeCun. LeJEPA (SIGReg). [arXiv:2511.08544](https://arxiv.org/abs/2511.08544), 2025.
5. Gumbsch et al. THICK: hierarchical world models with adaptive temporal abstractions. ICLR 2024.
6. Matthews et al. Craftax. ICML 2024.
7. Gallici et al. Simplifying deep temporal difference learning (PQN). 2024.
