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

| Folder | Contents | Planned changes |
|---|---|---|
| [Craftax](Craftax) | Environment | Pixel rendering and state export for data collection |
| [purejaxql](purejaxql) | PQN-RNN expert | Classic env config, checkpoint saving, achievement logging |
| [H-JEPA](H-JEPA) | World-model training and planning | Craftax data loader, discrete actions, CEM solver, variable-stride dataset and planner |

Pipeline: train expert → collect and filter episodes → train world models → plan and evaluate → probe. Run commands are added here as each stage is implemented.

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
