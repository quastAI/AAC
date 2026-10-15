# Risk register

Risks that remain where the implementation is sound. Bugs are not listed here; they are fixed in the code. L = likelihood, I = impact (low / med / high). From the code review of 2026-10-13 (commit a3a8bc6); statuses updated after the fixes (commit 8e12895).

## Objective and planning

| ID | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R1 | The goal frame shows inventory counts, vitals, the 7x9 view, mobs and daylight, but no achievement flags. Latent distance to the goal frame may track position and view more than progress (wake_up, eat_cow, place_*, defeat_* leave little or no trace in the frame). | high | high | Check before E1 that latent distance falls with achievement progress (E4 cost geometry on expert clips); report success per achievement type |
| R2 | Top-level horizon (4 x stride 4 = 16 env steps) is far below the task distances (25-400, diamond hundreds to thousands). The papers only validate horizons that cover the task; H-JEPA reports level-1 distance nearly flat away from the goal. | high | high | Choose stride and depth (E2) so that top horizon x top step covers the offsets; report success per offset |
| R3 | Argmin over ~3000 deterministic model evaluations per solve selects plans where the model is wrong (exploitation). Half of the upper-level samples are now code sequences of the data, but the other half and the CEM refit can still leave the support. | high | med | Plan diagnostics already measure it (optimism, prediction vs random plans); trust region and learned actors are the open decision |
| R4 | Mobs, night static and spawns are stochastic; a deterministic latent predictor regresses to the mean outcome, and action-only latent codes cannot choose among outcomes. | med | med | Probe mobs per level; compare open-loop drift day vs night |
| R5 | Subgoals change at every replan (every 2 env steps), so the level-1 target can flip between plans. | med | med | Subgoal tracking metric (measured at the next replan) |
| R6 | Diamond is hard even for the symbolic expert (PPO-RNN only "somewhat reliably" on Classic, Craftax App. F.3); protocol B without key states may give 0 everywhere. | high | med | Report progress by depth, not only the diamond rate |

## Latent actions (VQ)

| ID | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R7 | Codebook collapse looks healthy: a collapsed codebook gives 100% upper IDM accuracy, and dead-code re-seeding splits a collapsed encoder output into many codes (perplexity stays high). | med | high | Log the re-seed rate and pre-quantisation variance relative to code spacing; read IDM accuracy together with perplexity |
| R8 | Rare chunks (crafting, furnace, iron pickaxe) use codes that the absolute re-seed threshold keeps killing; exactly the steps the diamond chain needs. The re-seed rate is now logged (`VQ_CODE_CHURN`); the threshold is still absolute per batch. | med | high | Relative threshold or periodic re-seed; per-action code purity on validation |
| R9 | The two code groups can be redundant (joint usage near 64, not 4096) and CEM samples them independently. | med | med | Joint perplexity and tuple coverage metric |
| R10 | Upper levels keep slow features, including world identity; this costs latent capacity (cancels only in same-world differences). | med | med | E5 (IDM on/off) with the within-episode variance share |

## Data and environment

| ID | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R11 | GPU collection vs CPU env: float differences (world generation noise, light level) could change rare continuations. The in-process replay check cannot see it and compares only rewards and achievement counts. | low | high | Replay a handful of GPU-collected episodes in the CPU env (frames and full_state) once |
| R12 | A low expert diamond rate pushed expert seeds into the next range. **Mitigated:** `--seed-end` per run in D1, and D0 checks the overlap of all three files. | low | high | read the D0 overlap line |
| R13 | Diamond filter = world selection: all tasks come from worlds the greedy expert solved, one expert trajectory per world. | certain | med | State it with every result |
| R14 | Non-expert share is counted in episodes; random and noised episodes are short, so by frames (what the clip sampler sees) it may be a few percent. | high | med | Report the frame share in D0 |
| R15 | Depth 3 may lack clips (the H-JEPA Push-T 3-level result failed on that). | med | med | D0 clip-count table per depth |
| R16 | No provenance in the HDF5 (Craftax/JAX version, checkpoint, eps, commit); only the full_state size check guards the layout. | med | low | Write attrs in the collector |
| R17 | Expert training (1e9 steps) runs in one jit and saves only at the end; a wall-time kill loses all of it. | med | med | Chunked training with intermediate saves |

## Compute and evaluation

| ID | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R18 | `time_run.py` leaves out validation, health evaluation, data loading, end-of-training probing and planning eval; planning wall-clock is measured inside the FLOP counter and the forced SDPA math backend. | high | med | Time a full short job end to end before sizing the budget |
| R19 | Pixels are gzip-compressed in 100-frame chunks and read per sample; data loading may bottleneck. | med | med | Include the loader in the timing run |
| R20 | Health thresholds are heuristics set before any run. | high | low | Calibrate on the first LeWM run |
| R21 | Task files are reused when they exist (the runner skips tasks_*.pt); changing the `tasks:` spec in experiments.yaml silently reuses the old file. Eval chunks are now protected (signature with task hash and config hash) and compare_runs checks the task hash. | med | med | delete the task file when the spec changes |
| R22 | Under DDP, probing runs on rank 0 while other ranks continue with later end-of-training callbacks. | low | med | Barrier around probing |

## Added in the discussion of 2026-10-13

| ID | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R23 | Death by starvation or thirst takes about 330 steps (food/drink hit 0 after about 190-230 steps, then health falls 1 per about 16 steps). The longest training clip is about 49 env steps and the planner looks at most 64 steps ahead, so slow death is neither learned nor planned against. The diamond chain (protocol B) needs the plan to eat and drink on its own. | high | high | Top-level clips that span about 350 steps (cumulative stride of about 110 or more), a survival head (death is a frame in the data: health 0 or lava), D0 statistics on death causes and times per source |
| R24 | Noise of epsilon 0.3 per step corrupts most 4-step chunks (only about 24% are clean), so the upper levels see noised-expert chunks mostly as perturbed macro-actions, the argument that excludes uniform random data applies in part. | med | med | Choose the noise spectrum (for example 0.05, 0.1, 0.2) from D0: death rate per epsilon and share of clean chunks; temporally extended noise if needed |
| R25 | The data mix is open until the data exists. Expert episodes are long and noised/random ones are short; shares by episodes and by frames differ a lot. | high | med | Define the mix in frames, report it in D0 |
| R26 | Padding the terminal frame (absorbing death state) makes "copy the last latent" easier at the end of dying episodes and can dilute the loss if too many copies are added. | med | low | A limited number of copies (up to one top-level step), a `dead` flag per frame |
