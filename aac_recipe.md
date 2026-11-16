# Recipe: the two-stage world model in AAC (flat model, level 1)

**For:** Claude working in the AAC repo (`H-JEPA/h_jepa`).
**Source:** the Grace grid-world study: `Grace/docs/findings.md`, notebooks `08_joint_dp` and `09_final_model` (stages 0 to D and S), results in `Grace/results/final_model/`
(`decisions.json` has every decision with its reason; `stage_*.csv` the readouts; `stage_S_timeline.csv` the stage 2 timelines).

**Status of the evidence.** One synthetic world, **one seed**, a CNN encoder with a global latent of 128, frames of 48x48 with 4x4 tiles. The large effects are probably
robust (SIGReg off; the gate; the decoded prediction). The close decisions are not and must be re-checked in Craftax: error-weighted against plain reconstruction in
stage 1, the reconstruction weight 0.3 against 0.1, and the terms dropped at a 0.05 margin (the same recipe run twice differed by up to 0.05 in the noise readout).
The stage 1 length was deliberately not studied in the grid (it is specific to the world). Every setting below is a starting point to test, not a tuned value.
**What differs between the grid and AAC:** the grid predictor was a residual action-gated MLP on two context latents plus the earlier action; AAC's is a causal transformer, here with a
history of one latent. With history 1 the earlier-action channel is gone (in the grid the facing could be drawn from it on no-op steps); the bypass through the *current* action remains
on move steps, so the gate is still needed. The grid's timeline signals (including the edit sensitivities) are specific to it and will probably not transfer: see the stop rules below.

## Goal

A latent that holds everything the predictor can learn within its budget (including small, moving and rare content: mobs, ores, the faced tile), releases what it
cannot learn (noise), stays predictable, and is used unchanged by the CEM planner (latent MSE to the goal latent). B1 loses mobs and ores in the first steps; A5
(no prediction loss) lost them too, so SIGReg and the IDM are the suspects, not the prediction loss. The grid study found the same mechanism and a schedule that avoids it.

## The model in one paragraph

**Stage 1 (availability):** the encoder reconstructs every frame (each patch's error weighted by its own size, no labels) while the predictor learns to predict the next
latent against a stop-gradient target; no SIGReg, no IDM. Everything becomes available and nothing is pruned. **Stage 2 (consolidation),** an instant switch from a stage 1
checkpoint: the reconstruction of all frames is dropped; the predictor's output is decoded and compared with the real next frame (**decoded prediction**), and the real next
frame is reconstructed only where that decoded prediction was right (**prediction-gated reconstruction**). The latent keeps what the predictor foresees, drops what it cannot
foresee, and stays predictable.

    stage 1:  x_t --E--> z_t --D--> x^_t                      reconstruction (error-weighted), all frames
              z_{t-1}, a --P--> p      vs  sg(z_next)        next-latent MSE

    stage 2:  p --D--> x^_next      vs x_next                  decoded prediction
              z_next --D--> x^_next  vs x_next, per patch weight exp(-e_i/tau),  e_i = error of the decoded prediction on patch i (sg)
              p vs sg(z_next)                                  next-latent MSE

## What changes against B1

| setting | B1 | this model |
|---|---|---|
| SIGReg (`level1.loss.embed.sigreg`) | weight 0.09 | **off** |
| IDM (`level1.idm_coeff`) | 0.1 | **0 (off)** |
| prediction target | full gradient | **stop-gradient** in both stages (`level1.wm.detach_pred_target: true`); no target-gradient term |
| decoder | none (A2's view-map decoder is a label ablation) | **new: a pixel decoder from the latent `z`** |
| losses | next-latent MSE + SIGReg + IDM | stage 1: error-weighted reconstruction + next-latent MSE; stage 2: decoded prediction + gated reconstruction + next-latent MSE |
| schedule | one cosine schedule over 10 epochs | **two stages**: constant learning rate in stage 1; stage 2 forked from a stage 1 checkpoint with a fresh optimiser |
| class labels | none | none (the grid used ground-truth class weights at first; both stages work without them) |
| context (`level1.wm.history_size`) | 3 | **1** (Craftax is turn-based; the frame shows almost the whole state the next step depends on; the planner already plans with `plan_history` 1, and AAC's probe suite found history 1 and 3 roll out alike) |
| predictor | transformer, depth 4 | **unchanged in the first run**, so that a change cannot be confused with the other changes. With one latent a transformer has nothing to attend over (it acts as an MLP with extra machinery): a residual action-gated MLP, as in the grid, is a **later ablation** |
| encoder, action embedding, planner | | unchanged |

## Architecture

- **Encoder, predictor, action embedding:** as B1. ViT-tiny (hidden 192, 12 layers, 3 heads) on the 63x63 frame, patch 7 (one patch per tile, 81 patches + CLS);
  CLS -> MLP projector -> latent `z` (192). Causal-transformer predictor (depth 4, 8 heads, MLP 1024, dim_head 64, dropout 0.1, hidden 192) over a history of **1** latent
  (`history_size` 1, `rollout_n` 1), linear embedding of the 17 one-hot actions. The clip is then 2 frames: one context frame, the action, the next frame.
- **Decoder (new): `D: z -> frame`, reading only `z`** (after the projector), never the patch tokens. The hold has to sit on the latent the predictor predicts and the planner
  compares; a decoder on the patch tokens lets the tokens hold the content while `z` drops it (the grid's two-encoder design failed exactly that way).
  The output is per patch (81 patches of 7x7x3) so that the gate and the signals below work per patch. A simple start: an MLP from `z` to 81 x 128, plus a learned
  position embedding per patch, and a shared per-patch linear head 128 -> 147. This design is new (the grid used a CNN decoder on a global latent): check the decoder's
  reconstruction quality on real frames before trusting any decoded readout.

## Losses

Notation: `x_t` a frame, `z_t = E(x_t)`, `p = P(z_{t-1}, a)` the predicted next latent (history 1), `sg` stop-gradient. A patch error `e_i(y, x)` is the mean squared pixel
error over the 7x7x3 pixels of patch `i`. All pixel errors are plain (no class weights).

**Stage 1 (availability).** Make every learnable content available, let the predictor learn, prune nothing.

    L1 = lambda_rec * sum_i w_i * e_i(D(z_t), x_t)       summed over every frame of the clip (context and target)
       + || p - sg(z_next) ||^2                          next-latent MSE, stop-gradient target

- **Error weighting** (label-free): `w_i = (e_i / mean_j e_j)^1`, with a stop-gradient, over the batch's patches, normalised to mean 1, capped at 20. Badly reconstructed
  patches count more, so small content that a plain MSE neglects is pulled in, per sample, with no assumption about where it sits. It also rewards irreducible noise;
  stage 2 removes that.
- **`lambda_rec`:** 0.3 relative to the prediction loss in the grid. Loss scales differ in Craftax: calibrate so that the reconstruction's gradient on the encoder is of
  the order of the prediction's, then read the availability signals. In the grid 1, 0.3 and 0.1 all held the content; at 0.1 the mob started to suffer in stage 2.
- **Not in stage 1:** SIGReg, the IDM, and a target gradient (a weight-1 target gradient in the parent gave no gain in the grid: early on the predictor is bad, and the
  pull toward its output is the race again).

**Stage 2 (consolidation).** Fork from a stage 1 checkpoint: fresh optimiser state, same learning rate, **instant switch** (no ramp).

    L2 = sum_i e_i(D(p), x_next)                                              decoded prediction
       + sum_i exp(-sg(e_i(D(p), x_next)) / tau) * e_i(D(z_next), x_next)      prediction-gated reconstruction
       + || p - sg(z_next) ||^2                                               next-latent MSE

- **Weights:** 1, 1, 1. The all-frame reconstruction is gone.
- **The two decoded terms decode different latents.** The decoded prediction decodes the *predicted* latent `p`; its gradient trains the predictor, the context encoder
  and the decoder with a pixel signal. The gated reconstruction decodes the *real* latent `z_next`; its gradient reaches the encoder on the target frame. The gate's weights
  come from the decoded prediction's error per patch (stop-gradient): a patch the predictor foresees is reconstructed, a patch it cannot foresee (noise) costs nothing.
- **`tau`, scale-free:** `tau = 0.03 * mean per-pixel variance of the training frames` (the grid's 0.005 at a pixel variance of 0.154). Compute the variance once on a sample.
  It is the detail level: the per-pixel error a patch's prediction must beat to be held.

## Why each piece (grid evidence, fresh decoder on the real latent, seed 0)

**Stage 1 decisions** (each tested against the previous best):

| decision | evidence |
|---|---|
| SIGReg off | stage 2 forks (ground-truth weights) with against without SIGReg: held mob 0.39 -> 0.93, terrain 0.85 -> 0.98, all else equal; the parent without SIGReg held the mob at 0.98 (0.54 with) |
| no class labels | error weighting matched the ground-truth class weights within 0.03 everywhere except the noise (twinkle 0.08 against 0.00 after stage 2); plain MSE held the content in stage 1 too but lost the facing in stage 2 (0.66 against 0.99) |
| no target gradient in stage 1 | no gain in the parent or its fork (predicted hand fell from 0.16 to -0.05 in the parent) |
| `lambda_rec` 0.3 | 0.3 and 0.1 both held facing, hand, position; at 0.1 the stage 2 mob fell to 0.72 (0.88 at 0.3) and its prediction to 0.09 (0.25) |

**Stage 2 ablation** (stage S: from the final parent; the reference had a 4000-step ramp and a target gradient):

| stage 2 | facing | agent | hand | mob | twinkle (noise) | predicted facing / hand / mob | rank |
|---|---|---|---|---|---|---|---|
| reference (ramp, decoded prediction, gate, target gradient) | 0.98 | 0.97 | 0.99 | 0.88 | 0.01 | 0.99 / 0.52 / 0.25 | 101 |
| the same, run again | 1.00 | 0.98 | 0.99 | 0.90 | 0.06 | 0.98 / 0.50 / 0.23 | 100 |
| without the ramp (instant switch) | 1.00 | 0.99 | 0.99 | 0.86 | 0.01 | 0.99 / 0.52 / 0.28 | 100 |
| without the target gradient | 0.98 | 0.97 | 0.99 | 0.88 | 0.04 | 0.98 / 0.52 / 0.22 | 99 |
| without the gate | **0.72** | 0.83 | 0.99 | 0.91 | 0.00 | 0.99 / 0.52 / 0.45 | 99 |
| without the decoded prediction | **0.43** | 0.76 | 0.88 | 0.77 | **0.95** | 0.26 / -0.84 / -0.30 | **47** |
| reconstruction fade only (no decoded term) | 0.42 | 0.76 | **0.29** | **0.08** | 0.00 | 0.26 / -0.61 / 0.02 | 66 |
| **the final recipe (instant switch, decoded prediction + gate)** | **1.00** | **0.99** | 0.99 | 0.85 | **-0.01** | 0.99 / 0.52 / 0.29 | 98 |

The parent (end of stage 1) for comparison: facing 0.99, agent 0.98, hand 0.99, mob 0.95, twinkle 0.97 (all noise held), predicted facing / hand / mob 0.44 / 0.09 / 0.18, rank 91.

- **The decoded prediction is essential:** without it the latent loses facing and hand, keeps the noise, and its rank halves; the gate's weights also become meaningless.
- **The gate is needed** because of the **action bypass**: the predictor gets the actions, so it can draw action-determined content (the agent's facing) without the latent
  holding it, and the decoded prediction alone is then satisfied while the latent drops it. A shuffled-action test confirmed it: the predicted facing followed whatever action
  was fed; with the gate (or a target gradient) the predictor read the facing from the latent, without them from the action history. The gate costs a little mob prediction (0.25 against 0.45 without it).
- **The target gradient and the ramp are not needed** once SIGReg is off. With SIGReg on, the gate alone left part of the noise (0.20) and the target gradient completed the
  release; the ramp did not matter there either (an instant switch matched it).
- **Stage 2 needs a reconstruction-like term on the real latent:** the fade alone, with only the latent loss left and no SIGReg, erodes everything.

## Training setup

- **Optimiser:** AdamW, lr 5e-5, weight decay 1e-3, batch 128, bf16, gradient clip 1.0 (B1's values).
- **Learning rate: constant in both stages**, after a short warmup in stage 1; stage 2 starts with a fresh optimiser state at the same rate. B1's cosine schedule ends at
  `trainer.max_epochs` and cannot be extended; a constant rate is what lets either stage be **extended by resuming** at any time, and lets a stage 2 fork start from any stage 1 checkpoint.
  Nothing about the rate was tuned in the grid (it ran at a constant rate throughout). A short decay at the very end of stage 2, to polish the weights, is **untested**: if tried,
  make it a separate final step after the stop signal below says the model has settled, so that it does not interfere with the resume logic.
- **Stage 1: run, validate, extend.** Run a first stage 1, read it with the metrics below, and if the state probes of the mobs, ores or the faced tile are still rising,
  **resume and add epochs**; do not restart. **Save a checkpoint every epoch** (`save_every_n_epochs: 1`, as in B1); these are also the checkpoints stage 2 can be forked from.
- **Switch (end of stage 1), "stage 1 is ready":** at two consecutive epoch checkpoints, (1) the MLP state probe of every entity the frame shows has stopped rising (change inside
  its bootstrap interval, or below about 0.01); (2) the held-out reconstruction, its mean and its worst percentile per patch, is flat; (3) the rank has not collapsed and the
  train/validation gap of the reconstruction and next-latent losses is not growing; (4) softly, the predicted-state probe beats copying the last latent on the entities that change
  with the action. Starting guess from B1: its probe scores settled by epoch 2 to 3. Content that is not available at the switch does not come back in stage 2.
  **To validate in Craftax:** fork stage 2 from two or three stage 1 checkpoints; the earliest fork that is as good as the latest is the switch, and the proxy above should pick it.
- **Stage 2: run, validate, extend, the same way.** **Save a checkpoint every epoch** and read the metrics below at each. **Extend by resuming while the state probes of the facing
  or the faced tile are still rising**, because in the grid the action-determined content (the facing) **dipped and only came back late**: its sensitivity in the latent (how far a
  turn moves `z`, relative to an unrelated frame) went 0.05, 0.04, 0.02, 0.02, 0.07, 0.11 at 2k ... 12k steps, still rising at the end. Stopping at 6k to 8k would have kept the dip.
  No label-free signal caught that recovery, so the label-free signals are guards, not the stop rule; with epoch checkpoints (about 15.5k steps) one flat reading is not enough,
  and a dip inside one epoch will not be seen. A new stage 2 run is needed only when the recipe changes, not when the length was too short. A starting length from the grid:
  about 0.6 of stage 1. **Stage 2 is ready** when, at two consecutive epoch checkpoints, the state probes (fresh decoder and probe) of the facing, the faced tile and the mobs have
  stopped rising, the noise sensitivity is low and flat, and the rank is flat.

## Metrics: the clean set (it replaces AAC's old big probe suite)

AAC's large entity-by-entity probe suite (`probe_suite.py`: the chain pixels -> tokens -> CLS -> latent, sprite swaps, many concept probes) is **not used** for this model: it is slow
and answers more questions than the two stages ask. The following set replaces it. AAC's cheap **health monitor** stays (it already logs the rank, the latent change per step, the
open-loop drift and the cost geometry (latent distance against the time gap) on held-out clips).

**The compact state probe.** One linear head and one MLP head, fitted to the whole state vector from the `state` column on the **training worlds** and scored on the **validation
worlds**, with a fixed step budget and seed at every epoch so that epochs compare. Targets: inventory (12), vitals (4), facing, daylight, position (2), the mobs in view (4), the
7x9 view map (block type and mobs per tile; it holds the ores), and the faced tile. Report accuracy or R2 with the majority baseline and a bootstrap interval over episodes. The
labels are used only to read, never for training the world model. The reference for reading a score is B1's published probe values (no per-epoch ceiling is computed). The
readiness rules above use the MLP probe (the grid's linear probes under-read compressed latents badly: facing linear 0.02, decoder 0.95); report the **linear / MLP ratio** as a
metric-quality number for CEM, not as a gate.

| when | metric | labels? | use |
|---|---|---|---|
| training, both stages | every loss and its **validation loss** (reconstruction, next-latent error and its ratio to copy-last, and in stage 2 the decoded prediction and the gated reconstruction); the train/validation gap | no | health, overfitting; the held-out reconstruction is the pixel readout in stage 1 |
| training, both stages | the health monitor (every 2000 steps) | no | collapse and spikes caught inside an epoch |
| every epoch, both stages | the compact state probe (linear + MLP) on the latent | yes, only to read | **stage 1 readiness, stage 2 stop rule** |
| every epoch, both stages | held-out reconstruction: the mean and the 99th percentile of the per-patch error (the own decoder) | no | stage 1: is small content available; stage 2: the noise patches are no longer reconstructed |
| every epoch, both stages | effective rank | no | **collapse guard** (SIGReg is off): ~93-98 in a healthy grid run, 47 when the decoded prediction was missing |
| every epoch, both stages | **predicted-state probe**: the same probe applied to the predicted latent, scored only where the entity changed, against copying the last latent (AAC's rollout probe does this) | yes, only to read | what the predictor already foresees, per entity |
| every epoch, both stages | **decoded prediction**: the decoder applied to the predicted latent, the error per patch against the real next frame, and the share of patches below `tau` | no | what the predictor foresees (the detail level); the gate's own quantity |
| stage 2, every epoch | a **fresh decoder** to pixels and a fresh state probe on the latent (fixed budget and seed, the same pixel weighting for every model) | yes for the state | the model's own decoder can stop reading real latents once it trains on predictions; there is **no fresh decoder in stage 1** (the own decoder is trained on real latents, so its held-out loss says the same for free) |
| stage 2, every epoch | **action sensitivity**: replace the action and see how far the predicted latent and frame follow it, per entity; AAC's predictor against shuffled actions (action lift) | no for the lift | the action bypass: content drawn from the action alone is not held in the latent |
| stage 2, every epoch | **latent sensitivity**: how far the latent moves for a change of one state factor (the facing, a mob, a tile, an item) relative to an unrelated frame; counterfactual frames from an edited state in the simulator, or AAC's pasted-sprite probe | yes, only to read | the metric CEM sees; the facing's dip and recovery |
| stage 2, every epoch | **noise sensitivity**: the same state rendered twice with different noise, and how far the latent moves | no | the release check, the grid's twinkle test. Craftax has real noise: AAC's change analysis puts 56 % of the aligned frame-difference energy on rest and night noise. **To check:** whether the renderer can draw the same state with a different noise draw |

**Uninformative, do not steer by them:** the mean per-patch error of the decoded prediction (flat from the first checkpoint: easy patches dominate it) and the unpredictable
share `||z_next - p|| / ||z_next - z_last||` (flat at 0.5 to 0.6). Raw next-latent MSE is no progress measure either: slow content dominates it, and without SIGReg it can be
lowered by shrinking the latent.

## Pitfalls the grid study ran into

- **Probes under-read** compressed latents (the facing: MLP probe 0.1, decoder 0.95). Judge content with a decoder trained on the frozen latent, with the same pixel weighting for every model.
- **The model's own decoder can stop reading real latents** when it is trained mostly on predictions. Check content with a fresh decoder; use the own decoder for the predicted frame.
- **Low amplitude is not absence.** A decoder with free gain reads content held at a tiny variance; noise can hide that way, and the planner's MSE barely sees such content.
- **Without SIGReg a pure latent loss can collapse** (a constant latent predicts itself). In stage 2 the two decoded terms hold the latent; watch the rank in every run.
- **Unpredictable is not irrelevant.** A mob's random step cannot be predicted but becomes state. In the grid the final model holds the mob (0.85) but predicts it poorly
  (0.29); a stochastic entity is held as state, not foreseen.
- **The latent metric is uneven:** in the grid a one-cell move changed the latent about 6 times more than a turn or a hand change, so a CEM planner on raw MSE weighs position
  first. Stage 2 made the metric cleaner (noise edits 12x below a turn), but not even. Check per factor with P1; a learned cost or a metric head is the fix if it fails.

## First runs in AAC, in order

1. **Stage 1 alone:** SIGReg off, IDM off, the decoder from `z` with error-weighted reconstruction, stop-gradient target, constant learning rate, a checkpoint every epoch.
   The clean metric set at every epoch (stage 1 rows), the state probe against B1's probe values and A5 at matching steps. The question: are mobs, ores and the faced tile held now,
   above B1 and near the untrained encoder? If they are still rising, resume and add epochs. Watch the rank and the held-out reconstruction.
2. **Stage 2 forked from it** (instant switch; decoded prediction and gated reconstruction; `tau` from the pixel variance; constant rate, fresh optimiser). The clean metric set at
   every epoch, the stage 2 rows included (fresh decoder, action, latent and noise sensitivity), and P1 against B1 at the end. If the facing or the faced tile is still rising at
   the end, resume stage 2 and extend.
3. **Then the close decisions:** the switch time (forks from several stage 1 checkpoints, against the readiness proxy), `lambda_rec` (0.1, 0.3, 1), `tau` (a sweep, with P1),
   plain against error-weighted reconstruction, the MLP predictor against the transformer, a short end-of-stage-2 decay, a blurred decoded-prediction target (a data-pipeline option
   that limits the pixel detail the predictor is asked to foresee), and seeds for the decisions that hinge on small margins.
