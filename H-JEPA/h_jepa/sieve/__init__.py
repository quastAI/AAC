"""Sieve: the two-stage world model of `aac_recipe.md` (README id V4), its losses and its metrics.

Stage 1 (availability): the encoder's latent is decoded back into the frame (each patch's error weighted by its own
size, no labels) while the predictor learns the next latent against a stop-gradient target; no SIGReg, no IDM.
Stage 2 (consolidation), forked from a stage-1 checkpoint with a fresh optimiser: the predicted latent is decoded and
compared with the real next frame (decoded prediction), and the real next frame is reconstructed only where that
decoded prediction was right (prediction-gated reconstruction). The latent keeps what the predictor foresees.
One stage (`level1.sieve.stage: joint`, README id V5, config `craftax_sieve_joint`): both in one training from the start,
the next frame reconstructed under a relative, floored gate on the decoded difference (about uniform early, selective as
the predictor learns; `losses.py`).

Training (inside the usual `main_hjepa.py`, config `craftax_sieve`):
    decoder    `PatchDecoder`: latent -> 81 patches of 7x7x3 (one per tile and inventory cell), a level-1 submodule
    losses     the terms of either stage or of the one-stage mode, called from `hjepa_utils.hjepa_forward`

Metrics (one worker next to the training reads every epoch checkpoint; `python -m sieve.watch`):
    sample           the fixed frames the metrics read: training worlds (probes, fresh decoder) and validation worlds (scores)
    counterfactuals  frames re-rendered from stored simulator states after one edit (facing, a mob, a tile, an item) or
                     with another noise draw: what moves the latent
    probe            the compact state probe (one linear and one MLP head on the whole state), labels only to read
    check            every metric of one checkpoint
    watch            the worker: evaluates each new epoch checkpoint of a run, then writes the report
    report           the table over epochs and the readiness rules of the recipe (when to fork stage 2, when to stop)
    gate_analysis    what a gate keeps per patch (abs, rel, diff and the one-stage gate), grouped by what the patch shows
    concentration    how many latent axes a factor needs (standard, PCA and random bases)
"""
