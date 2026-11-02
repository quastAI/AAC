"""A3 and A4: the flat baseline's losses on an MLP encoder of the simulator state (README, "Ids").

An ablation, never a model candidate: the encoder reads the `state` column (1345 floats per frame: block type and mobs of
the 63 view tiles, inventory, vitals, facing, daylight, sleeping), the observation the expert acted on, instead of the
pixels. There is no ViT and no CLS token, so nothing has to be pooled from a grid of patches into one vector. Everything
after the encoder is B1's: a projector with BatchNorm into a latent of 192, a predictor that scores every position of a
clip of 4 frames from up to 3 latents before it and the actions, SIGReg (0.09) on the latents of every position, the
one-step IDM (0.1). A4 is A3 with the prediction target detached (stop-gradient): the prediction loss then reaches the
encoder only through the context.

The question (2026-11-02): does the encoder remove variance the predictor cannot predict yet, before the predictor learns
it (a race in the first epoch)? If A3 loses the mobs and ores as B1 does, the losses remove them whatever the encoder;
if A4 keeps them, the target path is the route. If A3 keeps them, the ViT and its CLS token are the more likely cause.
The run is read at many steps of its first epoch (`trace`), from step 0 (the untrained encoder) on.

    data     the state table and the clips (reuses `bc.load_table`)
    model    encoder, predictor, the losses
    trace    what the latent holds at a step, on a fixed sample of held-out episodes (uses `slowness`)
    train    the training with its checks: `python -m state_jepa.train --train-file .. --val-file .. --out <run dir>`
    report   the runs side by side: `python -m state_jepa.report --out <md> A3_s42=<run dir> A4_s42=<run dir>`

Run from `H-JEPA/h_jepa` with `PYTHONPATH=..:.`; `experiments/run.py run A3` and `run A4` build the commands. A laptop CPU
is enough: 29 ms per step at batch 128 with history 3 on an M1 Max (8 threads), one epoch of the training file in about
7.5 minutes plus the checks.
"""
