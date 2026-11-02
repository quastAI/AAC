"""The race of the first epochs: what the latent holds at many steps of the training, for pixel and state runs alike.

Hypothesis (2026-11-02): the encoder removes, from the first steps, the variance the predictor cannot predict yet, whatever the
predictor could predict later; stop-gradient on the prediction target (`level1.wm.detach_pred_target`) is the direct test, and an
encoder of the simulator state instead of the pixels (ids A3 and A4) separates the losses from the ViT and its CLS token.

    DenseCheckpoints (utils.py) saves the model at the untrained start and at many steps of the first epochs
    evaluate   reads the latent of every checkpoint on a fixed held-out sample (`readout`) -> <run>/race/trace.json, meta.json
    report     runs side by side: `python -m race.report --out <md> B2=<run dir> V3=<run dir> A3=<run dir> A4=<run dir>`

Run from `H-JEPA/h_jepa` with `PYTHONPATH=..:.`; `experiments/run.py` builds the commands (`trace: true` on a run).
"""
