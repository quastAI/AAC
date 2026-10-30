"""Slowness analysis of the flat model: what the variance that SIGReg asks of the latent is made of, by time scale.

Analysis only, no model setting and no loss. A training batch of F1 holds 128 clips of 128 different episodes (3529 train
episodes), so SIGReg's batch variance is "another episode, another moment". Whether that variance is carried by cheap
slow factors (inventory, daylight, the episode itself) or by fast ones is measured here on held-out worlds, on the F1
checkpoints of every epoch and on an untrained control, with the CLS token before the projector next to the latent.

    sample    a nested sample of the validation file: episodes x windows spread over the episode x consecutive frames
    encode    CLS and latent of every frame, for several checkpoints, stored once (npz) so that the analyses are reruns
    variance  the variance of a feature split by time scale (between episodes, between windows, within a window, within a clip)
    batches   SIGReg of the latents under different batch compositions (iid, E x K, windows), with a null control

Run from `H-JEPA/h_jepa` with `PYTHONPATH=..:.` like the other scripts.
"""
