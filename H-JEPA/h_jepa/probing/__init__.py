"""The probe suite: after training, what the flat model keeps of each entity of the game, where it loses it, and why.

Every entity (block types, mobs, inventory items, vitals, facing, sleeping, daylight, achievements) is followed along
one chain, and its verdict is the first link that fails:

    0 pixels       the tile's 7x7 pixels (the inventory strip's for items and vitals): the ceiling
    1 ViT tokens   its patch token after some layers and after the last; the untrained ViT's token is the control
    2 CLS          the ViT's CLS token before the projector
    3 latent       the latent (linear and MLP readouts; the untrained latent is the control)
    4 swap         a sprite pasted into a real frame: does the latent move, does its map decoder read the paste?
    5 one step     the predicted latent after an event (collect, place, make) or a mob's move
    6 rollout      open-loop predictions, scored only where the entity changed (the change test)

Why a stage loses a block type: its rarity against how much its sprite looks like another one, where in the latent's
variance it sits (the spectrum), and how much of the prediction loss its transitions carry (loss attribution). Whether
capacity or the objective drops it needs a training run (an auxiliary map decoder on the latent), not a probe.

Modules: `targets` (what is read), `readout` (probes, scores, episode bootstrap), `collect` (one pass over the clips),
`encoder` (block A), `predictor` (block B), `verdict`, `report`. The entry point is `probe_suite.py`.
"""
