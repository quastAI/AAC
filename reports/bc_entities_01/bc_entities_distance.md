# Temporal distance entities

A model reads the state of frame t and of the frame k later (k from 1 to 64 steps, 12 classes, one k per frame) and predicts k; an entity comes from another random pair, in both frames. Daylight is left out: the light level is a clock. 102985 validation samples of 128 episodes (the final expert's, held-out worlds). Cross-entropy 1.541 nats against 2.485 for the marginal distribution of the target: the model gains 0.944 nats ('skill'). Accuracy per step: 0.37. Skill lost: the rise of the cross-entropy when the entity's columns come from another frame, in shares of the skill (permutation: reliance, not necessity); interval: episode bootstrap. Constant: the columns never vary.

## Coarse parts

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| view | 4.541 | 4.48 to 4.61 | 4.2875 | -0.1844 |
| vitals | 1.921 | 1.89 to 1.96 | 1.8135 | -0.1638 |
| sleeping | 0.492 | 0.47 to 0.51 | 0.4648 | -0.0734 |
| facing | 0.169 | 0.16 to 0.18 | 0.1597 | -0.0378 |
| inventory | 0.096 | 0.09 to 0.10 | 0.0902 | -0.0243 |

## Block types in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| grass | 1.268 | 1.23 to 1.31 | 1.1970 | -0.1156 |
| path | 0.598 | 0.58 to 0.62 | 0.5649 | -0.0651 |
| water | 0.520 | 0.50 to 0.54 | 0.4905 | -0.0616 |
| stone | 0.493 | 0.48 to 0.51 | 0.4651 | -0.0636 |
| sand | 0.454 | 0.44 to 0.47 | 0.4282 | -0.0586 |
| tree | 0.093 | 0.09 to 0.10 | 0.0879 | -0.0199 |
| furnace | 0.026 | 0.02 to 0.03 | 0.0248 | -0.0070 |
| lava | 0.022 | 0.02 to 0.02 | 0.0210 | -0.0050 |
| crafting_table | 0.019 | 0.02 to 0.02 | 0.0180 | -0.0063 |
| out_of_bounds | 0.018 | 0.01 to 0.02 | 0.0169 | -0.0038 |
| plant | 0.005 | 0.00 to 0.01 | 0.0051 | -0.0015 |
| coal | 0.004 | 0.00 to 0.01 | 0.0033 | -0.0011 |
| iron | 0.002 | 0.00 to 0.00 | 0.0019 | -0.0008 |
| diamond | 0.001 | 0.00 to 0.00 | 0.0013 | -0.0002 |
| wood (constant) | 0.000 | 0.00 to 0.00 | 0.0000 | +0.0000 |
| ripe_plant | -0.000 | -0.00 to 0.00 | -0.0000 | -0.0000 |

## Mobs in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| zombie | 0.030 | 0.03 to 0.03 | 0.0286 | -0.0074 |
| cow | 0.010 | 0.01 to 0.01 | 0.0093 | -0.0025 |
| skeleton | 0.005 | 0.00 to 0.01 | 0.0045 | -0.0014 |
| arrow | 0.002 | 0.00 to 0.00 | 0.0019 | -0.0008 |

## Distance from the player (all channels)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| 3 | 0.754 | 0.74 to 0.77 | 0.7123 | -0.0906 |
| 2 | 0.479 | 0.47 to 0.49 | 0.4521 | -0.0694 |
| 1 | 0.368 | 0.36 to 0.38 | 0.3472 | -0.0628 |
| 4 | 0.318 | 0.31 to 0.33 | 0.3003 | -0.0567 |
| 0 | 0.010 | 0.01 to 0.01 | 0.0097 | -0.0049 |

## Inventory

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| iron_sword | 0.019 | 0.02 to 0.02 | 0.0176 | -0.0064 |
| diamond | 0.016 | 0.01 to 0.02 | 0.0149 | -0.0032 |
| stone | 0.015 | 0.01 to 0.02 | 0.0145 | -0.0049 |
| iron_pickaxe | 0.011 | 0.01 to 0.01 | 0.0103 | -0.0033 |
| stone_sword | 0.007 | 0.01 to 0.01 | 0.0067 | -0.0016 |
| stone_pickaxe | 0.007 | 0.01 to 0.01 | 0.0063 | -0.0012 |
| wood | 0.005 | 0.00 to 0.01 | 0.0045 | -0.0015 |
| coal | 0.003 | 0.00 to 0.00 | 0.0032 | -0.0017 |
| wood_sword | 0.002 | 0.00 to 0.00 | 0.0017 | -0.0008 |
| wood_pickaxe | 0.002 | 0.00 to 0.00 | 0.0017 | -0.0010 |
| iron | 0.002 | 0.00 to 0.00 | 0.0015 | -0.0010 |
| sapling | 0.001 | 0.00 to 0.00 | 0.0011 | -0.0004 |

## Vitals

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| energy | 1.557 | 1.52 to 1.60 | 1.4702 | -0.1370 |
| drink | 0.130 | 0.12 to 0.14 | 0.1230 | -0.0275 |
| food | 0.019 | 0.02 to 0.02 | 0.0183 | -0.0045 |
| health | 0.007 | 0.01 to 0.01 | 0.0071 | -0.0018 |

## Facing, daylight, sleeping

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| sleeping | 0.492 | 0.47 to 0.51 | 0.4648 | -0.0734 |
| facing | 0.169 | 0.16 to 0.18 | 0.1597 | -0.0378 |

## Skill lost by place (the player stands at the centre, up is up)

| | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| -3 | 0.006 | 0.003 | 0.004 | 0.007 | 0.007 | 0.005 | 0.005 | 0.004 | 0.006 |
| -2 | 0.002 | 0.003 | 0.006 | 0.009 | 0.015 | 0.009 | 0.005 | 0.004 | 0.003 |
| -1 | 0.002 | 0.004 | 0.008 | 0.020 | 0.041 | 0.021 | 0.009 | 0.005 | 0.005 |
| +0 | 0.003 | 0.005 | 0.012 | 0.034 | 0.010 | 0.037 | 0.014 | 0.006 | 0.004 |
| +1 | 0.002 | 0.003 | 0.008 | 0.018 | 0.042 | 0.021 | 0.009 | 0.004 | 0.003 |
| +2 | 0.003 | 0.002 | 0.004 | 0.007 | 0.014 | 0.008 | 0.004 | 0.004 | 0.004 |
| +3 | 0.006 | 0.003 | 0.004 | 0.006 | 0.008 | 0.005 | 0.003 | 0.003 | 0.005 |

## Trained again without the entity

Skill lost: the rise of the validation cross-entropy over the model with everything, in shares of its skill (one seed, no interval).

| entity | skill lost | permutation | ce | accuracy |
|---|---|---|---|---|
| group:vitals | 0.372 | 1.921 | 1.8916 | 0.2719 |
| vitals:energy | 0.228 | 1.557 | 1.7558 | 0.3065 |
| group:view | 0.191 | 4.541 | 1.7212 | 0.3055 |
| vitals:drink | 0.041 | 0.130 | 1.5792 | 0.3568 |
| group:facing | 0.036 | 0.169 | 1.5742 | 0.3534 |
| group:inventory | 0.031 | 0.096 | 1.5702 | 0.3588 |
| group:sleeping | 0.026 | 0.492 | 1.5655 | 0.3642 |
| mob:zombie | 0.008 | 0.030 | 1.5477 | 0.3662 |
| vitals:food | 0.005 | 0.019 | 1.5449 | 0.3658 |
| inventory:iron_sword | 0.003 | 0.019 | 1.5432 | 0.3657 |
| inventory:diamond | 0.002 | 0.016 | 1.5424 | 0.3654 |
