# BC entities, chunk 16

A model reads the state of one frame and predicts the expert's next 16 actions. 103275 validation samples of 128 episodes (the final expert's, held-out worlds). Cross-entropy 1.531 nats against 2.128 for the marginal distribution of the target: the model gains 0.597 nats ('skill'). Accuracy per step: 0.67, 0.57, 0.52, 0.50, 0.48, 0.47, 0.45, 0.44, 0.44, 0.43, 0.42, 0.42, 0.42, 0.41, 0.40, 0.40. Skill lost: the rise of the cross-entropy when the entity's columns come from another frame, in shares of the skill (permutation: reliance, not necessity); interval: episode bootstrap. Constant: the columns never vary.

## Coarse parts

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| sleeping | 1.202 | 1.16 to 1.24 | 0.7180 | -0.1250 | 1.8752 | 0.3615 |
| view | 0.674 | 0.66 to 0.70 | 0.4024 | -0.1043 | 1.1860 | 0.2338 |
| vitals | 0.440 | 0.42 to 0.46 | 0.2627 | -0.0501 | 0.2406 | 0.3085 |
| inventory | 0.391 | 0.36 to 0.42 | 0.2335 | -0.0459 | 0.3505 | 0.1734 |
| facing | 0.273 | 0.26 to 0.29 | 0.1629 | -0.0699 | 0.6025 | 0.0475 |
| daylight | 0.034 | 0.03 to 0.04 | 0.0202 | -0.0042 | 0.0308 | 0.0182 |

## Block types in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| grass | 0.463 | 0.43 to 0.49 | 0.2766 | -0.0584 | 0.3843 | 0.2262 |
| water | 0.225 | 0.21 to 0.24 | 0.1342 | -0.0335 | 0.2720 | 0.1023 |
| stone | 0.205 | 0.19 to 0.22 | 0.1222 | -0.0309 | 0.3089 | 0.0923 |
| path | 0.134 | 0.12 to 0.15 | 0.0800 | -0.0203 | 0.1584 | 0.0607 |
| sand | 0.117 | 0.11 to 0.13 | 0.0698 | -0.0172 | 0.1198 | 0.0538 |
| out_of_bounds | 0.037 | 0.03 to 0.04 | 0.0222 | -0.0063 | 0.0398 | 0.0159 |
| crafting_table | 0.030 | 0.03 to 0.03 | 0.0181 | -0.0033 | 0.0564 | 0.0088 |
| tree | 0.028 | 0.03 to 0.03 | 0.0169 | -0.0049 | 0.0909 | 0.0072 |
| diamond | 0.012 | 0.01 to 0.02 | 0.0073 | -0.0016 | 0.0163 | 0.0049 |
| furnace | 0.010 | 0.01 to 0.01 | 0.0062 | -0.0017 | 0.0227 | 0.0035 |
| lava | 0.009 | 0.01 to 0.01 | 0.0055 | -0.0014 | 0.0153 | 0.0029 |
| iron | 0.007 | 0.01 to 0.01 | 0.0040 | -0.0010 | 0.0160 | 0.0014 |
| coal | 0.007 | 0.01 to 0.01 | 0.0040 | -0.0011 | 0.0162 | 0.0019 |
| plant | 0.006 | 0.00 to 0.01 | 0.0033 | -0.0008 | 0.0080 | 0.0015 |
| ripe_plant | 0.001 | 0.00 to 0.00 | 0.0005 | -0.0001 | 0.0019 | 0.0002 |
| wood (constant) | 0.000 | 0.00 to 0.00 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |

## Mobs in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| zombie | 0.115 | 0.11 to 0.12 | 0.0686 | -0.0172 | 0.1555 | 0.0477 |
| skeleton | 0.011 | 0.01 to 0.01 | 0.0067 | -0.0020 | 0.0313 | 0.0029 |
| cow | 0.009 | 0.01 to 0.01 | 0.0051 | -0.0014 | 0.0373 | 0.0013 |
| arrow | 0.007 | 0.01 to 0.01 | 0.0042 | -0.0005 | 0.0076 | 0.0035 |

## Distance from the player (all channels)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| 1 | 0.237 | 0.23 to 0.25 | 0.1418 | -0.0400 | 0.8372 | 0.0615 |
| 3 | 0.161 | 0.15 to 0.17 | 0.0962 | -0.0301 | 0.1270 | 0.0694 |
| 2 | 0.123 | 0.12 to 0.13 | 0.0737 | -0.0224 | 0.1244 | 0.0475 |
| 4 | 0.078 | 0.07 to 0.08 | 0.0466 | -0.0124 | 0.0854 | 0.0279 |
| 0 | 0.002 | 0.00 to 0.00 | 0.0013 | -0.0003 | 0.0018 | 0.0012 |

## Inventory

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| iron_sword | 0.119 | 0.11 to 0.13 | 0.0710 | -0.0132 | 0.1073 | 0.0499 |
| diamond | 0.078 | 0.07 to 0.09 | 0.0467 | -0.0118 | 0.0561 | 0.0409 |
| stone | 0.014 | 0.01 to 0.02 | 0.0084 | -0.0018 | 0.0114 | 0.0069 |
| wood | 0.013 | 0.01 to 0.02 | 0.0079 | -0.0011 | 0.0188 | 0.0038 |
| iron | 0.011 | 0.01 to 0.01 | 0.0067 | -0.0014 | 0.0101 | 0.0050 |
| iron_pickaxe | 0.011 | 0.01 to 0.01 | 0.0065 | -0.0017 | 0.0111 | 0.0045 |
| coal | 0.008 | 0.01 to 0.01 | 0.0050 | -0.0016 | 0.0064 | 0.0040 |
| stone_sword | 0.003 | 0.00 to 0.00 | 0.0019 | -0.0003 | 0.0027 | 0.0014 |
| stone_pickaxe | 0.003 | 0.00 to 0.00 | 0.0016 | -0.0002 | 0.0027 | 0.0011 |
| wood_pickaxe | 0.001 | 0.00 to 0.00 | 0.0009 | -0.0002 | 0.0012 | 0.0007 |
| sapling | 0.001 | 0.00 to 0.00 | 0.0009 | -0.0004 | 0.0018 | 0.0005 |
| wood_sword | 0.001 | 0.00 to 0.00 | 0.0007 | -0.0001 | 0.0004 | 0.0007 |

## Vitals

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| energy | 0.374 | 0.36 to 0.39 | 0.2233 | -0.0474 | 0.1072 | 0.2982 |
| drink | 0.155 | 0.14 to 0.17 | 0.0923 | -0.0178 | 0.1338 | 0.0774 |
| health | 0.023 | 0.02 to 0.03 | 0.0138 | -0.0034 | 0.0192 | 0.0131 |
| food | 0.023 | 0.02 to 0.03 | 0.0137 | -0.0036 | 0.0199 | 0.0106 |

## Facing, daylight, sleeping

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| sleeping | 1.202 | 1.16 to 1.24 | 0.7180 | -0.1250 | 1.8752 | 0.3615 |
| facing | 0.273 | 0.26 to 0.29 | 0.1629 | -0.0699 | 0.6025 | 0.0475 |
| daylight | 0.034 | 0.03 to 0.04 | 0.0202 | -0.0042 | 0.0308 | 0.0182 |

## Skill lost by place (the player stands at the centre, up is up)

| | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| -3 | 0.004 | 0.003 | 0.004 | 0.004 | 0.007 | 0.004 | 0.003 | 0.005 | 0.005 |
| -2 | 0.002 | 0.004 | 0.005 | 0.005 | 0.009 | 0.004 | 0.004 | 0.004 | 0.002 |
| -1 | 0.003 | 0.003 | 0.006 | 0.013 | 0.050 | 0.012 | 0.004 | 0.003 | 0.002 |
| +0 | 0.005 | 0.005 | 0.008 | 0.034 | 0.002 | 0.035 | 0.007 | 0.004 | 0.003 |
| +1 | 0.003 | 0.004 | 0.006 | 0.014 | 0.045 | 0.013 | 0.005 | 0.003 | 0.002 |
| +2 | 0.002 | 0.003 | 0.005 | 0.005 | 0.010 | 0.005 | 0.005 | 0.003 | 0.002 |
| +3 | 0.005 | 0.004 | 0.003 | 0.004 | 0.007 | 0.004 | 0.006 | 0.004 | 0.004 |

## Trained again without the entity

Skill lost: the rise of the validation cross-entropy over the model with everything, in shares of its skill (one seed, no interval).

| entity | skill lost | permutation | ce | accuracy |
|---|---|---|---|---|
| group:sleeping | 0.461 | 1.202 | 1.8060 | 0.3779 |
| group:view | 0.236 | 0.674 | 1.6712 | 0.4135 |
| group:vitals | 0.125 | 0.440 | 1.6054 | 0.4503 |
| vitals:energy | 0.095 | 0.374 | 1.5874 | 0.4536 |
| group:facing | 0.093 | 0.273 | 1.5862 | 0.4309 |
| group:inventory | 0.078 | 0.391 | 1.5769 | 0.4538 |
| mob:zombie | 0.044 | 0.115 | 1.5565 | 0.4584 |
| inventory:diamond | 0.029 | 0.078 | 1.5479 | 0.4598 |
| inventory:iron_sword | 0.022 | 0.119 | 1.5436 | 0.4622 |
| vitals:drink | 0.019 | 0.155 | 1.5418 | 0.4608 |
| group:daylight | 0.014 | 0.034 | 1.5387 | 0.4619 |
| vitals:health | 0.012 | 0.023 | 1.5377 | 0.4624 |
