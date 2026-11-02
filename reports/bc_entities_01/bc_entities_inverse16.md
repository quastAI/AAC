# Multi-step inverse dynamics, 16 steps

A model reads the state of frame t and of frame t+16 and predicts the 16 actions between them, each step as its own distribution (many action sequences lead to the same state: the target is not deterministic); an entity comes from another random pair, in both frames. 103275 validation samples of 128 episodes (the final expert's, held-out worlds). Cross-entropy 1.239 nats against 2.128 for the marginal distribution of the target: the model gains 0.889 nats ('skill'). Accuracy per step: 0.66, 0.56, 0.54, 0.53, 0.52, 0.52, 0.52, 0.52, 0.52, 0.52, 0.53, 0.54, 0.55, 0.56, 0.61, 0.77. Skill lost: the rise of the cross-entropy when the entity's columns come from another frame, in shares of the skill (permutation: reliance, not necessity); interval: episode bootstrap. Constant: the columns never vary.

## Coarse parts

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| view | 0.818 | 0.80 to 0.83 | 0.7268 | -0.1898 | 1.0247 | 0.6256 |
| sleeping | 0.601 | 0.58 to 0.62 | 0.5344 | -0.1115 | 1.0234 | 0.9870 |
| facing | 0.415 | 0.40 to 0.43 | 0.3687 | -0.0960 | 0.3586 | 2.5053 |
| inventory | 0.218 | 0.21 to 0.23 | 0.1941 | -0.0340 | 0.2371 | 0.1911 |
| vitals | 0.144 | 0.14 to 0.15 | 0.1276 | -0.0252 | 0.1275 | 0.1273 |
| daylight | 0.011 | 0.01 to 0.01 | 0.0100 | -0.0023 | 0.0101 | 0.0102 |

## Block types in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| grass | 0.185 | 0.18 to 0.20 | 0.1645 | -0.0426 | 0.2454 | 0.1460 |
| water | 0.124 | 0.12 to 0.13 | 0.1104 | -0.0256 | 0.1839 | 0.0820 |
| path | 0.098 | 0.09 to 0.10 | 0.0869 | -0.0294 | 0.1351 | 0.1143 |
| stone | 0.097 | 0.09 to 0.10 | 0.0859 | -0.0269 | 0.1860 | 0.0997 |
| sand | 0.049 | 0.05 to 0.05 | 0.0432 | -0.0113 | 0.0712 | 0.0390 |
| crafting_table | 0.027 | 0.02 to 0.03 | 0.0240 | -0.0045 | 0.0329 | 0.0443 |
| furnace | 0.022 | 0.02 to 0.02 | 0.0198 | -0.0026 | 0.0204 | 0.0417 |
| tree | 0.019 | 0.02 to 0.02 | 0.0167 | -0.0048 | 0.0586 | 0.0135 |
| out_of_bounds | 0.016 | 0.01 to 0.02 | 0.0141 | -0.0042 | 0.0215 | 0.0059 |
| plant | 0.008 | 0.01 to 0.01 | 0.0071 | -0.0021 | 0.0083 | 0.0096 |
| lava | 0.007 | 0.01 to 0.01 | 0.0066 | -0.0020 | 0.0108 | 0.0037 |
| diamond | 0.007 | 0.01 to 0.01 | 0.0066 | -0.0013 | 0.0112 | 0.0059 |
| coal | 0.004 | 0.00 to 0.01 | 0.0037 | -0.0012 | 0.0111 | 0.0049 |
| iron | 0.003 | 0.00 to 0.00 | 0.0030 | -0.0012 | 0.0097 | 0.0041 |
| ripe_plant | 0.000 | 0.00 to 0.00 | 0.0004 | -0.0001 | 0.0015 | 0.0002 |
| wood (constant) | 0.000 | 0.00 to 0.00 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |

## Mobs in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| zombie | 0.043 | 0.04 to 0.05 | 0.0386 | -0.0097 | 0.0919 | 0.0291 |
| skeleton | 0.008 | 0.01 to 0.01 | 0.0067 | -0.0018 | 0.0226 | 0.0060 |
| cow | 0.006 | 0.00 to 0.01 | 0.0049 | -0.0014 | 0.0240 | 0.0025 |
| arrow | 0.003 | 0.00 to 0.00 | 0.0023 | -0.0005 | 0.0061 | 0.0024 |

## Distance from the player (all channels)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| 3 | 0.186 | 0.18 to 0.19 | 0.1649 | -0.0539 | 0.1367 | 0.0623 |
| 1 | 0.179 | 0.18 to 0.19 | 0.1593 | -0.0470 | 0.5472 | 0.3936 |
| 2 | 0.100 | 0.10 to 0.10 | 0.0888 | -0.0240 | 0.0973 | 0.0725 |
| 4 | 0.093 | 0.09 to 0.10 | 0.0829 | -0.0254 | 0.0853 | 0.0290 |
| 0 | 0.006 | 0.01 to 0.01 | 0.0051 | -0.0014 | 0.0024 | 0.0030 |

## Inventory

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| iron_sword | 0.051 | 0.05 to 0.06 | 0.0456 | -0.0091 | 0.0546 | 0.0413 |
| diamond | 0.023 | 0.02 to 0.03 | 0.0203 | -0.0035 | 0.0202 | 0.0235 |
| wood | 0.014 | 0.01 to 0.01 | 0.0123 | -0.0009 | 0.0194 | 0.0098 |
| stone | 0.011 | 0.01 to 0.01 | 0.0097 | -0.0020 | 0.0123 | 0.0097 |
| iron | 0.007 | 0.01 to 0.01 | 0.0060 | -0.0013 | 0.0064 | 0.0057 |
| coal | 0.005 | 0.00 to 0.01 | 0.0042 | -0.0009 | 0.0040 | 0.0044 |
| iron_pickaxe | 0.004 | 0.00 to 0.00 | 0.0039 | -0.0009 | 0.0054 | 0.0038 |
| stone_pickaxe | 0.002 | 0.00 to 0.00 | 0.0017 | -0.0004 | 0.0026 | 0.0019 |
| stone_sword | 0.002 | 0.00 to 0.00 | 0.0016 | -0.0004 | 0.0022 | 0.0019 |
| sapling | 0.001 | 0.00 to 0.00 | 0.0010 | -0.0002 | 0.0013 | 0.0007 |
| wood_sword | 0.000 | 0.00 to 0.00 | 0.0003 | -0.0001 | 0.0004 | 0.0002 |
| wood_pickaxe | 0.000 | 0.00 to 0.00 | 0.0003 | -0.0000 | 0.0004 | 0.0003 |

## Vitals

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| drink | 0.055 | 0.05 to 0.06 | 0.0488 | -0.0113 | 0.0533 | 0.0468 |
| energy | 0.052 | 0.05 to 0.06 | 0.0465 | -0.0081 | 0.0388 | 0.0491 |
| health | 0.012 | 0.01 to 0.01 | 0.0106 | -0.0016 | 0.0104 | 0.0123 |
| food | 0.009 | 0.01 to 0.01 | 0.0079 | -0.0016 | 0.0093 | 0.0087 |

## Facing, daylight, sleeping

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 16 |
|---|---|---|---|---|---|---|
| sleeping | 0.601 | 0.58 to 0.62 | 0.5344 | -0.1115 | 1.0234 | 0.9870 |
| facing | 0.415 | 0.40 to 0.43 | 0.3687 | -0.0960 | 0.3586 | 2.5053 |
| daylight | 0.011 | 0.01 to 0.01 | 0.0100 | -0.0023 | 0.0101 | 0.0102 |

## Skill lost by place (the player stands at the centre, up is up)

| | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| -3 | 0.005 | 0.004 | 0.004 | 0.004 | 0.006 | 0.004 | 0.004 | 0.004 | 0.005 |
| -2 | 0.003 | 0.002 | 0.003 | 0.004 | 0.007 | 0.004 | 0.003 | 0.002 | 0.002 |
| -1 | 0.003 | 0.002 | 0.003 | 0.010 | 0.036 | 0.009 | 0.004 | 0.002 | 0.002 |
| +0 | 0.003 | 0.003 | 0.005 | 0.025 | 0.006 | 0.024 | 0.005 | 0.002 | 0.003 |
| +1 | 0.003 | 0.002 | 0.004 | 0.010 | 0.035 | 0.008 | 0.002 | 0.002 | 0.003 |
| +2 | 0.003 | 0.002 | 0.003 | 0.005 | 0.007 | 0.004 | 0.004 | 0.003 | 0.003 |
| +3 | 0.005 | 0.004 | 0.004 | 0.004 | 0.007 | 0.004 | 0.004 | 0.004 | 0.005 |

## Trained again without the entity

Skill lost: the rise of the validation cross-entropy over the model with everything, in shares of its skill (one seed, no interval).

| entity | skill lost | permutation | ce | accuracy |
|---|---|---|---|---|
| group:view | 0.245 | 0.818 | 1.4562 | 0.4898 |
| group:facing | 0.095 | 0.415 | 1.3234 | 0.5266 |
| group:sleeping | 0.069 | 0.601 | 1.2997 | 0.5440 |
| group:vitals | 0.035 | 0.144 | 1.2696 | 0.5551 |
| group:inventory | 0.033 | 0.218 | 1.2677 | 0.5582 |
| mob:zombie | 0.016 | 0.043 | 1.2533 | 0.5573 |
| vitals:energy | 0.015 | 0.052 | 1.2518 | 0.5584 |
| vitals:drink | 0.010 | 0.055 | 1.2476 | 0.5573 |
| inventory:diamond | 0.005 | 0.023 | 1.2434 | 0.5605 |
| inventory:iron_sword | 0.005 | 0.051 | 1.2433 | 0.5605 |
| group:daylight | 0.005 | 0.011 | 1.2433 | 0.5593 |
| inventory:wood | 0.005 | 0.014 | 1.2430 | 0.5608 |
