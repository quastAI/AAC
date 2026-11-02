# Multi-step inverse dynamics, 4 steps

A model reads the state of frame t and of frame t+4 and predicts the 4 actions between them, each step as its own distribution (many action sequences lead to the same state: the target is not deterministic); an entity comes from another random pair, in both frames. 104811 validation samples of 128 episodes (the final expert's, held-out worlds). Cross-entropy 0.732 nats against 2.130 for the marginal distribution of the target: the model gains 1.398 nats ('skill'). Accuracy per step: 0.77, 0.70, 0.70, 0.82. Skill lost: the rise of the cross-entropy when the entity's columns come from another frame, in shares of the skill (permutation: reliance, not necessity); interval: episode bootstrap. Constant: the columns never vary.

## Coarse parts

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| view | 1.137 | 1.12 to 1.15 | 1.5894 | -0.3388 | 2.2813 | 0.9753 |
| sleeping | 0.998 | 0.97 to 1.02 | 1.3949 | -0.2217 | 1.7763 | 1.3159 |
| facing | 0.720 | 0.70 to 0.74 | 1.0071 | -0.1878 | 0.4349 | 2.8464 |
| inventory | 0.181 | 0.17 to 0.19 | 0.2536 | -0.0490 | 0.2811 | 0.2237 |
| vitals | 0.092 | 0.09 to 0.10 | 0.1280 | -0.0278 | 0.1410 | 0.1160 |
| daylight | 0.007 | 0.01 to 0.01 | 0.0094 | -0.0027 | 0.0104 | 0.0094 |

## Block types in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| grass | 0.262 | 0.25 to 0.27 | 0.3669 | -0.1014 | 0.4721 | 0.2229 |
| stone | 0.135 | 0.13 to 0.14 | 0.1887 | -0.0642 | 0.2762 | 0.1450 |
| path | 0.123 | 0.12 to 0.13 | 0.1725 | -0.0654 | 0.1934 | 0.1529 |
| water | 0.107 | 0.10 to 0.11 | 0.1501 | -0.0434 | 0.1975 | 0.0991 |
| sand | 0.063 | 0.06 to 0.07 | 0.0881 | -0.0260 | 0.1124 | 0.0547 |
| tree | 0.041 | 0.04 to 0.04 | 0.0578 | -0.0207 | 0.0963 | 0.0340 |
| crafting_table | 0.038 | 0.04 to 0.04 | 0.0529 | -0.0140 | 0.0580 | 0.0573 |
| furnace | 0.035 | 0.03 to 0.04 | 0.0494 | -0.0110 | 0.0524 | 0.0564 |
| out_of_bounds | 0.010 | 0.01 to 0.01 | 0.0144 | -0.0039 | 0.0213 | 0.0077 |
| diamond | 0.008 | 0.01 to 0.01 | 0.0109 | -0.0024 | 0.0146 | 0.0076 |
| coal | 0.007 | 0.01 to 0.01 | 0.0095 | -0.0035 | 0.0181 | 0.0048 |
| plant | 0.007 | 0.01 to 0.01 | 0.0094 | -0.0019 | 0.0118 | 0.0104 |
| iron | 0.007 | 0.01 to 0.01 | 0.0094 | -0.0037 | 0.0170 | 0.0052 |
| lava | 0.007 | 0.01 to 0.01 | 0.0093 | -0.0025 | 0.0138 | 0.0055 |
| ripe_plant | 0.001 | 0.00 to 0.00 | 0.0014 | -0.0004 | 0.0020 | 0.0008 |
| wood (constant) | 0.000 | 0.00 to 0.00 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |

## Mobs in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| zombie | 0.050 | 0.05 to 0.05 | 0.0697 | -0.0162 | 0.1323 | 0.0401 |
| skeleton | 0.013 | 0.01 to 0.01 | 0.0187 | -0.0043 | 0.0367 | 0.0109 |
| cow | 0.013 | 0.01 to 0.01 | 0.0177 | -0.0051 | 0.0417 | 0.0070 |
| arrow | 0.003 | 0.00 to 0.00 | 0.0041 | -0.0011 | 0.0079 | 0.0027 |

## Distance from the player (all channels)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| 1 | 0.455 | 0.45 to 0.46 | 0.6359 | -0.1766 | 1.0690 | 0.5883 |
| 3 | 0.194 | 0.19 to 0.20 | 0.2709 | -0.0986 | 0.3507 | 0.1187 |
| 2 | 0.162 | 0.16 to 0.17 | 0.2263 | -0.0746 | 0.2382 | 0.1344 |
| 4 | 0.081 | 0.08 to 0.08 | 0.1126 | -0.0371 | 0.1509 | 0.0401 |
| 0 | 0.013 | 0.01 to 0.01 | 0.0184 | -0.0064 | 0.0094 | 0.0067 |

## Inventory

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| iron_sword | 0.038 | 0.03 to 0.04 | 0.0525 | -0.0090 | 0.0583 | 0.0459 |
| wood | 0.017 | 0.02 to 0.02 | 0.0236 | -0.0060 | 0.0220 | 0.0248 |
| diamond | 0.015 | 0.01 to 0.02 | 0.0206 | -0.0041 | 0.0208 | 0.0198 |
| stone | 0.006 | 0.01 to 0.01 | 0.0088 | -0.0021 | 0.0107 | 0.0079 |
| iron_pickaxe | 0.004 | 0.00 to 0.01 | 0.0061 | -0.0016 | 0.0075 | 0.0051 |
| iron | 0.004 | 0.00 to 0.00 | 0.0053 | -0.0013 | 0.0054 | 0.0054 |
| coal | 0.002 | 0.00 to 0.00 | 0.0032 | -0.0011 | 0.0033 | 0.0029 |
| stone_pickaxe | 0.002 | 0.00 to 0.00 | 0.0027 | -0.0008 | 0.0046 | 0.0020 |
| stone_sword | 0.001 | 0.00 to 0.00 | 0.0020 | -0.0007 | 0.0031 | 0.0017 |
| sapling | 0.001 | 0.00 to 0.00 | 0.0011 | -0.0001 | 0.0018 | 0.0008 |
| wood_pickaxe | 0.000 | 0.00 to 0.00 | 0.0004 | -0.0002 | 0.0005 | 0.0003 |
| wood_sword | 0.000 | 0.00 to 0.00 | 0.0002 | -0.0000 | 0.0001 | 0.0002 |

## Vitals

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| drink | 0.038 | 0.04 to 0.04 | 0.0528 | -0.0144 | 0.0596 | 0.0486 |
| energy | 0.027 | 0.02 to 0.03 | 0.0372 | -0.0060 | 0.0363 | 0.0372 |
| health | 0.008 | 0.01 to 0.01 | 0.0118 | -0.0023 | 0.0128 | 0.0109 |
| food | 0.008 | 0.01 to 0.01 | 0.0108 | -0.0024 | 0.0106 | 0.0100 |

## Facing, daylight, sleeping

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| sleeping | 0.998 | 0.97 to 1.02 | 1.3949 | -0.2217 | 1.7763 | 1.3159 |
| facing | 0.720 | 0.70 to 0.74 | 1.0071 | -0.1878 | 0.4349 | 2.8464 |
| daylight | 0.007 | 0.01 to 0.01 | 0.0094 | -0.0027 | 0.0104 | 0.0094 |

## Skill lost by place (the player stands at the centre, up is up)

| | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| -3 | 0.003 | 0.003 | 0.003 | 0.004 | 0.007 | 0.004 | 0.003 | 0.003 | 0.003 |
| -2 | 0.002 | 0.002 | 0.004 | 0.006 | 0.017 | 0.006 | 0.003 | 0.003 | 0.002 |
| -1 | 0.003 | 0.003 | 0.006 | 0.018 | 0.112 | 0.017 | 0.005 | 0.003 | 0.003 |
| +0 | 0.004 | 0.004 | 0.012 | 0.079 | 0.013 | 0.076 | 0.011 | 0.004 | 0.003 |
| +1 | 0.003 | 0.003 | 0.006 | 0.018 | 0.104 | 0.016 | 0.005 | 0.003 | 0.003 |
| +2 | 0.002 | 0.002 | 0.004 | 0.006 | 0.017 | 0.005 | 0.004 | 0.003 | 0.002 |
| +3 | 0.003 | 0.003 | 0.003 | 0.004 | 0.007 | 0.004 | 0.003 | 0.003 | 0.003 |

## Trained again without the entity

Skill lost: the rise of the validation cross-entropy over the model with everything, in shares of its skill (one seed, no interval).

| entity | skill lost | permutation | ce | accuracy |
|---|---|---|---|---|
| group:view | 0.310 | 1.137 | 1.1652 | 0.5871 |
| group:facing | 0.089 | 0.720 | 0.8568 | 0.7028 |
| group:sleeping | 0.086 | 0.998 | 0.8521 | 0.7185 |
| group:vitals | 0.023 | 0.092 | 0.7641 | 0.7416 |
| mob:zombie | 0.018 | 0.050 | 0.7570 | 0.7438 |
| group:inventory | 0.016 | 0.181 | 0.7541 | 0.7473 |
| vitals:energy | 0.011 | 0.027 | 0.7483 | 0.7460 |
| vitals:drink | 0.009 | 0.038 | 0.7451 | 0.7446 |
| inventory:wood | 0.005 | 0.017 | 0.7389 | 0.7477 |
| group:daylight | 0.003 | 0.007 | 0.7370 | 0.7482 |
| inventory:iron_sword | 0.002 | 0.038 | 0.7357 | 0.7499 |
| inventory:diamond | 0.000 | 0.015 | 0.7325 | 0.7505 |
