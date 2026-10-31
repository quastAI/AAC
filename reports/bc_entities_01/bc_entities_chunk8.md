# BC entities, chunk 8

A model reads the state of one frame and predicts the expert's next 8 actions. 104299 validation samples of 128 episodes (the final expert's, held-out worlds). Cross-entropy 1.375 nats against 2.130 for the marginal distribution of the target: the model gains 0.755 nats ('skill'). Accuracy per step: 0.69, 0.59, 0.53, 0.51, 0.48, 0.47, 0.46, 0.45. Skill lost: the rise of the cross-entropy when the entity's columns come from another frame, in shares of the skill (permutation: reliance, not necessity); interval: episode bootstrap. Constant: the columns never vary.

## Coarse parts

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| sleeping | 1.293 | 1.25 to 1.33 | 0.9769 | -0.1657 | 1.8479 | 0.5899 |
| view | 0.774 | 0.76 to 0.80 | 0.5849 | -0.1397 | 1.4700 | 0.3294 |
| facing | 0.362 | 0.35 to 0.38 | 0.2733 | -0.1009 | 0.6895 | 0.1173 |
| inventory | 0.359 | 0.34 to 0.39 | 0.2711 | -0.0491 | 0.3914 | 0.2064 |
| vitals | 0.292 | 0.28 to 0.31 | 0.2210 | -0.0384 | 0.2635 | 0.2106 |
| daylight | 0.023 | 0.02 to 0.03 | 0.0172 | -0.0036 | 0.0246 | 0.0143 |

## Block types in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| grass | 0.433 | 0.41 to 0.46 | 0.3269 | -0.0702 | 0.4363 | 0.2694 |
| water | 0.223 | 0.21 to 0.24 | 0.1688 | -0.0425 | 0.3283 | 0.1103 |
| stone | 0.214 | 0.20 to 0.23 | 0.1617 | -0.0434 | 0.3705 | 0.1011 |
| path | 0.127 | 0.12 to 0.14 | 0.0960 | -0.0261 | 0.1736 | 0.0685 |
| sand | 0.117 | 0.11 to 0.13 | 0.0881 | -0.0213 | 0.1453 | 0.0656 |
| crafting_table | 0.044 | 0.04 to 0.05 | 0.0332 | -0.0060 | 0.0764 | 0.0170 |
| tree | 0.043 | 0.04 to 0.05 | 0.0327 | -0.0096 | 0.1081 | 0.0111 |
| out_of_bounds | 0.036 | 0.03 to 0.04 | 0.0275 | -0.0071 | 0.0463 | 0.0192 |
| furnace | 0.015 | 0.01 to 0.02 | 0.0114 | -0.0035 | 0.0352 | 0.0051 |
| diamond | 0.015 | 0.01 to 0.02 | 0.0111 | -0.0025 | 0.0199 | 0.0067 |
| lava | 0.012 | 0.01 to 0.01 | 0.0087 | -0.0023 | 0.0193 | 0.0046 |
| iron | 0.010 | 0.01 to 0.01 | 0.0074 | -0.0026 | 0.0196 | 0.0035 |
| coal | 0.009 | 0.01 to 0.01 | 0.0065 | -0.0026 | 0.0197 | 0.0026 |
| plant | 0.006 | 0.00 to 0.01 | 0.0045 | -0.0009 | 0.0101 | 0.0033 |
| ripe_plant | 0.001 | 0.00 to 0.00 | 0.0008 | -0.0002 | 0.0023 | 0.0003 |
| wood (constant) | 0.000 | 0.00 to 0.00 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |

## Mobs in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| zombie | 0.122 | 0.12 to 0.13 | 0.0919 | -0.0216 | 0.1971 | 0.0597 |
| skeleton | 0.017 | 0.02 to 0.02 | 0.0127 | -0.0033 | 0.0417 | 0.0053 |
| cow | 0.015 | 0.01 to 0.02 | 0.0110 | -0.0030 | 0.0467 | 0.0031 |
| arrow | 0.008 | 0.01 to 0.01 | 0.0059 | -0.0008 | 0.0100 | 0.0047 |

## Distance from the player (all channels)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| 1 | 0.350 | 0.34 to 0.36 | 0.2646 | -0.0677 | 1.0944 | 0.0899 |
| 3 | 0.149 | 0.14 to 0.16 | 0.1127 | -0.0349 | 0.1466 | 0.0876 |
| 2 | 0.149 | 0.14 to 0.16 | 0.1122 | -0.0358 | 0.1519 | 0.0675 |
| 4 | 0.083 | 0.08 to 0.09 | 0.0629 | -0.0167 | 0.1016 | 0.0426 |
| 0 | 0.002 | 0.00 to 0.00 | 0.0014 | -0.0005 | 0.0022 | 0.0009 |

## Inventory

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| iron_sword | 0.116 | 0.11 to 0.12 | 0.0879 | -0.0163 | 0.1182 | 0.0680 |
| diamond | 0.063 | 0.06 to 0.07 | 0.0475 | -0.0119 | 0.0571 | 0.0412 |
| wood | 0.021 | 0.02 to 0.02 | 0.0156 | -0.0022 | 0.0281 | 0.0084 |
| stone | 0.013 | 0.01 to 0.01 | 0.0095 | -0.0020 | 0.0159 | 0.0077 |
| iron_pickaxe | 0.012 | 0.01 to 0.01 | 0.0094 | -0.0022 | 0.0143 | 0.0074 |
| iron | 0.010 | 0.01 to 0.01 | 0.0075 | -0.0017 | 0.0108 | 0.0057 |
| coal | 0.006 | 0.00 to 0.01 | 0.0047 | -0.0014 | 0.0060 | 0.0039 |
| stone_pickaxe | 0.003 | 0.00 to 0.00 | 0.0026 | -0.0005 | 0.0048 | 0.0020 |
| stone_sword | 0.003 | 0.00 to 0.00 | 0.0023 | -0.0004 | 0.0036 | 0.0021 |
| sapling | 0.001 | 0.00 to 0.00 | 0.0011 | -0.0004 | 0.0014 | 0.0009 |
| wood_pickaxe | 0.001 | 0.00 to 0.00 | 0.0009 | -0.0003 | 0.0012 | 0.0009 |
| wood_sword | 0.001 | 0.00 to 0.00 | 0.0006 | -0.0000 | 0.0004 | 0.0008 |

## Vitals

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| energy | 0.180 | 0.17 to 0.19 | 0.1356 | -0.0231 | 0.1037 | 0.1710 |
| drink | 0.140 | 0.13 to 0.15 | 0.1056 | -0.0235 | 0.1608 | 0.0822 |
| health | 0.024 | 0.02 to 0.03 | 0.0183 | -0.0041 | 0.0271 | 0.0149 |
| food | 0.021 | 0.02 to 0.02 | 0.0156 | -0.0045 | 0.0215 | 0.0117 |

## Facing, daylight, sleeping

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| sleeping | 1.293 | 1.25 to 1.33 | 0.9769 | -0.1657 | 1.8479 | 0.5899 |
| facing | 0.362 | 0.35 to 0.38 | 0.2733 | -0.1009 | 0.6895 | 0.1173 |
| daylight | 0.023 | 0.02 to 0.03 | 0.0172 | -0.0036 | 0.0246 | 0.0143 |

## Skill lost by place (the player stands at the centre, up is up)

| | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| -3 | 0.004 | 0.004 | 0.004 | 0.005 | 0.008 | 0.004 | 0.004 | 0.003 | 0.005 |
| -2 | 0.003 | 0.003 | 0.005 | 0.006 | 0.016 | 0.006 | 0.004 | 0.003 | 0.003 |
| -1 | 0.003 | 0.003 | 0.006 | 0.018 | 0.084 | 0.017 | 0.005 | 0.002 | 0.003 |
| +0 | 0.004 | 0.004 | 0.011 | 0.057 | 0.002 | 0.058 | 0.010 | 0.004 | 0.003 |
| +1 | 0.003 | 0.004 | 0.006 | 0.017 | 0.077 | 0.015 | 0.005 | 0.003 | 0.003 |
| +2 | 0.003 | 0.003 | 0.005 | 0.006 | 0.016 | 0.006 | 0.005 | 0.003 | 0.003 |
| +3 | 0.005 | 0.003 | 0.004 | 0.005 | 0.009 | 0.004 | 0.004 | 0.003 | 0.005 |

## Trained again without the entity

Skill lost: the rise of the validation cross-entropy over the model with everything, in shares of its skill (one seed, no interval).

| entity | skill lost | permutation | ce | accuracy |
|---|---|---|---|---|
| group:sleeping | 0.461 | 1.293 | 1.7227 | 0.4124 |
| group:view | 0.290 | 0.774 | 1.5939 | 0.4459 |
| group:facing | 0.116 | 0.362 | 1.4623 | 0.4750 |
| group:vitals | 0.094 | 0.292 | 1.4455 | 0.5071 |
| group:inventory | 0.067 | 0.359 | 1.4255 | 0.5118 |
| vitals:energy | 0.051 | 0.180 | 1.4134 | 0.5137 |
| mob:zombie | 0.050 | 0.122 | 1.4123 | 0.5118 |
| vitals:drink | 0.022 | 0.140 | 1.3914 | 0.5162 |
| inventory:iron_sword | 0.020 | 0.116 | 1.3895 | 0.5185 |
| inventory:diamond | 0.019 | 0.063 | 1.3892 | 0.5182 |
| vitals:health | 0.013 | 0.024 | 1.3846 | 0.5190 |
| group:daylight | 0.010 | 0.023 | 1.3825 | 0.5183 |
