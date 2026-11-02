# Multi-step inverse dynamics, 8 steps

A model reads the state of frame t and of frame t+8 and predicts the 8 actions between them, each step as its own distribution (many action sequences lead to the same state: the target is not deterministic); an entity comes from another random pair, in both frames. 104299 validation samples of 128 episodes (the final expert's, held-out worlds). Cross-entropy 1.010 nats against 2.130 for the marginal distribution of the target: the model gains 1.120 nats ('skill'). Accuracy per step: 0.71, 0.62, 0.60, 0.59, 0.60, 0.60, 0.64, 0.79. Skill lost: the rise of the cross-entropy when the entity's columns come from another frame, in shares of the skill (permutation: reliance, not necessity); interval: episode bootstrap. Constant: the columns never vary.

## Coarse parts

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| view | 0.942 | 0.93 to 0.96 | 1.0547 | -0.2588 | 1.5793 | 0.7674 |
| sleeping | 0.934 | 0.91 to 0.96 | 1.0462 | -0.1815 | 1.5152 | 1.2400 |
| facing | 0.510 | 0.50 to 0.53 | 0.5714 | -0.1251 | 0.4055 | 2.7006 |
| inventory | 0.189 | 0.18 to 0.20 | 0.2113 | -0.0368 | 0.2633 | 0.1915 |
| vitals | 0.105 | 0.10 to 0.11 | 0.1174 | -0.0237 | 0.1363 | 0.1087 |
| daylight | 0.010 | 0.01 to 0.01 | 0.0107 | -0.0026 | 0.0117 | 0.0109 |

## Block types in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| grass | 0.231 | 0.22 to 0.24 | 0.2585 | -0.0682 | 0.3143 | 0.1949 |
| stone | 0.114 | 0.11 to 0.12 | 0.1274 | -0.0427 | 0.2562 | 0.1257 |
| water | 0.111 | 0.10 to 0.12 | 0.1243 | -0.0331 | 0.1880 | 0.0903 |
| path | 0.105 | 0.10 to 0.11 | 0.1172 | -0.0436 | 0.1509 | 0.1302 |
| sand | 0.058 | 0.05 to 0.06 | 0.0648 | -0.0175 | 0.0913 | 0.0493 |
| crafting_table | 0.034 | 0.03 to 0.04 | 0.0383 | -0.0076 | 0.0459 | 0.0554 |
| furnace | 0.029 | 0.03 to 0.03 | 0.0321 | -0.0050 | 0.0331 | 0.0513 |
| tree | 0.024 | 0.02 to 0.03 | 0.0265 | -0.0092 | 0.0772 | 0.0177 |
| out_of_bounds | 0.012 | 0.01 to 0.01 | 0.0129 | -0.0037 | 0.0221 | 0.0069 |
| diamond | 0.009 | 0.01 to 0.01 | 0.0097 | -0.0020 | 0.0152 | 0.0082 |
| lava | 0.007 | 0.01 to 0.01 | 0.0081 | -0.0025 | 0.0141 | 0.0048 |
| plant | 0.007 | 0.01 to 0.01 | 0.0076 | -0.0016 | 0.0124 | 0.0108 |
| coal | 0.005 | 0.00 to 0.01 | 0.0060 | -0.0023 | 0.0167 | 0.0051 |
| iron | 0.005 | 0.00 to 0.01 | 0.0053 | -0.0019 | 0.0146 | 0.0046 |
| ripe_plant | 0.001 | 0.00 to 0.00 | 0.0007 | -0.0002 | 0.0017 | 0.0004 |
| wood (constant) | 0.000 | 0.00 to 0.00 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |

## Mobs in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| zombie | 0.044 | 0.04 to 0.05 | 0.0490 | -0.0111 | 0.1252 | 0.0313 |
| skeleton | 0.010 | 0.01 to 0.01 | 0.0116 | -0.0030 | 0.0327 | 0.0084 |
| cow | 0.007 | 0.01 to 0.01 | 0.0084 | -0.0022 | 0.0368 | 0.0028 |
| arrow | 0.003 | 0.00 to 0.00 | 0.0030 | -0.0009 | 0.0078 | 0.0025 |

## Distance from the player (all channels)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| 1 | 0.289 | 0.28 to 0.30 | 0.3242 | -0.0930 | 0.8465 | 0.4741 |
| 3 | 0.169 | 0.16 to 0.17 | 0.1889 | -0.0651 | 0.2016 | 0.0794 |
| 2 | 0.120 | 0.12 to 0.13 | 0.1344 | -0.0392 | 0.1444 | 0.0954 |
| 4 | 0.086 | 0.08 to 0.09 | 0.0963 | -0.0314 | 0.1089 | 0.0354 |
| 0 | 0.009 | 0.01 to 0.01 | 0.0100 | -0.0027 | 0.0048 | 0.0053 |

## Inventory

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| iron_sword | 0.040 | 0.04 to 0.04 | 0.0451 | -0.0074 | 0.0579 | 0.0397 |
| wood | 0.017 | 0.02 to 0.02 | 0.0186 | -0.0022 | 0.0247 | 0.0191 |
| diamond | 0.016 | 0.01 to 0.02 | 0.0181 | -0.0028 | 0.0194 | 0.0194 |
| stone | 0.010 | 0.01 to 0.01 | 0.0110 | -0.0024 | 0.0160 | 0.0106 |
| iron | 0.005 | 0.00 to 0.01 | 0.0056 | -0.0012 | 0.0065 | 0.0055 |
| iron_pickaxe | 0.004 | 0.00 to 0.01 | 0.0050 | -0.0010 | 0.0074 | 0.0045 |
| coal | 0.003 | 0.00 to 0.00 | 0.0032 | -0.0005 | 0.0034 | 0.0033 |
| stone_pickaxe | 0.002 | 0.00 to 0.00 | 0.0018 | -0.0004 | 0.0034 | 0.0019 |
| stone_sword | 0.001 | 0.00 to 0.00 | 0.0015 | -0.0003 | 0.0026 | 0.0017 |
| sapling | 0.001 | 0.00 to 0.00 | 0.0009 | -0.0002 | 0.0014 | 0.0007 |
| wood_pickaxe | 0.000 | 0.00 to 0.00 | 0.0003 | -0.0001 | 0.0004 | 0.0003 |
| wood_sword | 0.000 | 0.00 to 0.00 | 0.0001 | -0.0000 | 0.0001 | 0.0001 |

## Vitals

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| drink | 0.042 | 0.04 to 0.04 | 0.0470 | -0.0117 | 0.0607 | 0.0390 |
| energy | 0.035 | 0.03 to 0.04 | 0.0388 | -0.0062 | 0.0372 | 0.0419 |
| health | 0.009 | 0.01 to 0.01 | 0.0106 | -0.0017 | 0.0113 | 0.0109 |
| food | 0.009 | 0.01 to 0.01 | 0.0098 | -0.0019 | 0.0107 | 0.0097 |

## Facing, daylight, sleeping

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 8 |
|---|---|---|---|---|---|---|
| sleeping | 0.934 | 0.91 to 0.96 | 1.0462 | -0.1815 | 1.5152 | 1.2400 |
| facing | 0.510 | 0.50 to 0.53 | 0.5714 | -0.1251 | 0.4055 | 2.7006 |
| daylight | 0.010 | 0.01 to 0.01 | 0.0107 | -0.0026 | 0.0117 | 0.0109 |

## Skill lost by place (the player stands at the centre, up is up)

| | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| -3 | 0.004 | 0.003 | 0.003 | 0.004 | 0.006 | 0.003 | 0.003 | 0.003 | 0.004 |
| -2 | 0.002 | 0.002 | 0.003 | 0.004 | 0.011 | 0.004 | 0.002 | 0.002 | 0.002 |
| -1 | 0.003 | 0.002 | 0.004 | 0.013 | 0.067 | 0.012 | 0.004 | 0.002 | 0.002 |
| +0 | 0.003 | 0.004 | 0.008 | 0.045 | 0.009 | 0.044 | 0.007 | 0.003 | 0.002 |
| +1 | 0.003 | 0.003 | 0.005 | 0.012 | 0.063 | 0.011 | 0.003 | 0.002 | 0.003 |
| +2 | 0.002 | 0.002 | 0.003 | 0.005 | 0.011 | 0.004 | 0.003 | 0.002 | 0.003 |
| +3 | 0.004 | 0.003 | 0.003 | 0.004 | 0.007 | 0.004 | 0.003 | 0.003 | 0.004 |

## Trained again without the entity

Skill lost: the rise of the validation cross-entropy over the model with everything, in shares of its skill (one seed, no interval).

| entity | skill lost | permutation | ce | accuracy |
|---|---|---|---|---|
| group:view | 0.282 | 0.942 | 1.3258 | 0.5321 |
| group:facing | 0.093 | 0.510 | 1.1148 | 0.6057 |
| group:sleeping | 0.087 | 0.934 | 1.1079 | 0.6195 |
| group:vitals | 0.026 | 0.105 | 1.0397 | 0.6369 |
| group:inventory | 0.022 | 0.189 | 1.0346 | 0.6421 |
| mob:zombie | 0.019 | 0.044 | 1.0314 | 0.6384 |
| vitals:energy | 0.018 | 0.035 | 1.0307 | 0.6404 |
| vitals:drink | 0.007 | 0.042 | 1.0185 | 0.6402 |
| inventory:wood | 0.005 | 0.017 | 1.0155 | 0.6437 |
| group:daylight | 0.005 | 0.010 | 1.0153 | 0.6437 |
| inventory:diamond | 0.004 | 0.016 | 1.0142 | 0.6437 |
| inventory:iron_sword | 0.002 | 0.040 | 1.0129 | 0.6444 |
