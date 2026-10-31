# BC entities, chunk 4

A model reads the state of one frame and predicts the expert's next 4 actions. 104811 validation samples of 128 episodes (the final expert's, held-out worlds). Cross-entropy 1.209 nats against 2.130 for the marginal distribution of the target: the model gains 0.921 nats ('skill'). Accuracy per step: 0.70, 0.60, 0.54, 0.51. Skill lost: the rise of the cross-entropy when the entity's columns come from another frame, in shares of the skill (permutation: reliance, not necessity); interval: episode bootstrap. Constant: the columns never vary.

## Coarse parts

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| sleeping | 1.414 | 1.38 to 1.45 | 1.3031 | -0.1967 | 2.0709 | 0.8561 |
| view | 0.929 | 0.91 to 0.95 | 0.8554 | -0.1892 | 1.6366 | 0.4575 |
| facing | 0.470 | 0.45 to 0.48 | 0.4327 | -0.1415 | 0.7746 | 0.2365 |
| inventory | 0.347 | 0.33 to 0.37 | 0.3192 | -0.0578 | 0.4134 | 0.2466 |
| vitals | 0.242 | 0.23 to 0.26 | 0.2226 | -0.0423 | 0.2743 | 0.1913 |
| daylight | 0.019 | 0.02 to 0.02 | 0.0171 | -0.0033 | 0.0225 | 0.0139 |

## Block types in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| grass | 0.405 | 0.38 to 0.43 | 0.3729 | -0.0802 | 0.4789 | 0.2991 |
| stone | 0.246 | 0.24 to 0.26 | 0.2269 | -0.0630 | 0.3940 | 0.1299 |
| water | 0.236 | 0.22 to 0.25 | 0.2176 | -0.0547 | 0.3429 | 0.1384 |
| path | 0.135 | 0.12 to 0.14 | 0.1245 | -0.0347 | 0.1903 | 0.0852 |
| sand | 0.120 | 0.11 to 0.13 | 0.1105 | -0.0275 | 0.1620 | 0.0804 |
| tree | 0.062 | 0.06 to 0.07 | 0.0572 | -0.0168 | 0.1168 | 0.0192 |
| crafting_table | 0.056 | 0.05 to 0.06 | 0.0515 | -0.0091 | 0.0855 | 0.0284 |
| out_of_bounds | 0.037 | 0.03 to 0.04 | 0.0341 | -0.0080 | 0.0494 | 0.0241 |
| furnace | 0.023 | 0.02 to 0.03 | 0.0214 | -0.0067 | 0.0459 | 0.0097 |
| diamond | 0.018 | 0.01 to 0.02 | 0.0164 | -0.0037 | 0.0230 | 0.0116 |
| lava | 0.014 | 0.01 to 0.01 | 0.0126 | -0.0033 | 0.0213 | 0.0072 |
| iron | 0.013 | 0.01 to 0.01 | 0.0118 | -0.0038 | 0.0215 | 0.0055 |
| coal | 0.011 | 0.01 to 0.01 | 0.0105 | -0.0036 | 0.0210 | 0.0041 |
| plant | 0.007 | 0.01 to 0.01 | 0.0061 | -0.0017 | 0.0116 | 0.0034 |
| ripe_plant | 0.002 | 0.00 to 0.00 | 0.0017 | -0.0004 | 0.0029 | 0.0007 |
| wood (constant) | 0.000 | 0.00 to 0.00 | 0.0000 | +0.0000 | 0.0000 | 0.0000 |

## Mobs in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| zombie | 0.139 | 0.13 to 0.15 | 0.1284 | -0.0298 | 0.2241 | 0.0792 |
| skeleton | 0.024 | 0.02 to 0.03 | 0.0223 | -0.0051 | 0.0494 | 0.0091 |
| cow | 0.022 | 0.02 to 0.02 | 0.0205 | -0.0053 | 0.0529 | 0.0046 |
| arrow | 0.008 | 0.01 to 0.01 | 0.0073 | -0.0013 | 0.0122 | 0.0055 |

## Distance from the player (all channels)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| 1 | 0.534 | 0.52 to 0.55 | 0.4922 | -0.1170 | 1.2527 | 0.1616 |
| 2 | 0.179 | 0.17 to 0.18 | 0.1649 | -0.0576 | 0.1678 | 0.1038 |
| 3 | 0.130 | 0.13 to 0.14 | 0.1201 | -0.0344 | 0.1484 | 0.0970 |
| 4 | 0.082 | 0.08 to 0.09 | 0.0757 | -0.0210 | 0.1073 | 0.0537 |
| 0 | 0.002 | 0.00 to 0.00 | 0.0017 | -0.0004 | 0.0021 | 0.0012 |

## Inventory

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| iron_sword | 0.102 | 0.10 to 0.11 | 0.0941 | -0.0171 | 0.1158 | 0.0749 |
| diamond | 0.052 | 0.05 to 0.06 | 0.0477 | -0.0122 | 0.0550 | 0.0409 |
| wood | 0.026 | 0.02 to 0.03 | 0.0236 | -0.0038 | 0.0322 | 0.0164 |
| iron_pickaxe | 0.013 | 0.01 to 0.01 | 0.0122 | -0.0031 | 0.0168 | 0.0088 |
| stone | 0.011 | 0.01 to 0.01 | 0.0102 | -0.0026 | 0.0161 | 0.0072 |
| iron | 0.007 | 0.01 to 0.01 | 0.0066 | -0.0018 | 0.0078 | 0.0056 |
| coal | 0.005 | 0.00 to 0.01 | 0.0046 | -0.0008 | 0.0053 | 0.0038 |
| stone_pickaxe | 0.004 | 0.00 to 0.01 | 0.0040 | -0.0011 | 0.0067 | 0.0026 |
| stone_sword | 0.003 | 0.00 to 0.00 | 0.0028 | -0.0008 | 0.0043 | 0.0020 |
| sapling | 0.001 | 0.00 to 0.00 | 0.0012 | -0.0002 | 0.0014 | 0.0010 |
| wood_pickaxe | 0.001 | 0.00 to 0.00 | 0.0009 | -0.0003 | 0.0010 | 0.0008 |
| wood_sword | 0.000 | 0.00 to 0.00 | 0.0004 | -0.0001 | 0.0004 | 0.0004 |

## Vitals

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| drink | 0.134 | 0.13 to 0.14 | 0.1231 | -0.0269 | 0.1652 | 0.0965 |
| energy | 0.099 | 0.09 to 0.11 | 0.0909 | -0.0131 | 0.0873 | 0.0953 |
| health | 0.023 | 0.02 to 0.03 | 0.0215 | -0.0049 | 0.0277 | 0.0174 |
| food | 0.019 | 0.02 to 0.02 | 0.0175 | -0.0046 | 0.0219 | 0.0143 |

## Facing, daylight, sleeping

| entity | skill lost | 95% interval | Δ ce | Δ accuracy | Δ ce, step 1 | Δ ce, step 4 |
|---|---|---|---|---|---|---|
| sleeping | 1.414 | 1.38 to 1.45 | 1.3031 | -0.1967 | 2.0709 | 0.8561 |
| facing | 0.470 | 0.45 to 0.48 | 0.4327 | -0.1415 | 0.7746 | 0.2365 |
| daylight | 0.019 | 0.02 to 0.02 | 0.0171 | -0.0033 | 0.0225 | 0.0139 |

## Skill lost by place (the player stands at the centre, up is up)

| | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| -3 | 0.004 | 0.004 | 0.003 | 0.004 | 0.009 | 0.004 | 0.004 | 0.003 | 0.005 |
| -2 | 0.003 | 0.003 | 0.005 | 0.006 | 0.026 | 0.007 | 0.005 | 0.003 | 0.003 |
| -1 | 0.003 | 0.003 | 0.007 | 0.022 | 0.140 | 0.020 | 0.004 | 0.002 | 0.003 |
| +0 | 0.004 | 0.005 | 0.016 | 0.094 | 0.002 | 0.093 | 0.015 | 0.004 | 0.003 |
| +1 | 0.002 | 0.003 | 0.006 | 0.021 | 0.125 | 0.019 | 0.004 | 0.002 | 0.003 |
| +2 | 0.003 | 0.003 | 0.005 | 0.007 | 0.025 | 0.005 | 0.005 | 0.003 | 0.003 |
| +3 | 0.004 | 0.003 | 0.004 | 0.004 | 0.010 | 0.004 | 0.004 | 0.003 | 0.004 |

## Trained again without the entity

Skill lost: the rise of the validation cross-entropy over the model with everything, in shares of its skill (one seed, no interval).

| entity | skill lost | permutation | ce | accuracy |
|---|---|---|---|---|
| group:sleeping | 0.464 | 1.414 | 1.6364 | 0.4579 |
| group:view | 0.347 | 0.929 | 1.5291 | 0.4733 |
| group:facing | 0.142 | 0.470 | 1.3401 | 0.5192 |
| group:vitals | 0.075 | 0.242 | 1.2782 | 0.5681 |
| mob:zombie | 0.060 | 0.139 | 1.2642 | 0.5699 |
| group:inventory | 0.057 | 0.347 | 1.2619 | 0.5721 |
| vitals:energy | 0.035 | 0.099 | 1.2411 | 0.5779 |
| vitals:drink | 0.023 | 0.134 | 1.2306 | 0.5764 |
| inventory:iron_sword | 0.015 | 0.102 | 1.2229 | 0.5805 |
| inventory:diamond | 0.015 | 0.052 | 1.2224 | 0.5803 |
| group:daylight | 0.009 | 0.019 | 1.2174 | 0.5809 |
| inventory:wood | 0.009 | 0.026 | 1.2170 | 0.5820 |
