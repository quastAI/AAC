# Multi-step inverse dynamics, 1 steps

A model reads the state of frame t and of frame t+1 and predicts the 1 actions between them, each step as its own distribution (many action sequences lead to the same state: the target is not deterministic); an entity comes from another random pair, in both frames. 105195 validation samples of 128 episodes (the final expert's, held-out worlds). Cross-entropy 0.202 nats against 2.130 for the marginal distribution of the target: the model gains 1.929 nats ('skill'). Accuracy per step: 0.93. Skill lost: the rise of the cross-entropy when the entity's columns come from another frame, in shares of the skill (permutation: reliance, not necessity); interval: episode bootstrap. Constant: the columns never vary.

## Coarse parts

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| view | 2.057 | 2.02 to 2.09 | 3.9677 | -0.3862 |
| facing | 1.928 | 1.90 to 1.96 | 3.7194 | -0.3859 |
| sleeping | 1.754 | 1.73 to 1.78 | 3.3835 | -0.2468 |
| inventory | 0.099 | 0.09 to 0.11 | 0.1918 | -0.0338 |
| vitals | 0.065 | 0.06 to 0.07 | 0.1251 | -0.0274 |
| daylight | 0.006 | 0.00 to 0.01 | 0.0112 | -0.0041 |

## Block types in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| stone | 0.227 | 0.21 to 0.24 | 0.4388 | -0.0573 |
| grass | 0.156 | 0.15 to 0.17 | 0.3007 | -0.0607 |
| water | 0.118 | 0.11 to 0.12 | 0.2285 | -0.0470 |
| tree | 0.094 | 0.09 to 0.10 | 0.1818 | -0.0292 |
| path | 0.088 | 0.08 to 0.09 | 0.1698 | -0.0359 |
| furnace | 0.037 | 0.03 to 0.04 | 0.0718 | -0.0136 |
| sand | 0.037 | 0.03 to 0.04 | 0.0709 | -0.0193 |
| crafting_table | 0.029 | 0.03 to 0.03 | 0.0558 | -0.0131 |
| out_of_bounds | 0.011 | 0.01 to 0.01 | 0.0221 | -0.0041 |
| plant | 0.006 | 0.01 to 0.01 | 0.0114 | -0.0025 |
| diamond | 0.005 | 0.00 to 0.01 | 0.0103 | -0.0022 |
| coal | 0.004 | 0.00 to 0.00 | 0.0080 | -0.0017 |
| iron | 0.004 | 0.00 to 0.00 | 0.0074 | -0.0020 |
| lava | 0.003 | 0.00 to 0.00 | 0.0048 | -0.0012 |
| ripe_plant | 0.001 | 0.00 to 0.00 | 0.0027 | -0.0004 |
| wood (constant) | 0.000 | 0.00 to 0.00 | 0.0000 | +0.0000 |

## Mobs in the view (all tiles)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| zombie | 0.051 | 0.05 to 0.05 | 0.0982 | -0.0187 |
| skeleton | 0.014 | 0.01 to 0.02 | 0.0269 | -0.0071 |
| cow | 0.012 | 0.01 to 0.01 | 0.0222 | -0.0058 |
| arrow | 0.001 | 0.00 to 0.00 | 0.0025 | -0.0008 |

## Distance from the player (all channels)

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| 1 | 1.018 | 0.99 to 1.04 | 1.9633 | -0.2252 |
| 2 | 0.120 | 0.12 to 0.12 | 0.2313 | -0.0557 |
| 3 | 0.114 | 0.11 to 0.12 | 0.2206 | -0.0580 |
| 4 | 0.024 | 0.02 to 0.03 | 0.0464 | -0.0137 |
| 0 | 0.001 | 0.00 to 0.00 | 0.0020 | -0.0005 |

## Inventory

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| iron_sword | 0.015 | 0.01 to 0.02 | 0.0280 | -0.0085 |
| diamond | 0.008 | 0.01 to 0.01 | 0.0164 | -0.0053 |
| iron_pickaxe | 0.003 | 0.00 to 0.00 | 0.0056 | -0.0019 |
| stone | 0.003 | 0.00 to 0.00 | 0.0052 | -0.0016 |
| iron | 0.002 | 0.00 to 0.00 | 0.0043 | -0.0016 |
| coal | 0.002 | 0.00 to 0.00 | 0.0038 | -0.0010 |
| wood | 0.002 | 0.00 to 0.00 | 0.0037 | -0.0011 |
| stone_pickaxe | 0.001 | 0.00 to 0.00 | 0.0024 | -0.0009 |
| stone_sword | 0.001 | 0.00 to 0.00 | 0.0013 | -0.0004 |
| sapling | 0.000 | 0.00 to 0.00 | 0.0008 | +0.0001 |
| wood_pickaxe | 0.000 | 0.00 to 0.00 | 0.0003 | -0.0001 |
| wood_sword | 0.000 | 0.00 to 0.00 | 0.0002 | -0.0001 |

## Vitals

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| drink | 0.028 | 0.03 to 0.03 | 0.0534 | -0.0148 |
| energy | 0.021 | 0.02 to 0.02 | 0.0404 | -0.0082 |
| health | 0.007 | 0.01 to 0.01 | 0.0130 | -0.0039 |
| food | 0.004 | 0.00 to 0.00 | 0.0073 | -0.0023 |

## Facing, daylight, sleeping

| entity | skill lost | 95% interval | Δ ce | Δ accuracy |
|---|---|---|---|---|
| facing | 1.928 | 1.90 to 1.96 | 3.7194 | -0.3859 |
| sleeping | 1.754 | 1.73 to 1.78 | 3.3835 | -0.2468 |
| daylight | 0.006 | 0.00 to 0.01 | 0.0112 | -0.0041 |

## Skill lost by place (the player stands at the centre, up is up)

| | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| -3 | 0.001 | 0.001 | 0.001 | 0.001 | 0.003 | 0.001 | 0.000 | 0.001 | 0.001 |
| -2 | 0.001 | 0.001 | 0.002 | 0.002 | 0.010 | 0.002 | 0.001 | 0.001 | 0.001 |
| -1 | 0.001 | 0.001 | 0.002 | 0.010 | 0.241 | 0.009 | 0.002 | 0.001 | 0.001 |
| +0 | 0.001 | 0.003 | 0.007 | 0.161 | 0.001 | 0.167 | 0.008 | 0.003 | 0.001 |
| +1 | 0.001 | 0.001 | 0.003 | 0.010 | 0.218 | 0.009 | 0.002 | 0.001 | 0.000 |
| +2 | 0.001 | 0.001 | 0.002 | 0.002 | 0.010 | 0.002 | 0.002 | 0.001 | 0.001 |
| +3 | 0.001 | 0.001 | 0.001 | 0.001 | 0.004 | 0.001 | 0.001 | 0.001 | 0.001 |

## Trained again without the entity

Skill lost: the rise of the validation cross-entropy over the model with everything, in shares of its skill (one seed, no interval).

| entity | skill lost | permutation | ce | accuracy |
|---|---|---|---|---|
| group:view | 0.178 | 2.057 | 0.5448 | 0.8232 |
| group:sleeping | 0.113 | 1.754 | 0.4193 | 0.8685 |
| group:facing | 0.037 | 1.928 | 0.2721 | 0.9114 |
| group:vitals | 0.022 | 0.065 | 0.2432 | 0.9203 |
| mob:zombie | 0.012 | 0.051 | 0.2247 | 0.9257 |
| vitals:drink | 0.010 | 0.028 | 0.2204 | 0.9262 |
| vitals:energy | 0.007 | 0.021 | 0.2149 | 0.9293 |
| group:inventory | 0.006 | 0.099 | 0.2138 | 0.9299 |
| mob:skeleton | 0.004 | 0.014 | 0.2098 | 0.9311 |
| group:daylight | 0.003 | 0.006 | 0.2072 | 0.9315 |
| inventory:iron_sword | 0.001 | 0.015 | 0.2042 | 0.9330 |
| mob:cow | -0.000 | 0.012 | 0.2009 | 0.9348 |
