# Mass-aware sampling weights

`analysis/mass_weights.py` on `craftax_expert_val.h5`: 262 episodes, 151,017 frames, 25 s. Rule: kept only if mass x D >= kappa; D = 192, kappa = 1.5, so the threshold is 0.78% of the pixel variance. Weights between 1/cap and cap, effective sample size >= the stated share.

**Label-free classes:** k-means of the 7x7 tile patches, k = 96. Against the simulator's labels: purity 0.92, normalised mutual information 0.56.
**Label-free weights:** effective sample size 0.50, max weight 8.1, 28% of the frames upweighted; 68 clusters under the threshold, 5 of 68 targeted ones at 90% of it afterwards.
**Weights solved with the labels:** effective sample size 0.50, max weight 8.2.

## What the weights do to the real classes (mass in % of the pixel variance)

`best 10%`: the class's mass if only the 10% of the frames that carry the most of it per unit of variance were sampled (effective sample size 0.1; each class on its own, so the classes cannot all have it at once). It is bounded by the class's share *within* a frame. `in frames`: the share of frames that contain it.

| class | natural | label-free weights | weights with labels | best 10% | in frames | threshold |
|---|---|---|---|---|---|---|
| grass | 12.78% (kept) | 15.68% (kept) | 12.23% (kept) | 48.07% (kept) | 93% | 0.78% |
| sand | 10.91% (kept) | 11.34% (kept) | 9.84% (kept) | 55.65% (kept) | 70% | 0.78% |
| water | 6.27% (kept) | 6.24% (kept) | 5.74% (kept) | 36.69% (kept) | 53% | 0.78% |
| stone | 3.63% (kept) | 4.75% (kept) | 3.99% (kept) | 18.72% (kept) | 76% | 0.78% |
| tree | 1.57% (kept) | 2.04% (kept) | 1.42% (kept) | 8.03% (kept) | 67% | 0.78% |
| path | 1.32% (kept) | 1.65% (kept) | 1.58% (kept) | 8.87% (kept) | 69% | 0.78% |
| lava | 0.95% (kept) | 1.39% (kept) | 1.16% (kept) | 10.44% (kept) | 17% | 0.78% |
| coal | 0.30% | 0.49% | 0.36% | 2.37% (kept) | 28% | 0.78% |
| mob:cow | 0.28% | 0.47% | 0.35% | 1.82% (kept) | 33% | 0.78% |
| furnace | 0.26% | 0.40% | 0.33% | 2.15% (kept) | 25% | 0.78% |
| iron | 0.17% | 0.23% | 0.21% | 1.56% (kept) | 21% | 0.78% |
| out_of_bounds | 0.15% | 0.19% | 0.20% | 1.38% (kept) | 4% | 0.78% |
| crafting_table | 0.15% | 0.21% | 0.19% | 1.04% (kept) | 27% | 0.78% |
| mob:zombie | 0.12% | 0.25% | 0.18% | 0.91% (kept) | 31% | 0.78% |
| mob:skeleton | 0.07% | 0.12% | 0.21% | 0.79% (kept) | 13% | 0.78% |
| plant | 0.07% | 0.11% | 0.12% | 0.63% | 8% | 0.78% |
| diamond | 0.05% | 0.06% | 0.10% | 0.46% | 4% | 0.78% |
| mob:arrow | 0.01% | 0.02% | 0.04% | 0.12% | 8% | 0.78% |
| ripe_plant | 0.00% | 0.00% | 0.00% | 0.01% | 0% | 0.78% |
