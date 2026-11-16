# Pixel mass of Craftax-Classic content

`analysis/pixel_mass.py` on `craftax_expert_val.h5`: 262 random episodes (seed 0), 151,017 frames, 20 s. Shares are of the total pixel variance of the frame (what a decoder that predicts the mean image gets wrong).

## Plain frames

- map rows 75% of the variance, inventory strip 25%.
- all block and mob classes together explain 40%; daylight (linear, per pixel) about 9%; the rest is within-class texture and noise.

| class | tile share | share of the pixel variance |
|---|---|---|
| block:grass | 40.29% | 13.09% |
| block:sand | 11.59% | 11.21% |
| block:water | 13.26% | 6.27% |
| block:stone | 17.41% | 3.63% |
| block:tree | 3.24% | 1.57% |
| block:path | 10.33% | 1.36% |
| block:lava | 0.90% | 0.95% |
| mob:cow | 0.65% | 0.34% |
| block:coal | 0.64% | 0.30% |
| block:furnace | 0.49% | 0.26% |
| player:facing | - | 0.19% |
| block:iron | 0.46% | 0.17% |
| mob:zombie | 0.62% | 0.15% |
| block:out_of_bounds | 0.75% | 0.15% |
| block:crafting_table | 0.45% | 0.15% |
| block:plant | 0.13% | 0.07% |
| mob:skeleton | 0.22% | 0.06% |
| block:diamond | 0.06% | 0.05% |
| mob:arrow | 0.14% | 0.01% |
| block:ripe_plant | 0.00% | 0.00% |

## Differences x(t+1) - x(t)

- the player moves (the map scrolls) in 42% of the steps and those steps carry 90% of the plain difference energy (per step: moved 1.14e+07, still 9.41e+05).
- compensating the scroll leaves 24% of the plain difference energy. What remains, by tile:

| aligned difference | share |
|---|---|
| tiles with a mob in either frame | 20.8% |
| tiles whose block changed | 4.9% |
| the player tile (facing, sleeping) | 18.0% |
| rest: daylight, noise, animation | 56.3% |

## Mass against what a trained latent kept

Spearman rank correlation, per block or mob class, between its pixel mass (or tile share) and the ridge R^2 of its readout (`view:<block>`, `mob:<mob>`) from the latent at the given step of a race trace.

`kept` = the R^2 at that step over the same run's R^2 at step 0 (untrained encoder), for classes with R^2 above 0.1 at step 0: what training removed, not what random features already carry.

| trace | step | classes | rho (mass) | rho (tile share) | rho (mass vs kept) |
|---|---|---|---|---|---|
| A5 | 2000 | 18 | 0.84 | 0.86 | 0.73 (n=18) |
| B2 | 2000 | 18 | 0.84 | 0.78 | 0.68 (n=18) |
| V3 | 2000 | 18 | 0.86 | 0.80 | 0.67 (n=18) |
