# Crop and pixel mass

`analysis/crop_mass.py` on `craftax_expert_val.h5`, 151,017 frames (daylight >= 0.0), 17 s. Mass = share of the cropped tiles' pixel variance (inventory strip left out); threshold kappa/D = 0.78% (D = 192, kappa = 1.5). `cov` = share of the class's tiles that stay in the crop.

| class | full 7x9 (63 tiles) | 7x7 (49 tiles) | 5x7 (35 tiles) | 5x5 (25 tiles) | 3x5 (15 tiles) | 3x3 (9 tiles) |
|---|---|---|---|---|---|---|
| grass | 17.01%* (cov 100%) | 16.86%* (cov 78%) | 16.70%* (cov 55%) | 16.43%* (cov 39%) | 16.02%* (cov 22%) | 15.17%* (cov 12%) |
| sand | 14.52%* (cov 100%) | 14.70%* (cov 79%) | 14.87%* (cov 57%) | 15.17%* (cov 42%) | 15.31%* (cov 26%) | 15.45%* (cov 16%) |
| water | 8.34%* (cov 100%) | 8.38%* (cov 77%) | 8.30%* (cov 54%) | 8.12%* (cov 37%) | 7.68%* (cov 20%) | 6.94%* (cov 11%) |
| stone | 4.83%* (cov 100%) | 4.85%* (cov 77%) | 4.83%* (cov 54%) | 4.67%* (cov 37%) | 4.31%* (cov 21%) | 3.79%* (cov 11%) |
| tree | 2.09%* (cov 100%) | 2.06%* (cov 76%) | 2.06%* (cov 54%) | 2.05%* (cov 38%) | 2.01%* (cov 22%) | 1.88%* (cov 12%) |
| path | 1.75%* (cov 100%) | 1.80%* (cov 78%) | 1.86%* (cov 57%) | 1.94%* (cov 42%) | 2.02%* (cov 26%) | 2.12%* (cov 16%) |
| lava | 1.27%* (cov 100%) | 1.11%* (cov 68%) | 0.96%* (cov 42%) | 0.80%* (cov 25%) | 0.61% (cov 12%) | 0.44% (cov 5%) |
| coal | 0.40% (cov 100%) | 0.40% (cov 77%) | 0.40% (cov 54%) | 0.40% (cov 39%) | 0.38% (cov 22%) | 0.35% (cov 12%) |
| mob:cow | 0.37% (cov 100%) | 0.37% (cov 77%) | 0.37% (cov 55%) | 0.36% (cov 39%) | 0.36% (cov 23%) | 0.35% (cov 14%) |
| furnace | 0.35% (cov 100%) | 0.40% (cov 89%) | 0.48% (cov 75%) | 0.57% (cov 63%) | 0.70% (cov 46%) | 0.90%* (cov 35%) |
| iron | 0.22% (cov 100%) | 0.22% (cov 76%) | 0.22% (cov 55%) | 0.23% (cov 40%) | 0.22% (cov 24%) | 0.20% (cov 13%) |
| out_of_bounds | 0.20% (cov 100%) | 0.15% (cov 57%) | 0.07% (cov 18%) | 0.05% (cov 10%) | 0.03% (cov 4%) | 0.02% (cov 2%) |
| crafting_table | 0.19% (cov 100%) | 0.22% (cov 89%) | 0.27% (cov 76%) | 0.32% (cov 65%) | 0.41% (cov 51%) | 0.54% (cov 40%) |
| mob:zombie | 0.16% (cov 100%) | 0.17% (cov 83%) | 0.18% (cov 60%) | 0.20% (cov 47%) | 0.21% (cov 29%) | 0.25% (cov 21%) |
| mob:skeleton | 0.10% (cov 100%) | 0.09% (cov 71%) | 0.08% (cov 44%) | 0.06% (cov 26%) | 0.06% (cov 14%) | 0.05% (cov 7%) |
| plant | 0.09% (cov 100%) | 0.10% (cov 85%) | 0.11% (cov 66%) | 0.12% (cov 53%) | 0.15% (cov 37%) | 0.19% (cov 29%) |
| diamond | 0.06% (cov 100%) | 0.07% (cov 85%) | 0.08% (cov 68%) | 0.08% (cov 50%) | 0.09% (cov 33%) | 0.06% (cov 13%) |
| mob:arrow | 0.02% (cov 100%) | 0.02% (cov 93%) | 0.02% (cov 79%) | 0.03% (cov 67%) | 0.03% (cov 40%) | 0.02% (cov 23%) |
| ripe_plant | 0.00% (cov 100%) | 0.00% (cov 89%) | 0.00% (cov 74%) | 0.00% (cov 62%) | 0.00% (cov 35%) | 0.00% (cov 27%) |
| **classes at or above the threshold** | 7 of 19 | 7 of 19 | 7 of 19 | 7 of 19 | 6 of 19 | 7 of 19 |
