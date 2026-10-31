# SIGReg pressure by batch composition

Axes are the principal axes of the latent, classed by what the game state explains of them (held-out episodes, MLP; heuristic threshold R^2 > 0.5). `variance` is the class's share of the latent's variance, `within` of the variance inside a window, `energy` of the gradient's energy (batch mean removed), `expand` of the positive first-order gain from stretching the axis. `shift` is the share of the gradient that moves the batch mean.

## epoch1

Axes: slow 6, explained 55, unexplained 131

| composition | SIGReg | shift | class | axes | variance | within | energy | expand |
|---|---|---|---|---|---|---|---|---|
| iid8 | 1.74 | 0.66 | slow | 6 | 0.16 | 0.16 | 0.04 | 0.00 |
| iid8 | 1.74 | 0.66 | explained | 55 | 0.80 | 0.80 | 0.33 | 0.87 |
| iid8 | 1.74 | 0.66 | unexplained | 131 | 0.04 | 0.04 | 0.63 | 0.13 |
| episodes8 | 3.10 | 0.73 | slow | 6 | 0.16 | 0.15 | 0.05 | 0.08 |
| episodes8 | 3.10 | 0.73 | explained | 55 | 0.80 | 0.81 | 0.36 | 0.84 |
| episodes8 | 3.10 | 0.73 | unexplained | 131 | 0.04 | 0.04 | 0.60 | 0.08 |
| window8 | 8.77 | 0.73 | slow | 6 | 0.16 | 0.12 | 0.05 | 0.09 |
| window8 | 8.77 | 0.73 | explained | 55 | 0.80 | 0.83 | 0.41 | 0.85 |
| window8 | 8.77 | 0.73 | unexplained | 131 | 0.04 | 0.05 | 0.54 | 0.06 |

## epoch10

Axes: slow 21, explained 74, unexplained 97

| composition | SIGReg | shift | class | axes | variance | within | energy | expand |
|---|---|---|---|---|---|---|---|---|
| iid8 | 1.37 | 0.70 | slow | 21 | 0.24 | 0.24 | 0.12 | 0.00 |
| iid8 | 1.37 | 0.70 | explained | 74 | 0.66 | 0.66 | 0.40 | 0.00 |
| iid8 | 1.37 | 0.70 | unexplained | 97 | 0.10 | 0.10 | 0.48 | 1.00 |
| episodes8 | 2.91 | 0.75 | slow | 21 | 0.24 | 0.23 | 0.12 | 0.15 |
| episodes8 | 2.91 | 0.75 | explained | 74 | 0.66 | 0.67 | 0.41 | 0.70 |
| episodes8 | 2.91 | 0.75 | unexplained | 97 | 0.10 | 0.10 | 0.47 | 0.16 |
| window8 | 8.85 | 0.74 | slow | 21 | 0.24 | 0.21 | 0.12 | 0.14 |
| window8 | 8.85 | 0.74 | explained | 74 | 0.66 | 0.68 | 0.44 | 0.72 |
| window8 | 8.85 | 0.74 | unexplained | 97 | 0.10 | 0.11 | 0.44 | 0.13 |
