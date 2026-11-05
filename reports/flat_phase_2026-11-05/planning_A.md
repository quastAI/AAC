# Run comparison

## Planning

| run | tasks | task file | success | progress | diamond | progress over baseline | TFLOPs per task | critical | warnings |
|---|---|---|---|---|---|---|---|---|---|
| random | 128 | 19712115 | 0.109 | 0.138 | 0.0625 | - | 0 | 0 | 0 |
| B1 | 128 | 19712115 | 0.18 | 0.236 | 0.125 | 0.0982 | 1.64 | 1 | 2 |
| V3 | 128 | 19712115 | 0.234 | 0.285 | 0.125 | 0.147 | 1.6 | 0 | 1 |

Success by task group (baseline in brackets):

| run | offset_8 | required_1 | required_2 | required_3 | required_4 | required_5 |
|---|---|---|---|---|---|---|
| random | 0.11 | 0.15 | 0 | 0 | 0 | 0 |
| B1 | 0.18 (0.11) | 0.23 (0.15) | 0.091 (0) | 0 (0) | 0 (0) | 0 (0) |
| V3 | 0.23 (0.11) | 0.26 (0.15) | 0.23 (0) | 0.11 (0) | 0 (0) | 0 (0) |

