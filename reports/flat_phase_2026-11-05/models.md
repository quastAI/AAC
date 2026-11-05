# Run comparison

## Training

| run | level | erank / dim | pred / copy | action lift | useful horizon | VQ perplexity | IDM accuracy | critical | warnings |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 1 | 101 / 192 | 0.201 | 0.862 | 10 | - | 0.927 | 0 | 1 |
| V3 | 1 | 158 / 192 | 0.707 | 0.418 | 10 | - | 0.928 | 0 | 1 |
| A3 | 1 | 92 / 192 | 0.0533 | 0.973 | 10 | - | 0.922 | 0 | 1 |
| A4 | 1 | 167 / 192 | 0.236 | 0.814 | 10 | - | 0.93 | 0 | 0 |

## Probing (score of fast / medium / slow state; untrained encoder in brackets)

| run | level | fast | medium | slow |
|---|---|---|---|---|
| B1 | 1 | 0.52 (0.62) | 0.9 (0.7) | 0.93 (0.81) |
| V3 | 1 | 0.44 (0.62) | 0.85 (0.7) | 0.82 (0.81) |

