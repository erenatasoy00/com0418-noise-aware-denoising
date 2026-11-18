# Noise-type classifier and noise-level estimator

Thresholds fitted on Set12 (validation, 108 images): impulse purity > 0.544 → salt & pepper; brightness slope > 0.327 → speckle; otherwise Gaussian.

## Test: BSD68, 9 seen conditions (612 images) — accuracy 99.8%

| true \ predicted | gaussian | sp | speckle |
|---|---|---|---|
| gaussian | 203 | 0 | 1 |
| sp | 0 | 204 | 0 |
| speckle | 0 | 0 | 204 |

| class | precision | recall | F1 |
|---|---|---|---|
| gaussian | 1.000 | 0.995 | 0.998 |
| sp | 1.000 | 1.000 | 1.000 |
| speckle | 0.995 | 1.000 | 0.998 |

## Robustness: BSD68, unseen levels σ=35 / 15% / var 0.07 (204 images) — accuracy 100.0%

| true \ predicted | gaussian | sp | speckle |
|---|---|---|---|
| gaussian | 68 | 0 | 0 |
| sp | 0 | 68 | 0 |
| speckle | 0 | 0 | 68 |

| class | precision | recall | F1 |
|---|---|---|---|
| gaussian | 1.000 | 1.000 | 1.000 |
| sp | 1.000 | 1.000 | 1.000 |
| speckle | 1.000 | 1.000 | 1.000 |

## Noise-level estimation on BSD68 (true type given)

| noise | true level | estimated (mean ± std) |
|---|---|---|
| gaussian | 15 | 15.649 ± 1.454 |
| gaussian | 25 | 24.449 ± 1.766 |
| gaussian | 50 | 44.958 ± 3.317 |
| sp | 0.05 | 0.063 ± 0.041 |
| sp | 0.1 | 0.113 ± 0.039 |
| sp | 0.2 | 0.211 ± 0.035 |
| speckle | 0.02 | 0.026 ± 0.006 |
| speckle | 0.05 | 0.056 ± 0.007 |
| speckle | 0.1 | 0.110 ± 0.010 |
| gaussian | 35 | 33.053 ± 2.385 |
| sp | 0.15 | 0.162 ± 0.037 |
| speckle | 0.07 | 0.077 ± 0.008 |
