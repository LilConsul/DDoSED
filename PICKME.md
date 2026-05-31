# DDoSED

Lightweight ML-based DDoS detection experiments using windowed traffic features.

Development dependencies (for pre-commit hooks, linters, etc.)

```bash
uv sync --extra dev
```

How to create models
```bash
uv run python -m src.train
```


## What this adds

- Dataset shuffling + train/test split.
- Windowing over 1..N rows with aggregation (mean/std/min/max for numeric, mode for categorical).
- Training script for lightweight models with metrics saved to reports.

## Quick start

```powershell
uv sync
uv run python -m src.demo
```

## Train models

Attack score (1-10) regression:

```powershell
uv run python -m src.train --max-window-size 100 --step 1
```

Reports are written to `reports/` and trained models to `models/`.

## Notes

- Windowing is configurable via `--max-window-size` and `--step`.
- Attack score is computed as `round((attacks / total_rows) * 10)` and clamped to 1..10.
- Use `--no-shuffle-before-windowing` if you want to preserve the original order.
- Select model keys with `--models`. Available keys: `lin_reg`, `ridge`, `sgd_reg`, `rf`, `rf_depth10`, `extra_trees`.
- For large runs, consider `--sample-size` to speed up experiments.

## Interpreting metrics

- MAE (Mean Absolute Error): Average absolute difference between prediction and true value. Lower is better. In your 1–10 score task, MAE 0.26 means predictions are off by ~0.26 points on average.
- RMSE (Root Mean Squared Error): Like MAE but penalizes larger errors more. Lower is better.
- R2 (Coefficient of Determination): How much variance in the target your model explains (vs. a simple baseline). Higher is better; 1.0 is perfect, 0.0 means no better than predicting the mean; negative is worse than the mean baseline.
- Single-window prediction time (seconds): Time to score one preprocessed window using the trained model (no preprocessing included).

## Best model from current run

- Best MAE (lower is better): `extra_trees`, window size 469
  - MAE 0.2073, RMSE 0.2738, R2 -0.2798
- Best RMSE (lower is better): `extra_trees`, window size 468
  - RMSE 0.2301, MAE 0.2125, R2 0.0
- Best R2 (higher is better): `sgd_reg`, window size 358
  - R2 0.3255, MAE 0.2852, RMSE 0.3710

## Interpreting the best results with examples

- Best MAE: `extra_trees`, window 469 (MAE 0.2073)
  - Meaning: On average, the predicted 1-10 score is off by about 0.21 points.
  - Example: If the true score for a window is 7, the prediction is often around 6.8-7.2.
- Best RMSE: `extra_trees`, window 468 (RMSE 0.2301)
  - Meaning: Large errors are rare. RMSE being close to MAE suggests few big mistakes.
  - Example: If most windows are predicted within +/-0.3, RMSE stays low; a few big misses would push RMSE up faster than MAE.
- Best R2: `sgd_reg`, window 358 (R2 0.3255)
  - Meaning: This model explains about 32.5% of the variability in the attack score.
  - Example: If the score changes from 2 to 9 across windows, this model captures some of that variation, but still misses a lot.

## How good are these results?

- The MAE/RMSE are low relative to a 1–10 scale, which suggests predictions are often close to the target score.
- The R2 values are modest (around 0.30 at best), which suggests the models explain some variance but still miss a lot.
- Net: these results look usable for a lightweight baseline, but there is room for improvement (feature engineering, different windowing, and model tuning could raise R2).
