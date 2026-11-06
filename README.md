# Regime-Aware Equity Return Forecasting

Departmental Honors Project, University of Kansas (2026–2027).

I am testing whether macroeconomic and market regimes help forecast one-month-ahead U.S. equity-market excess returns. The first stage builds a monthly dataset from historical FRED/ALFRED snapshots and compares straightforward forecasting methods. A Hidden Markov Model (HMM), regime-aware models, and political-risk variables are planned for later stages; they are **not** part of the current results.

## Current results

The monthly sample runs from January 2005 to August 2026 (260 forecast origins). With a 120-month initial training window, the baseline comparison contains 140 out-of-sample forecasts.

| Model | RMSE (percentage points) | OOS R² vs. historical mean |
| --- | ---: | ---: |
| Historical mean | 4.466 | 0.000 |
| OLS | 4.889 | -0.198 |
| Ridge | 4.751 | -0.131 |
| Elastic Net | 4.589 | -0.056 |
| Equal-weight combination | 4.728 | -0.121 |

The historical mean is the strongest baseline in this sample. The results do not yet establish that regime information improves forecasts.

## Repository structure

- `src/pit_pipeline.py` — FRED/ALFRED snapshots, transformations, and monthly as-of dataset
- `src/milestone2_models.py` — feature engineering, walk-forward regressions, and model evaluation
- `data/features_asof.csv` — saved monthly input data so the baseline can run without downloading data
- `results/` — saved baseline performance, descriptive statistics, and walk-forward predictions
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores API keys, caches, and local outputs

## Run the baseline

Install the requirements:

```bash
python -m pip install -r requirements.txt
```

Reproduce the baseline with the included input:

```bash
python src/milestone2_models.py --input data/features_asof.csv --out output/milestone2 --min-train 120
```

To download the historical macro snapshots again, set the `FRED_API_KEY` environment variable **locally**, then run:

```bash
python src/pit_pipeline.py --start 2005-01 --end 2026-08 --series UNRATE,CPIAUCSL,INDPRO,PAYEMS,FEDFUNDS,GS10 --include-realized-target --out output/real
```

The pipeline caches API responses locally to avoid repeating requests. Never commit the API key or the raw cache.

## Data notes

Predictors are selected from FRED/ALFRED snapshots using forecast-date vintage parameters. Kenneth French's monthly market excess return (`Mkt-RF`) is used for the realized target, not an actual S&P 500 return series. The saved snapshots were cross-checked against cached responses, but this does **not** independently prove each variable's original publication time. Political-risk indices with retrospectively constructed histories also require separate availability checks. A conservative one-month return-label delay was tested as a robustness check.

This repository contains the ongoing research code and results, not the faculty submission report.