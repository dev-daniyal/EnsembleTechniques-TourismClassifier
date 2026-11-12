# Tourism Package Classifier

Predicts which customers of the fictional "Visit with us" travel company will buy a new Wellness Tourism Package (`ProdTaken`), so marketing can target outreach instead of contacting customers at random. Compares logistic regression, random forest, histogram gradient boosting, XGBoost and a stacked ensemble.

## What changed in v2 (modernisation of the original notebook)

| Problem in the original | Fix |
|---|---|
| **Target leakage**: missing values were imputed with group means that included `ProdTaken`, before the train/test split | Imputation is part of the sklearn `Pipeline` and is fit on training folds only |
| Removed APIs (`pandas_profiling`, `plot_roc_curve`, `BaggingClassifier(base_estimator=…)`), Python 3.8 | Current pandas 2/3, scikit-learn ≥1.4, XGBoost ≥2; Python ≥3.10 |
| Hard-coded `class_weight={0:0.18,1:0.82}`, single split, grid searches re-run by hand | `balanced` weights, stratified CV, model choice by out-of-fold PR-AUC, one-shot hold-out evaluation |
| Fixed 0.5 threshold | Threshold tuned on out-of-fold predictions for F-beta (`--beta`) |
| 300-cell, 11 MB notebook + HTML export, no code reuse | Importable package, CLI, tests, CI, short executed notebook |

The original notebook is kept in `legacy/` (outputs stripped).

## Usage

```bash
pip install -e ".[dev]"
tourism-train --data data/Tourism.xlsx        # sheet "Tourism"; or: tourism-train --synthetic
pytest
```

Useful options:

| Flag | Effect |
|---|---|
| `--tune 20` | Randomised hyper-parameter search (20 draws per model, nested CV, slower) |
| `--beta 2` | Tune the cut-off for F2 (favours recall / wider outreach) |
| `--cost-contact 1 --profit-purchase 8` | Pick the cut-off that maximises expected campaign profit instead |
| `--models xgboost stacking` | Compare only a subset |
| `--n-jobs -1` | Parallelise CV |

Each run writes to `artifacts/`: `model.joblib` (fitted pipeline + tuned threshold), `cv_results.csv`, `test_metrics.json`, and `report.md` with `evaluation.png` (ROC, precision-recall, calibration, confusion matrix) and `importance.png` (permutation importance).

Score new customers (CSV or XLSX with the same columns; `ProdTaken` optional):

```bash
tourism-predict --model artifacts/model.joblib --data new_customers.csv --out scored.csv
```

The output is ranked by purchase probability with a `contact` flag from the tuned threshold.

## Dataset

"Tourism Package Prediction" (4,888 customers, 20 columns), widely available as a public teaching dataset. Put it at `data/Tourism.xlsx` (git-ignored). Without it, `--synthetic` generates data with the same schema and quirks (`Fe Male`, typo'd pitch durations, missing values). **Metrics on synthetic data only demonstrate that the pipeline works.**

## Layout

- `src/tourism_classifier/` – `data.py` (load/clean/synthetic), `models.py` (pipelines), `train.py` (CV, tuning, thresholding), `report.py` (plots/report), `cli.py`
- `notebooks/tourism_classifier.ipynb` – walkthrough
- `tests/` – unit and end-to-end tests
