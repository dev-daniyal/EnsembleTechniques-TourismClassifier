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

The original notebook is kept in `legacy/` (outputs stripped). The `Tourism.xlsx` dataset is not in the repo.

## Usage

```bash
pip install -e ".[dev]"
tourism-train --data data/Tourism.xlsx        # sheet "Tourism"
tourism-train --synthetic                     # demo data, same schema
pytest
```

Outputs go to `artifacts/` (`model.joblib` with the fitted pipeline and tuned threshold, `cv_results.csv`, `test_metrics.json`).

```python
import joblib
bundle = joblib.load("artifacts/model.joblib")
p = bundle["model"].predict_proba(X)[:, 1]
contact = p >= bundle["threshold"]
```

## Layout

- `src/tourism_classifier/` – `data.py` (load/clean/synthetic), `models.py` (pipelines), `train.py` (CV, thresholding, evaluation), `cli.py`
- `notebooks/tourism_classifier.ipynb` – walkthrough
- `tests/` – unit and end-to-end tests
