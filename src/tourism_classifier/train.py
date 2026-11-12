"""Cross-validated model comparison, threshold tuning and held-out evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .data import clean, split_xy
from .models import SEED, build_models


def best_threshold(y_true, proba, beta: float = 1.0) -> float:
    """Probability cut-off maximising F-beta (beta>1 favours recall / wider outreach)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    return float(thr[np.argmax(f)])


def score(y_true, proba, threshold: float, beta: float = 1.0) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    rep = classification_report(y_true, pred, output_dict=True, zero_division=0)["1"]
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": rep["precision"], "recall": rep["recall"], "f1": rep["f1-score"],
        f"f{beta:g}": fbeta_score(y_true, pred, beta=beta), "threshold": threshold,
    }


@dataclass
class Result:
    cv_table: pd.DataFrame
    best_name: str
    test_metrics: dict[str, float]
    confusion: list[list[int]]
    model: object


def run(df_raw: pd.DataFrame, models: list[str] | None = None, beta: float = 1.0,
        cv: int = 5, test_size: float = 0.25, n_jobs: int = 1) -> Result:
    """Compare models by out-of-fold PR-AUC, then refit the winner and score the hold-out set."""
    X, y = split_xy(clean(df_raw))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED)
    candidates = build_models(pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    if models:
        candidates = {k: v for k, v in candidates.items() if k in models}
    skf = StratifiedKFold(cv, shuffle=True, random_state=SEED)

    rows, oof = [], {}
    for name, est in candidates.items():
        p = cross_val_predict(clone(est), X_tr, y_tr, cv=skf, method="predict_proba",
                              n_jobs=n_jobs)[:, 1]
        oof[name] = p
        rows.append({"model": name, **score(y_tr, p, best_threshold(y_tr, p, beta), beta)})
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)

    best = table.loc[0, "model"]
    thr = float(table.loc[0, "threshold"])
    final = clone(candidates[best]).fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]
    metrics = score(y_te, p_te, thr, beta)
    cm = confusion_matrix(y_te, (p_te >= thr).astype(int)).tolist()
    final.threshold_ = thr
    return Result(table, best, metrics, cm, final)


def save(result: Result, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": result.model, "threshold": result.model.threshold_}, out / "model.joblib")
    result.cv_table.to_csv(out / "cv_results.csv", index=False)
    (out / "test_metrics.json").write_text(json.dumps(
        {"model": result.best_name, "metrics": result.test_metrics,
         "confusion_matrix": result.confusion}, indent=2))
