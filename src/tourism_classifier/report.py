"""Plots and a markdown model report for a finished run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from .models import SEED
from .train import Result


def feature_importance(res: Result, n_repeats: int = 10) -> pd.Series:
    """Permutation importance (drop in PR-AUC) on the held-out set, per raw input column."""
    r = permutation_importance(res.model, res.X_test, res.y_test, scoring="average_precision",
                               n_repeats=n_repeats, random_state=SEED, n_jobs=1)
    return pd.Series(r.importances_mean, index=res.X_test.columns).sort_values(ascending=False)


def write_report(res: Result, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = res.X_test, res.y_test
    proba = res.model.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax[0]); ax[0].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax[1]); ax[1].set_title("Precision-Recall")
    CalibrationDisplay.from_predictions(y, proba, n_bins=8, ax=ax[2]); ax[2].set_title("Calibration")
    ConfusionMatrixDisplay.from_predictions(y, (proba >= res.threshold).astype(int), ax=ax[3])
    ax[3].set_title(f"Confusion @ {res.threshold:.2f}")
    fig.tight_layout(); fig.savefig(out / "evaluation.png", dpi=130); plt.close(fig)

    imp = feature_importance(res)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.head(12)[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Drop in PR-AUC when shuffled"); ax.set_title("Permutation importance (hold-out)")
    fig.tight_layout(); fig.savefig(out / "importance.png", dpi=130); plt.close(fig)

    (tn, fp), (fn, tp) = res.confusion
    lines = [
        "# Model report", "", f"**Selected model:** `{res.best_name}`  ",
        f"**Decision threshold:** {res.threshold:.3f}", "",
        "## Cross-validated comparison (training split, out-of-fold)", "",
        res.cv_table.round(3).to_markdown(index=False), "",
        "## Hold-out metrics", "",
        *[f"- {k}: {v:.3f}" for k, v in res.test_metrics.items()], "",
        f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}", "",
        "![evaluation](evaluation.png)", "",
        "## Top drivers", "", imp.head(10).round(4).to_frame("pr_auc_drop").to_markdown(), "",
        "![importance](importance.png)", "",
    ]
    path = out / "report.md"
    path.write_text("\n".join(lines))
    return path
