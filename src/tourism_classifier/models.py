"""Leakage-free sklearn pipelines for the candidate models."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import CATEGORICAL, NUMERIC

SEED = 24


def preprocessor(scale: bool = False) -> ColumnTransformer:
    """Impute (inside CV folds) and one-hot encode. Missingness flags are kept as features."""
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                     sparse_output=False)),
        ]), CATEGORICAL),
    ], verbose_feature_names_out=False)


def _base_estimators(pos_weight: float) -> dict[str, object]:
    return {
        "logreg": Pipeline([("prep", preprocessor(scale=True)),
                            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "random_forest": Pipeline([("prep", preprocessor()), ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED))]),
        "hist_gb": Pipeline([("prep", preprocessor()), ("clf", HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=400, class_weight="balanced", early_stopping=True,
            random_state=SEED))]),
        "xgboost": Pipeline([("prep", preprocessor()), ("clf", XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="aucpr",
            n_jobs=-1, random_state=SEED))]),
    }


def build_models(pos_weight: float = 4.0) -> dict[str, object]:
    """Candidate models, including a stacked ensemble of the tree-based ones.

    ``pos_weight`` should be (negatives / positives) of the training data.
    """
    models = _base_estimators(pos_weight)
    models["stacking"] = StackingClassifier(
        estimators=[(k, models[k]) for k in ("random_forest", "hist_gb", "xgboost")],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba", cv=5, n_jobs=1,
    )
    return models
