import numpy as np

from tourism_classifier.data import FEATURES, TARGET, clean, make_synthetic, split_xy
from tourism_classifier.train import best_threshold, run


def test_clean_fixes_quirks():
    df = clean(make_synthetic(500))
    assert "CustomerID" not in df
    assert "Fe Male" not in set(df["Gender"])
    assert df["DurationOfPitch"].max() <= 100


def test_clean_leaves_missing_for_pipeline():
    df = clean(make_synthetic(500))
    assert df["Age"].isna().any()  # imputation must happen inside CV folds
    X, y = split_xy(df)
    assert list(X.columns) == FEATURES and set(y) <= {0, 1}
    assert TARGET not in X


def test_best_threshold_separable():
    y = np.array([0, 0, 0, 1, 1]); p = np.array([.1, .2, .3, .7, .9])
    assert 0.3 < best_threshold(y, p) <= 0.7


def test_end_to_end_beats_chance():
    res = run(make_synthetic(1500), models=["logreg", "hist_gb"], cv=3)
    assert res.test_metrics["roc_auc"] > 0.65
    assert res.model.predict_proba(make_synthetic(10).pipe(clean).pipe(lambda d: d[FEATURES])).shape == (10, 2)
