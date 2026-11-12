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


def test_schema_validation_message():
    import pytest
    with pytest.raises(ValueError, match="missing required columns"):
        clean(make_synthetic(50).drop(columns=["Passport"]))


def test_profit_threshold_contacts_only_profitable():
    from tourism_classifier.train import profit_threshold
    y = np.array([0] * 80 + [1] * 20)
    p = np.concatenate([np.linspace(0, .5, 80), np.linspace(.4, 1, 20)])
    t_cheap = profit_threshold(y, p, cost_contact=0.1, profit_purchase=10)
    t_dear = profit_threshold(y, p, cost_contact=9, profit_purchase=10)
    assert t_cheap < t_dear


def test_cli_train_and_predict(tmp_path):
    from tourism_classifier.cli import main, predict_main
    main(["--synthetic", "--models", "logreg", "--cv", "3", "--out", str(tmp_path),
          "--cost-contact", "1", "--profit-purchase", "8"])
    assert (tmp_path / "report.md").exists() and (tmp_path / "model.joblib").exists()
    csv = tmp_path / "new.csv"
    make_synthetic(40, seed=3).drop(columns=["ProdTaken"]).to_csv(csv, index=False)
    predict_main(["--model", str(tmp_path / "model.joblib"), "--data", str(csv),
                  "--out", str(tmp_path / "scored.csv")])
    assert (tmp_path / "scored.csv").exists()
