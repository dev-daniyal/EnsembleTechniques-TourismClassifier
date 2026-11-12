"""Command line interface.

    tourism-train   --data Tourism.xlsx [--tune 20] [--beta 2] [--cost-contact 1 --profit-purchase 8]
    tourism-train   --synthetic
    tourism-predict --model artifacts/model.joblib --data new_customers.csv --out scored.csv
"""

from __future__ import annotations

import argparse

import joblib

from .data import FEATURES, clean, load_raw, make_synthetic
from .report import write_report
from .train import run, save


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tourism-train", description="Train and evaluate models.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Tourism.xlsx (sheet 'Tourism') or CSV")
    src.add_argument("--synthetic", action="store_true", help="Use generated demo data")
    ap.add_argument("--models", nargs="*",
                    help="Subset of: logreg random_forest hist_gb xgboost stacking")
    ap.add_argument("--beta", type=float, default=1.0, help="F-beta for threshold tuning")
    ap.add_argument("--cost-contact", type=float, help="Cost of one outreach contact")
    ap.add_argument("--profit-purchase", type=float, help="Profit per purchase; with "
                    "--cost-contact, the threshold maximises expected profit")
    ap.add_argument("--tune", type=int, default=0, metavar="N",
                    help="Randomised hyper-parameter search with N draws (nested CV; slower)")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="artifacts", help="Output directory")
    a = ap.parse_args(argv)

    df = make_synthetic() if a.synthetic else load_raw(a.data)
    res = run(df, a.models, a.beta, a.cv, n_jobs=a.n_jobs, tune_iter=a.tune,
              cost_contact=a.cost_contact, profit_purchase=a.profit_purchase)
    print(res.cv_table.round(3).to_string(index=False))
    print(f"\nBest: {res.best_name}\nHold-out: { {k: round(v, 3) for k, v in res.test_metrics.items()} }")
    save(res, a.out)
    print(f"Report: {write_report(res, a.out)}")


def predict_main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tourism-predict", description="Score customers.")
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="scored.csv")
    a = ap.parse_args(argv)
    bundle = joblib.load(a.model)
    raw = load_raw(a.data)
    if "ProdTaken" not in raw:
        raw["ProdTaken"] = 0  # placeholder so the shared cleaner accepts unlabeled data
    X = clean(raw.drop_duplicates())[FEATURES]
    out = X.copy()
    out["purchase_probability"] = bundle["model"].predict_proba(X)[:, 1]
    out["contact"] = (out["purchase_probability"] >= bundle["threshold"]).astype(int)
    out.sort_values("purchase_probability", ascending=False).to_csv(a.out, index=False)
    print(f"Scored {len(out)} customers, {int(out['contact'].sum())} flagged -> {a.out}")


if __name__ == "__main__":
    main()
