"""Command line entry point: ``tourism-train --data Tourism.xlsx`` (or ``--synthetic``)."""

from __future__ import annotations

import argparse

from .data import load_raw, make_synthetic
from .train import run, save


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Path to Tourism.xlsx (sheet 'Tourism') or a CSV")
    src.add_argument("--synthetic", action="store_true", help="Use generated demo data")
    ap.add_argument("--models", nargs="*", help="Subset: logreg random_forest hist_gb xgboost stacking")
    ap.add_argument("--beta", type=float, default=1.0, help="F-beta for threshold tuning")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", default="artifacts", help="Output directory")
    a = ap.parse_args(argv)

    df = make_synthetic() if a.synthetic else load_raw(a.data)
    res = run(df, a.models, a.beta, a.cv, n_jobs=a.n_jobs)
    print(res.cv_table.round(3).to_string(index=False))
    print(f"\nBest: {res.best_name}\nHold-out: { {k: round(v, 3) for k, v in res.test_metrics.items()} }")
    save(res, a.out)
    print(f"Saved to {a.out}/")


if __name__ == "__main__":
    main()
