"""Loading, cleaning and synthetic generation of the Tourism dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "ProdTaken"
ID_COL = "CustomerID"
CATEGORICAL = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation",
]
NUMERIC = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
FEATURES = CATEGORICAL + NUMERIC


def load_raw(path: str | Path, sheet: str = "Tourism") -> pd.DataFrame:
    """Read the raw data from .xlsx (sheet ``Tourism``) or .csv."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Fail early with a readable message if expected columns are missing."""
    missing = [c for c in [*FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, target-free cleaning. Safe to apply before splitting.

    Imputation is deliberately NOT done here: it lives inside the sklearn pipeline so
    statistics are learned from training folds only (the legacy notebook imputed with
    group means that used ``ProdTaken``, leaking the label into the features).
    """
    validate_schema(df)
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    # Typo'd durations such as 126/127 (extra leading digit) -> 26/27.
    df.loc[df["DurationOfPitch"] > 100, "DurationOfPitch"] -= 100
    df = df.drop(columns=[ID_COL], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    for col in CATEGORICAL:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET].astype(int)


def make_synthetic(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Synthetic data with the real schema, quirks and ~18% positive rate (for tests/demo)."""
    rng = np.random.default_rng(seed)
    desig = rng.choice(["Executive", "Manager", "Senior Manager", "AVP", "VP"], n,
                       p=[.38, .37, .15, .08, .02])
    rank = pd.Series(desig).map({"Executive": 0, "Manager": 1, "Senior Manager": 2,
                                 "AVP": 3, "VP": 4}).to_numpy()
    income = rng.normal(17000 + 4500 * rank, 2500).round()
    age = rng.integers(18, 61, n).astype(float)
    passport = rng.binomial(1, .29, n)
    marital = rng.choice(["Married", "Single", "Divorced", "Unmarried"], n, p=[.48, .19, .19, .14])
    pitched = np.array(["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])[
        np.clip(rank + rng.integers(0, 2, n) - (rng.random(n) < .2), 0, 4)]
    followups = rng.integers(1, 7, n).astype(float)
    logit = (-3.2 + 1.9 * passport + .0001 * (income - 25000) + .55 * (marital == "Single") + .45 * (marital == "Unmarried")
             - .03 * (age - 35) + .3 * (followups - 3.5) + .5 * (rank == 3) - .35 * (pitched == "Deluxe")
             + .25 * (pitched == "Basic") + rng.normal(0, .3, n))
    taken = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "CustomerID": np.arange(200000, 200000 + n), "ProdTaken": taken, "Age": age,
        "TypeofContact": rng.choice(["Self Enquiry", "Company Invited"], n, p=[.7, .3]),
        "CityTier": rng.choice([1, 2, 3], n, p=[.65, .03, .32]),
        "DurationOfPitch": rng.integers(5, 37, n).astype(float),
        "Occupation": rng.choice(["Salaried", "Small Business", "Large Business", "Free Lancer"],
                                 n, p=[.48, .42, .09, .01]),
        "Gender": rng.choice(["Male", "Female", "Fe Male"], n, p=[.6, .3, .1]),
        "NumberOfPersonVisiting": rng.integers(1, 6, n), "NumberOfFollowups": followups,
        "ProductPitched": pitched, "PreferredPropertyStar": rng.choice([3., 4., 5.], n),
        "MaritalStatus": marital, "NumberOfTrips": rng.integers(1, 9, n).astype(float),
        "Passport": passport, "PitchSatisfactionScore": rng.integers(1, 6, n),
        "OwnCar": rng.binomial(1, .62, n), "NumberOfChildrenVisiting": rng.integers(0, 4, n).astype(float),
        "Designation": desig, "MonthlyIncome": income,
    })
    for col, frac in [("Age", .05), ("DurationOfPitch", .05), ("MonthlyIncome", .05),
                      ("TypeofContact", .005), ("NumberOfTrips", .03),
                      ("PreferredPropertyStar", .005), ("NumberOfChildrenVisiting", .01),
                      ("NumberOfFollowups", .01)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    df.loc[rng.choice(n, 2, replace=False), "DurationOfPitch"] = [126., 127.]
    return df
