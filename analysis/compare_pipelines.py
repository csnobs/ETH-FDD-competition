"""Cross-validated comparison: baseline vs cleaned pipeline.

Measures 5-fold R^2 to check whether the cleaning pipeline from
``src/cleaning.py`` improves on the mean-impute baseline in ``main.py``.

Run with:  python analysis/compare_pipelines.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.cleaning import build_model_pipeline

DATA_DIR = "data"
CV = KFold(n_splits=5, shuffle=True, random_state=42)


def baseline_pipeline(k: int = 100, random_state: int = 0) -> Pipeline:
    """Reproduces main.py preprocessing: mean impute -> SelectKBest -> OLS."""
    mi = partial(mutual_info_regression, random_state=random_state)
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("select", SelectKBest(mi, k=k)),
            ("model", LinearRegression()),
        ]
    )


def evaluate(name: str, pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> None:
    scores = cross_val_score(pipe, X, y, cv=CV, scoring="r2", n_jobs=-1)
    print(f"{name:<40} R^2 = {scores.mean():.4f} +/- {scores.std():.4f}")


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()

    print("5-fold cross-validated R^2 (higher is better)\n" + "-" * 60)
    evaluate("baseline (mean impute + OLS)", baseline_pipeline(), X, y)
    evaluate("cleaned + OLS", build_model_pipeline(LinearRegression()), X, y)
    evaluate("cleaned + Ridge(alpha=1)", build_model_pipeline(Ridge(alpha=1.0)), X, y)
    evaluate("cleaned + Ridge(alpha=10)", build_model_pipeline(Ridge(alpha=10.0)), X, y)


if __name__ == "__main__":
    main()
