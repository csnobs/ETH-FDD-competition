"""Systematic experiments to push CV R^2 well above 0.5.

Linear models on the cleaned data plateau around 0.35, so the main lever is a
non-linear model. Results are appended live to analysis/experiment_log.txt.

Run with:  python -u analysis/experiments.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.cleaning import Winsorizer

DATA_DIR = "data"
LOG = Path("analysis/experiment_log.txt")
CV = KFold(n_splits=5, shuffle=True, random_state=42)
RS = 0


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def ev(name: str, pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
    s = cross_val_score(pipe, X, y, cv=CV, scoring="r2", n_jobs=-1)
    log(f"{name:<52} R^2 = {s.mean():.4f} +/- {s.std():.4f}")
    return s.mean()


def linear_pipe(k, model) -> Pipeline:
    """Full cleaning for linear models (needs impute+scale)."""
    mi = partial(mutual_info_regression, random_state=RS)
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("winsorize", Winsorizer(1.0, 99.0)),
        ("scale", RobustScaler()),
    ]
    if k is not None:
        steps.append(("select", SelectKBest(mi, k=k)))
    steps.append(("model", model))
    return Pipeline(steps)


def tree_pipe(model, impute: bool = False) -> Pipeline:
    """Trees don't need scaling; HistGBM even handles NaN natively."""
    steps: list[tuple[str, object]] = []
    if impute:
        steps.append(("impute", SimpleImputer(strategy="median")))
    steps.append(("model", model))
    return Pipeline(steps)


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()

    log("\n===== EXPERIMENT RUN =====")
    log("-- reference (linear) --")
    ev("median + k=100 + Ridge(10)", linear_pipe(100, Ridge(10)), X, y)

    log("-- non-linear models (native NaN, no scaling) --")
    ev("HistGBM (defaults, raw NaN)",
       tree_pipe(HistGradientBoostingRegressor(random_state=RS)), X, y)
    ev("RandomForest(300) + median",
       tree_pipe(RandomForestRegressor(n_estimators=300, random_state=RS, n_jobs=-1),
                 impute=True), X, y)
    ev("ExtraTrees(400) + median",
       tree_pipe(ExtraTreesRegressor(n_estimators=400, random_state=RS, n_jobs=-1),
                 impute=True), X, y)

    log("-- HistGBM tuning --")
    ev("HistGBM lr=0.05, leaves=63, iter=500",
       tree_pipe(HistGradientBoostingRegressor(
           learning_rate=0.05, max_leaf_nodes=63, max_iter=500,
           l2_regularization=1.0, random_state=RS)), X, y)
    ev("HistGBM lr=0.03, leaves=31, iter=800",
       tree_pipe(HistGradientBoostingRegressor(
           learning_rate=0.03, max_leaf_nodes=31, max_iter=800,
           l2_regularization=1.0, early_stopping=True, random_state=RS)), X, y)
    ev("GradBoost(lr=0.05, n=400, depth=3) + median",
       tree_pipe(GradientBoostingRegressor(
           learning_rate=0.05, n_estimators=400, max_depth=3, subsample=0.8,
           random_state=RS), impute=True), X, y)

    log("===== END RUN =====\n")


if __name__ == "__main__":
    main()
