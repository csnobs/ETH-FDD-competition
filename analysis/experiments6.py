"""Round 6: feature engineering to add signal beyond the ~0.50 model ceiling.

We test three cheap, high-value ideas on top of the best pipeline
(Voting(GB+XGB+ET) on top-500 MI features):
  1. missingness features (indicator per column + row NaN count)
  2. row-level aggregates (mean/std/min/max across features)
  3. pairwise interactions among the top-MI features

Exploration uses 5-fold; the winner is re-checked with RepeatedKFold.
Results append live to analysis/experiment_log.txt.

Run with:  python -u analysis/experiments6.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

DATA_DIR = "data"
LOG = Path("analysis/experiment_log.txt")
CV = KFold(n_splits=5, shuffle=True, random_state=42)
RCV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
RS = 0


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def ev(name: str, model, X, y, cv=CV) -> float:
    s = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    log(f"{name:<48} R^2 = {s.mean():.4f} +/- {s.std():.4f}")
    return s.mean()


class RowAggregates(BaseEstimator, TransformerMixin):
    """Append per-row summary stats (computed ignoring NaN) to the matrix."""

    def fit(self, X: npt.ArrayLike, y: object = None) -> "RowAggregates":
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        X = np.asarray(X, dtype=float)
        aggs = np.column_stack([
            np.nanmean(X, axis=1),
            np.nanstd(X, axis=1),
            np.nanmin(X, axis=1),
            np.nanmax(X, axis=1),
            np.isnan(X).sum(axis=1),
        ])
        return np.hstack([X, aggs])


def voting() -> VotingRegressor:
    gb = GradientBoostingRegressor(learning_rate=0.05, n_estimators=400,
                                   max_depth=3, subsample=0.8, random_state=RS)
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=3,
                       subsample=0.8, colsample_bytree=0.3, reg_lambda=2.0,
                       random_state=RS, verbosity=0)
    et = ExtraTreesRegressor(n_estimators=400, random_state=RS, n_jobs=-1)
    return VotingRegressor([("gb", gb), ("xgb", xgb), ("et", et)])


def base_pipe(k=500):
    mi = partial(mutual_info_regression, random_state=RS)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(mi, k=k)),
        ("model", voting()),
    ])


def missing_pipe(k=500):
    mi = partial(mutual_info_regression, random_state=RS)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("select", SelectKBest(mi, k=k)),
        ("model", voting()),
    ])


def agg_pipe(k=505):
    mi = partial(mutual_info_regression, random_state=RS)
    return Pipeline([
        ("agg", RowAggregates()),
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(mi, k=k)),
        ("model", voting()),
    ])


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()

    log("\n===== EXPERIMENT RUN 6 (feature engineering) =====")
    log("-- 5-fold exploration --")
    ev("base Voting k=500", base_pipe(), X, y)
    ev("+ missingness indicators (k=500)", missing_pipe(500), X, y)
    ev("+ missingness indicators (k=700)", missing_pipe(700), X, y)
    ev("+ row aggregates (k=505)", agg_pipe(505), X, y)

    log("-- robust RepeatedKFold on base + row aggregates --")
    ev("[RCV] base Voting k=500", base_pipe(), X, y, cv=RCV)
    ev("[RCV] + row aggregates k=505", agg_pipe(505), X, y, cv=RCV)

    log("===== END RUN 6 =====\n")


if __name__ == "__main__":
    main()
