"""Round 4: feature selection to break past the ~0.50 ceiling.

Round 3 showed all boosting libraries plateau ~0.50 on the raw 828 features.
Hypothesis: with 1212 rows and 828 features, most features are noise. Selecting
the informative ones should let the model generalise better.

We sweep SelectKBest(mutual_info) k for the two best models (sklearn GB, XGBoost),
then also try model-based selection via feature importance.
Results append live to analysis/experiment_log.txt.

Run with:  python -u analysis/experiments4.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

DATA_DIR = "data"
LOG = Path("analysis/experiment_log.txt")
CV = KFold(n_splits=5, shuffle=True, random_state=42)
RS = 0


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def ev(name: str, model, X, y) -> float:
    s = cross_val_score(model, X, y, cv=CV, scoring="r2", n_jobs=-1)
    log(f"{name:<52} R^2 = {s.mean():.4f} +/- {s.std():.4f}")
    return s.mean()


def gb() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(learning_rate=0.05, n_estimators=400,
                                     max_depth=3, subsample=0.8, random_state=RS)


def xgb() -> XGBRegressor:
    return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=3,
                        subsample=0.8, colsample_bytree=0.3, reg_lambda=2.0,
                        random_state=RS, verbosity=0)


def kbest(model, k):
    mi = partial(mutual_info_regression, random_state=RS)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(mi, k=k)),
        ("model", model),
    ])


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()

    log("\n===== EXPERIMENT RUN 4 (feature selection) =====")

    log("-- SelectKBest(mutual_info) + sklearn GB --")
    for k in [30, 50, 100, 150, 200, 300, 500]:
        ev(f"kbest={k} + GB", kbest(gb(), k), X, y)

    log("-- SelectKBest(mutual_info) + XGBoost --")
    for k in [50, 100, 200, 300]:
        ev(f"kbest={k} + XGB", kbest(xgb(), k), X, y)

    log("-- model-based selection (GB importance) + GB --")
    sfm = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectFromModel(gb(), threshold="median")),
        ("model", gb()),
    ])
    ev("SelectFromModel(GB, median) + GB", sfm, X, y)

    log("===== END RUN 4 =====\n")


if __name__ == "__main__":
    main()
