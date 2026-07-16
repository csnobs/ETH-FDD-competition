"""Round 2: push CV R^2 well above 0.5.

Round 1 winner: GradientBoosting(lr=0.05, n=400, depth=3, subsample=0.8) = 0.502.
Here we tune gradient boosting further and try a stacking ensemble.
Results append live to analysis/experiment_log.txt.

Run with:  python -u analysis/experiments2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

DATA_DIR = "data"
LOG = Path("analysis/experiment_log.txt")
CV = KFold(n_splits=5, shuffle=True, random_state=42)
RS = 0


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def med(model) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


def ev(name: str, pipe, X, y) -> float:
    s = cross_val_score(pipe, X, y, cv=CV, scoring="r2", n_jobs=-1)
    log(f"{name:<52} R^2 = {s.mean():.4f} +/- {s.std():.4f}")
    return s.mean()


def gb(**kw) -> GradientBoostingRegressor:
    params = dict(learning_rate=0.05, n_estimators=400, max_depth=3,
                  subsample=0.8, random_state=RS)
    params.update(kw)
    return GradientBoostingRegressor(**params)


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()

    log("\n===== EXPERIMENT RUN 2 =====")
    log("-- GradientBoosting tuning (all + median impute) --")
    ev("GB base (n=400,d=3,lr=.05,sub=.8)", med(gb()), X, y)
    ev("GB + max_features=sqrt", med(gb(max_features="sqrt")), X, y)
    ev("GB n=600, max_features=sqrt", med(gb(n_estimators=600, max_features="sqrt")), X, y)
    ev("GB n=800, lr=.03, max_features=sqrt",
       med(gb(n_estimators=800, learning_rate=0.03, max_features="sqrt")), X, y)
    ev("GB d=4, n=400, max_features=sqrt", med(gb(max_depth=4, max_features="sqrt")), X, y)
    ev("GB d=2, n=800, max_features=sqrt",
       med(gb(max_depth=2, n_estimators=800, max_features="sqrt")), X, y)
    ev("GB n=600, mf=0.3, sub=.7",
       med(gb(n_estimators=600, max_features=0.3, subsample=0.7)), X, y)

    log("-- HistGBM depth-limited --")
    ev("HistGBM d=3, lr=.05, iter=600, l2=1",
       med(HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=600,
                                         l2_regularization=1.0, random_state=RS)), X, y)

    log("-- Stacking ensemble --")
    estimators = [
        ("gb", gb(n_estimators=600, max_features="sqrt")),
        ("et", ExtraTreesRegressor(n_estimators=400, random_state=RS, n_jobs=-1)),
        ("hgb", HistGradientBoostingRegressor(random_state=RS)),
    ]
    stack = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(),
                              n_jobs=-1, cv=5)
    ev("Stacking(GB+ET+HistGBM)->RidgeCV", med(stack), X, y)

    log("===== END RUN 2 =====\n")


if __name__ == "__main__":
    main()
