"""Round 3: stronger boosting libraries (LightGBM, XGBoost) to push R^2 >> 0.5.

Both handle missing values natively and need no scaling. We go incrementally:
defaults -> regularised tuning -> feature selection.
Results append live to analysis/experiment_log.txt.

Run with:  python -u analysis/experiments3.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightgbm import LGBMRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
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


def sel(model, k):
    mi = partial(mutual_info_regression, random_state=RS)
    return Pipeline([("select", SelectKBest(mi, k=k)), ("model", model)])


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()

    log("\n===== EXPERIMENT RUN 3 (LightGBM / XGBoost) =====")

    log("-- defaults (native NaN) --")
    ev("LightGBM defaults", LGBMRegressor(random_state=RS, verbose=-1), X, y)
    ev("XGBoost defaults", XGBRegressor(random_state=RS, verbosity=0), X, y)

    log("-- LightGBM regularised tuning --")
    ev("LGBM n=500,lr=.05,leaves=31,sub=.8,col=.8",
       LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                     reg_lambda=1.0, random_state=RS, verbose=-1), X, y)
    ev("LGBM n=1000,lr=.02,leaves=15,col=.5,mcs=30",
       LGBMRegressor(n_estimators=1000, learning_rate=0.02, num_leaves=15,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                     min_child_samples=30, reg_lambda=1.0, random_state=RS, verbose=-1), X, y)
    ev("LGBM n=800,lr=.03,leaves=31,col=.3",
       LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
                     reg_lambda=2.0, random_state=RS, verbose=-1), X, y)

    log("-- XGBoost regularised tuning --")
    ev("XGB n=500,lr=.05,d=4,sub=.8,col=.5,l2=1",
       XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                    subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                    random_state=RS, verbosity=0), X, y)
    ev("XGB n=800,lr=.03,d=3,sub=.8,col=.3,l2=2",
       XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=3,
                    subsample=0.8, colsample_bytree=0.3, reg_lambda=2.0,
                    random_state=RS, verbosity=0), X, y)

    log("-- feature selection + best LGBM --")
    best_lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
                              reg_lambda=2.0, random_state=RS, verbose=-1)
    for k in [200, 300, 400]:
        ev(f"SelectKBest(k={k}) + LGBM", sel(best_lgbm, k), X, y)

    log("===== END RUN 3 =====\n")


if __name__ == "__main__":
    main()
