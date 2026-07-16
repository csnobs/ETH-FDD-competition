"""Round 5: locate the k peak, average diverse models, and use repeated CV.

Round 4 best: GB + top-500 features = 0.515 (but within fold noise). Here we:
  1. fine-sweep k around 500 for GB
  2. average GB + XGB (+ ExtraTrees) on the selected features
  3. re-estimate the top candidates with RepeatedKFold to shrink the error bars

Results append live to analysis/experiment_log.txt.

Run with:  python -u analysis/experiments5.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def gb() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(learning_rate=0.05, n_estimators=400,
                                     max_depth=3, subsample=0.8, random_state=RS)


def xgb() -> XGBRegressor:
    return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=3,
                        subsample=0.8, colsample_bytree=0.3, reg_lambda=2.0,
                        random_state=RS, verbosity=0)


def et() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=400, random_state=RS, n_jobs=-1)


def pipe(model, k):
    mi = partial(mutual_info_regression, random_state=RS)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(mi, k=k)),
        ("model", model),
    ])


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()

    log("\n===== EXPERIMENT RUN 5 =====")

    log("-- fine k sweep + GB --")
    for k in [400, 500, 600, 700, 828]:
        ev(f"kbest={k} + GB", pipe(gb(), k), X, y)

    log("-- averaging ensembles (k=500) --")
    vote2 = VotingRegressor([("gb", gb()), ("xgb", xgb())])
    vote3 = VotingRegressor([("gb", gb()), ("xgb", xgb()), ("et", et())])
    ev("Voting(GB+XGB) k=500", pipe(vote2, 500), X, y)
    ev("Voting(GB+XGB+ET) k=500", pipe(vote3, 500), X, y)

    log("-- robust estimate: RepeatedKFold (5x3) on top candidates --")
    ev("[RCV] GB k=500", pipe(gb(), 500), X, y, cv=RCV)
    ev("[RCV] Voting(GB+XGB) k=500", pipe(vote2, 500), X, y, cv=RCV)
    ev("[RCV] Voting(GB+XGB+ET) k=500", pipe(vote3, 500), X, y, cv=RCV)

    log("===== END RUN 5 =====\n")


if __name__ == "__main__":
    main()
