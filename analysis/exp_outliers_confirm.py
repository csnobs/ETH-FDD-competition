"""Experiment 2b confirmation: robust (5x3) CV for iForest row-removal.

The exploration winner was IsolationForest(contamination=0.005). We confirm it
against no-removal for GB k=500 and the champion Voting k=500 with RepeatedKFold,
because the single-split gain (~+0.005) is within fold noise.

Run with:  python -u analysis/exp_outliers_confirm.py
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
    IsolationForest,
    VotingRegressor,
)
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from src.cleaning import Winsorizer

DATA_DIR = "data"
LOG = Path("analysis/experiment_log.txt")
RCV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
RS = 0


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def gb() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(learning_rate=0.05, n_estimators=400,
                                     max_depth=3, subsample=0.8, random_state=RS)


def xgb() -> XGBRegressor:
    return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=3,
                        subsample=0.8, colsample_bytree=0.3, reg_lambda=2.0,
                        random_state=RS, verbosity=0, n_jobs=-1)


def voting() -> VotingRegressor:
    return VotingRegressor([("gb", gb()), ("xgb", xgb()),
                            ("et", ExtraTreesRegressor(400, random_state=RS, n_jobs=-1))],
                           n_jobs=-1)


MODELS = {"GB k=500": (gb, 500), "Voting k=500": (voting, 500)}


def preprocess_fold(X_tr, X_val, y_tr):
    win = Winsorizer(1.0, 99.0).fit(X_tr)
    X_tr, X_val = win.transform(X_tr), win.transform(X_val)
    sc = RobustScaler().fit(X_tr)
    X_tr, X_val = sc.transform(X_tr), sc.transform(X_val)
    imp = SimpleImputer(strategy="median").fit(X_tr)
    X_tr, X_val = imp.transform(X_tr), imp.transform(X_val)
    vt = VarianceThreshold(0.0).fit(X_tr)
    X_tr, X_val = vt.transform(X_tr), vt.transform(X_val)
    mi = mutual_info_regression(X_tr, y_tr, random_state=RS)
    return X_tr, X_val, np.argsort(mi)[::-1]


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()
    cache = [(*preprocess_fold(X[tr], X[va], y[tr]), y[tr], y[va])
             for tr, va in RCV.split(X, y)]

    log("\n===== EXPERIMENT 2b CONFIRM: iForest c=0.005 (RepeatedKFold 5x3) =====")
    for model_name, (make, k) in MODELS.items():
        for remove in [False, True]:
            scores = []
            for X_tr, X_val, order, y_tr, y_val in cache:
                if remove:
                    det = IsolationForest(contamination=0.005, n_estimators=200,
                                          random_state=RS)
                    mask = det.fit_predict(X_tr) != -1
                else:
                    mask = np.ones(X_tr.shape[0], dtype=bool)
                cols = order[:k]
                m = make()
                m.fit(X_tr[mask][:, cols], y_tr[mask])
                scores.append(r2_score(y_val, m.predict(X_val[:, cols])))
            label = "iForest c=0.005" if remove else "no removal"
            log(f"   {model_name:<14} {label:<16} R^2 = {np.mean(scores):.4f} "
                f"+/- {np.std(scores):.4f}")
    log("===== END CONFIRM =====\n")


if __name__ == "__main__":
    main()
