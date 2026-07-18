"""Experiment 2b: outlier-row removal (Isolation Forest vs LOF), separate run.

Isolates the effect of removing anomalous TRAINING rows. Median imputation is
kept (NOT KNN) so this experiment is independent of Experiment 2a. Detectors are
fit on the processed training fold only (leak-safe); flagged rows are dropped
from training; the validation fold is always scored in full.

Bake-off: baseline (no removal) vs {IsolationForest, LOF} x {0.005, 0.01}
contamination, on the two strongest models (GB k=500, Voting k=500).

Run with:  python -u analysis/exp_outliers.py
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
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

from src.cleaning import Winsorizer

DATA_DIR = "data"
LOG = Path("analysis/experiment_log.txt")
CV = KFold(n_splits=5, shuffle=True, random_state=42)
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
    order = np.argsort(mi)[::-1]
    return X_tr, X_val, order


def keep_mask(name: str, contamination: float, X_tr: np.ndarray) -> np.ndarray:
    if name == "none":
        return np.ones(X_tr.shape[0], dtype=bool)
    if name == "iforest":
        det = IsolationForest(contamination=contamination, n_estimators=200,
                              random_state=RS)
        pred = det.fit_predict(X_tr)
    else:  # lof
        det = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        pred = det.fit_predict(X_tr)
    return pred != -1


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()
    folds = list(CV.split(X, y))

    # Preprocess once per fold (shared across all removal configs & models).
    cache = []
    for tr, va in folds:
        cache.append((*preprocess_fold(X[tr], X[va], y[tr]), y[tr], y[va]))

    configs = [("none", 0.0), ("iforest", 0.005), ("iforest", 0.01),
               ("lof", 0.005), ("lof", 0.01)]

    log("\n===== EXPERIMENT 2b: outlier-row removal (single 5-fold) =====")
    for model_name, (make, k) in MODELS.items():
        log(f"\n-- {model_name} --")
        base = None
        for det_name, contam in configs:
            scores, removed = [], []
            for X_tr, X_val, order, y_tr, y_val in cache:
                mask = keep_mask(det_name, contam, X_tr)
                removed.append(int((~mask).sum()))
                cols = order[:k]
                m = make()
                m.fit(X_tr[mask][:, cols], y_tr[mask])
                scores.append(r2_score(y_val, m.predict(X_val[:, cols])))
            mean = float(np.mean(scores))
            label = "no removal" if det_name == "none" else f"{det_name} c={contam}"
            if base is None:
                base = mean
            delta = "" if det_name == "none" else f"  delta={mean - base:+.4f}"
            better = "  <-- better" if det_name != "none" and mean > base else ""
            log(f"   {label:<18} R^2 = {mean:.4f}  (~{int(np.mean(removed))} rows removed){delta}{better}")
    log("===== END EXPERIMENT 2b =====\n")


if __name__ == "__main__":
    main()
