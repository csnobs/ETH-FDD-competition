"""Experiment 2a confirmation: robust (5x3) CV for the KNN vs median comparison.

Only the finding worth confirming survived exploration: KNN helped the linear
model (Ridge). We re-check Ridge and, as a tree reference, GB k=500, with
RepeatedKFold to shrink the error bars.

Run with:  python -u analysis/exp_knn_confirm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import RobustScaler

from src.cleaning import Winsorizer

DATA_DIR = "data"
LOG = Path("analysis/experiment_log.txt")
RCV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
RS = 0


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def preprocess_fold(X_tr, X_val, y_tr, imputer):
    win = Winsorizer(1.0, 99.0).fit(X_tr)
    X_tr, X_val = win.transform(X_tr), win.transform(X_val)
    sc = RobustScaler().fit(X_tr)
    X_tr, X_val = sc.transform(X_tr), sc.transform(X_val)
    imp = clone(imputer).fit(X_tr)
    X_tr, X_val = imp.transform(X_tr), imp.transform(X_val)
    vt = VarianceThreshold(0.0).fit(X_tr)
    X_tr, X_val = vt.transform(X_tr), vt.transform(X_val)
    mi = mutual_info_regression(X_tr, y_tr, random_state=RS)
    order = np.argsort(mi)[::-1]
    return X_tr, X_val, order


MODELS = {"A: Ridge(10) k=100": (lambda: Ridge(alpha=10.0), 100),
          "F: GB k=500": (lambda: GradientBoostingRegressor(
              learning_rate=0.05, n_estimators=400, max_depth=3,
              subsample=0.8, random_state=RS), 500)}


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()
    folds = list(RCV.split(X, y))

    log("\n===== EXPERIMENT 2a CONFIRM: KNN vs median (RepeatedKFold 5x3) =====")
    for imp_name, imputer in [("median", SimpleImputer(strategy="median")),
                              ("knn5", KNNImputer(n_neighbors=5))]:
        cache = [(*preprocess_fold(X[tr], X[va], y[tr], imputer), y[tr], y[va])
                 for tr, va in folds]
        for name, (make, k) in MODELS.items():
            scores = []
            for X_tr, X_val, order, y_tr, y_val in cache:
                cols = order[:k]
                m = make()
                m.fit(X_tr[:, cols], y_tr)
                scores.append(r2_score(y_val, m.predict(X_val[:, cols])))
            log(f"   {name:<24} {imp_name:<7} R^2 = {np.mean(scores):.4f} "
                f"+/- {np.std(scores):.4f}")
    log("===== END CONFIRM =====\n")


if __name__ == "__main__":
    main()
