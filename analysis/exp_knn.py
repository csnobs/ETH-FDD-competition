"""Experiment 2a: KNN imputation vs median imputation, models B-G.

Isolates the effect of the imputer: everything else (winsorize outliers,
RobustScaler, SelectKBest feature selection, model hyperparameters) is held
identical. RobustScaler is added so KNN distances are meaningful despite the
~20-orders-of-magnitude feature-scale spread.

Efficiency: for each (imputer, fold) we compute the expensive KNN imputation and
the mutual-info ranking ONCE and reuse them across all models (instead of
recomputing per model). Exploration uses single 5-fold; winners get confirmed
with RepeatedKFold separately.

Run with:  python -u analysis/exp_knn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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


# --- model zoo (name -> (factory, k or None for all features)) ---------------
def gb() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(learning_rate=0.05, n_estimators=400,
                                     max_depth=3, subsample=0.8, random_state=RS)


def xgb() -> XGBRegressor:
    return XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=3,
                        subsample=0.8, colsample_bytree=0.3, reg_lambda=2.0,
                        random_state=RS, verbosity=0, n_jobs=-1)


def et() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(n_estimators=400, random_state=RS, n_jobs=-1)


def voting() -> VotingRegressor:
    return VotingRegressor([("gb", gb()), ("xgb", xgb()), ("et", et())], n_jobs=-1)


MODELS: dict[str, tuple] = {
    "A: Ridge(10) k=100": (lambda: Ridge(alpha=10.0), 100),
    "B: HistGBM": (lambda: HistGradientBoostingRegressor(random_state=RS), None),
    "B: RandomForest(300)": (lambda: RandomForestRegressor(300, random_state=RS, n_jobs=-1), None),
    "B: ExtraTrees(400)": (et, None),
    "C: GradientBoosting": (gb, None),
    "E: LightGBM": (lambda: LGBMRegressor(n_estimators=1000, learning_rate=0.02,
                    num_leaves=15, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.5, min_child_samples=30, reg_lambda=1.0,
                    random_state=RS, verbose=-1, n_jobs=-1), None),
    "E: XGBoost": (xgb, None),
    "F: GB k=500": (gb, 500),
    "F: XGB k=300": (xgb, 300),
    "G: Voting(GB+XGB+ET) k=500": (voting, 500),
}


def preprocess_fold(X_tr, X_val, y_tr, imputer):
    """Winsorize -> scale -> impute -> drop constant -> MI ranking. Fit on train."""
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


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()
    folds = list(CV.split(X, y))

    imputers = {
        "median": SimpleImputer(strategy="median"),
        "knn5": KNNImputer(n_neighbors=5),
    }

    log("\n===== EXPERIMENT 2a: KNN vs MEDIAN imputation (single 5-fold) =====")
    results: dict[str, dict[str, float]] = {}
    for imp_name, imputer in imputers.items():
        log(f"\n-- preprocessing folds with {imp_name} imputation --")
        cache = []
        for i, (tr, va) in enumerate(folds):
            cache.append((*preprocess_fold(X[tr], X[va], y[tr], imputer), y[tr], y[va]))
            log(f"   fold {i} preprocessed")

        log(f"-- scoring models ({imp_name}) --")
        for name, (make, k) in MODELS.items():
            scores = []
            for X_tr, X_val, order, y_tr, y_val in cache:
                kk = k if k is not None else X_tr.shape[1]
                cols = order[:kk]
                m = make()
                m.fit(X_tr[:, cols], y_tr)
                scores.append(r2_score(y_val, m.predict(X_val[:, cols])))
            mean = float(np.mean(scores))
            results.setdefault(name, {})[imp_name] = mean
            log(f"   {name:<32} {imp_name:<7} R^2 = {mean:.4f} +/- {np.std(scores):.4f}")

    log("\n-- SUMMARY: median vs KNN (delta = knn - median) --")
    for name in MODELS:
        med, knn = results[name]["median"], results[name]["knn5"]
        flag = "  <-- KNN better" if knn > med else ""
        log(f"   {name:<32} median={med:.4f}  knn={knn:.4f}  delta={knn - med:+.4f}{flag}")
    log("===== END EXPERIMENT 2a =====\n")


if __name__ == "__main__":
    main()
