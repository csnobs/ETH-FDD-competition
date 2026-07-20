"""Step 3: hyperparameter search + tuned ensemble, aiming past 0.52 CV R^2.

Part 1: RandomizedSearchCV for XGBoost and LightGBM on the top-500 features.
Part 2: honest RepeatedKFold(5x3) comparison of the champion ensemble vs a
        tuned ensemble (per-fold feature selection, no leakage).

Run with:  python -u analysis/exp_tuning.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold
from xgboost import XGBRegressor

DATA_DIR = "data"
LOG = Path("analysis/experiment_log.txt")
SEARCH_CV = KFold(n_splits=5, shuffle=True, random_state=42)
RCV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
RS = 0


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def gb() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(learning_rate=0.05, n_estimators=400,
                                     max_depth=3, subsample=0.8, random_state=RS)


def et() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(400, random_state=RS, n_jobs=-1)


def select_topk(X_tr, X_val, y_tr, k):
    imp = SimpleImputer(strategy="median").fit(X_tr)
    X_tr, X_val = imp.transform(X_tr), imp.transform(X_val)
    vt = VarianceThreshold(0.0).fit(X_tr)
    X_tr, X_val = vt.transform(X_tr), vt.transform(X_val)
    mi = mutual_info_regression(X_tr, y_tr, random_state=RS)
    cols = np.argsort(mi)[::-1][:k]
    return X_tr[:, cols], X_val[:, cols]


def robust_cv(make_model, X, y, k=500) -> tuple[float, float]:
    scores = []
    for tr, va in RCV.split(X, y):
        X_tr, X_val = select_topk(X[tr], X[va], y[tr], k)
        m = make_model()
        m.fit(X_tr, y[tr])
        scores.append(r2_score(y[va], m.predict(X_val)))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"]).to_numpy()
    y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"].to_numpy()

    # Precompute top-500 features on full data (for the SEARCH only).
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    mi = mutual_info_regression(X_imp, y, random_state=RS)
    X500 = X_imp[:, np.argsort(mi)[::-1][:500]]

    log("\n===== STEP 3: hyperparameter search + tuned ensemble =====")

    log("-- RandomizedSearchCV: XGBoost --")
    xgb_dist = {
        "n_estimators": [400, 600, 800, 1000, 1200],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "max_depth": [2, 3, 4],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.2, 0.3, 0.5, 0.7],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "min_child_weight": [1, 3, 5],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=RS, verbosity=0, n_jobs=1),
        xgb_dist, n_iter=30, scoring="r2", cv=SEARCH_CV, random_state=RS, n_jobs=-1)
    xgb_search.fit(X500, y)
    log(f"   best XGB search R^2 = {xgb_search.best_score_:.4f}")
    log(f"   best XGB params = {xgb_search.best_params_}")

    log("-- RandomizedSearchCV: LightGBM --")
    lgbm_dist = {
        "n_estimators": [400, 600, 800, 1000, 1500],
        "learning_rate": [0.01, 0.02, 0.03, 0.05],
        "num_leaves": [7, 15, 31],
        "subsample": [0.6, 0.7, 0.8],
        "subsample_freq": [1],
        "colsample_bytree": [0.2, 0.3, 0.5],
        "min_child_samples": [20, 30, 50],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }
    lgbm_search = RandomizedSearchCV(
        LGBMRegressor(random_state=RS, verbose=-1, n_jobs=1),
        lgbm_dist, n_iter=30, scoring="r2", cv=SEARCH_CV, random_state=RS, n_jobs=-1)
    lgbm_search.fit(X500, y)
    log(f"   best LGBM search R^2 = {lgbm_search.best_score_:.4f}")
    log(f"   best LGBM params = {lgbm_search.best_params_}")

    xgb_best = xgb_search.best_params_
    lgbm_best = lgbm_search.best_params_

    def tuned_xgb():
        return XGBRegressor(**xgb_best, random_state=RS, verbosity=0, n_jobs=-1)

    def tuned_lgbm():
        return LGBMRegressor(**lgbm_best, random_state=RS, verbose=-1, n_jobs=-1)

    log("\n-- honest RepeatedKFold(5x3), per-fold selection --")
    champ = lambda: VotingRegressor(
        [("gb", gb()),
         ("xgb", XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=3,
                              subsample=0.8, colsample_bytree=0.3, reg_lambda=2.0,
                              random_state=RS, verbosity=0, n_jobs=-1)),
         ("et", et())], n_jobs=-1)
    m, s = robust_cv(champ, X, y)
    log(f"   champion Voting(GB+XGB+ET)            R^2 = {m:.4f} +/- {s:.4f}")

    tuned = lambda: VotingRegressor(
        [("gb", gb()), ("xgb", tuned_xgb()), ("lgbm", tuned_lgbm()), ("et", et())],
        n_jobs=-1)
    m2, s2 = robust_cv(tuned, X, y)
    log(f"   tuned Voting(GB+XGBt+LGBMt+ET)        R^2 = {m2:.4f} +/- {s2:.4f}")

    m3, s3 = robust_cv(tuned_xgb, X, y)
    log(f"   tuned XGB alone                       R^2 = {m3:.4f} +/- {s3:.4f}")

    log("===== END STEP 3 =====\n")


if __name__ == "__main__":
    main()
