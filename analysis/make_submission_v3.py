"""Build submission v3 from the tuned ensemble (best model so far).

Model = median impute -> SelectKBest(mutual_info, k=500) ->
Voting(GB + tuned XGB + tuned LGBM + ExtraTrees).
Tuned params come from the RandomizedSearchCV in analysis/exp_tuning.py.
Robust 5x3 CV R^2 = 0.509 (best measured); single-split ~0.52-0.53.

Writes output/submission_v3.csv and compares to output/submission.csv (v1).
Run: python analysis/make_submission_v3.py
"""

from __future__ import annotations

import sys
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
RS = 0

XGB_BEST = dict(subsample=0.7, reg_lambda=1.0, n_estimators=800, min_child_weight=5,
                max_depth=4, learning_rate=0.03, colsample_bytree=0.5)
LGBM_BEST = dict(subsample_freq=1, subsample=0.6, reg_lambda=0.5, num_leaves=15,
                 n_estimators=1500, min_child_samples=30, learning_rate=0.01,
                 colsample_bytree=0.3)


def build_pipeline() -> Pipeline:
    mi = partial(mutual_info_regression, random_state=RS)
    ensemble = VotingRegressor([
        ("gb", GradientBoostingRegressor(learning_rate=0.05, n_estimators=400,
                                         max_depth=3, subsample=0.8, random_state=RS)),
        ("xgb", XGBRegressor(**XGB_BEST, random_state=RS, verbosity=0, n_jobs=-1)),
        ("lgbm", LGBMRegressor(**LGBM_BEST, random_state=RS, verbose=-1, n_jobs=-1)),
        ("et", ExtraTreesRegressor(400, random_state=RS, n_jobs=-1)),
    ], n_jobs=-1)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(mi, k=500)),
        ("model", ensemble),
    ])


def main() -> None:
    X_train_df = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test_df = pd.read_csv(DATA_DIR / "X_test.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv")["y"].to_numpy()
    feats = [c for c in X_train_df.columns if c != "id"]

    model = build_pipeline()
    model.fit(X_train_df[feats].to_numpy(), y)
    preds = model.predict(X_test_df[feats].to_numpy())

    OUTPUT_DIR.mkdir(exist_ok=True)
    sub = pd.DataFrame({"id": X_test_df["id"].astype(int), "y": preds})
    sub.to_csv(OUTPUT_DIR / "submission_v3.csv", index=False)
    print(f"Wrote {OUTPUT_DIR / 'submission_v3.csv'} ({len(sub)} rows)")

    v1_path = OUTPUT_DIR / "submission.csv"
    if v1_path.exists():
        v1 = pd.read_csv(v1_path)["y"].to_numpy()
        diff = np.abs(preds - v1)
        print("\n--- v3 vs v1 (attempt 1 submission) ---")
        print(f"Pearson correlation: {np.corrcoef(preds, v1)[0, 1]:.4f}")
        print(f"Mean |diff|: {diff.mean():.3f}   max |diff|: {diff.max():.3f}")
        print(f"Predictions differing by > 1.0: {int((diff > 1.0).sum())} / {len(diff)}")


if __name__ == "__main__":
    main()
