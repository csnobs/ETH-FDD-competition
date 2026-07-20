"""Build the second-attempt submission and compare it to the first one.

v2 model = attempt-1 champion (median impute -> SelectKBest k=500 ->
Voting(GB+XGB+ET)) PLUS Isolation-Forest training-row removal (c=0.005) --
the one improvement the second attempt actually justified. Isolation Forest is
fit on a winsorized+scaled+imputed copy (so its distances are meaningful); the
model itself trains on the surviving original rows.

Writes output/submission_v2.csv and prints how far it diverges from
output/submission.csv (attempt 1). Run: python analysis/make_submission_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.cleaning import Winsorizer
from src.model import build_model_pipeline

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
RS = 0


def main() -> None:
    X_train_df = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test_df = pd.read_csv(DATA_DIR / "X_test.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv")["y"].to_numpy()
    feats = [c for c in X_train_df.columns if c != "id"]
    X = X_train_df[feats].to_numpy()
    X_test = X_test_df[feats].to_numpy()

    # --- Isolation Forest detection on a winsorized+scaled+imputed copy -------
    det_space = SimpleImputer(strategy="median").fit_transform(
        RobustScaler().fit_transform(Winsorizer(1.0, 99.0).fit_transform(X))
    )
    mask = IsolationForest(contamination=0.005, n_estimators=200,
                           random_state=RS).fit_predict(det_space) != -1
    print(f"Isolation Forest removed {int((~mask).sum())} of {len(X)} training rows")

    # --- Fit the attempt-1 champion on the surviving rows --------------------
    model = build_model_pipeline()
    model.fit(X[mask], y[mask])
    preds_v2 = model.predict(X_test)

    OUTPUT_DIR.mkdir(exist_ok=True)
    sub_v2 = pd.DataFrame({"id": X_test_df["id"].astype(int), "y": preds_v2})
    sub_v2.to_csv(OUTPUT_DIR / "submission_v2.csv", index=False)
    print(f"Wrote {OUTPUT_DIR / 'submission_v2.csv'} ({len(sub_v2)} rows)")

    # --- Compare with attempt-1 submission -----------------------------------
    v1_path = OUTPUT_DIR / "submission.csv"
    if v1_path.exists():
        v1 = pd.read_csv(v1_path)["y"].to_numpy()
        diff = np.abs(preds_v2 - v1)
        corr = np.corrcoef(preds_v2, v1)[0, 1]
        print("\n--- v2 vs v1 (attempt 1) ---")
        print(f"Pearson correlation: {corr:.4f}")
        print(f"Mean |diff|: {diff.mean():.3f}   median |diff|: {np.median(diff):.3f}")
        print(f"Max |diff|:  {diff.max():.3f}")
        print(f"Predictions differing by > 1.0: {int((diff > 1.0).sum())} / {len(diff)}")
        print(f"Predictions differing by > 2.0: {int((diff > 2.0).sum())} / {len(diff)}")
        print(f"(target y ranges ~42-97, std ~9.7, so judge diffs against that scale)")
    else:
        print("No attempt-1 submission.csv found to compare against.")


if __name__ == "__main__":
    main()
