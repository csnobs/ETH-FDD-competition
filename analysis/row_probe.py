"""Row-level anomaly / perturbed-sample probe."""

from __future__ import annotations

import numpy as np
import pandas as pd

DATA_DIR = "data"
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"])
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv").drop(columns=["id"])

# robust per-column z using median / MAD, so scale-heterogeneity doesn't dominate
med = X_train.median()
mad = (X_train - med).abs().median() * 1.4826
mad = mad.replace(0, np.nan)
z = (X_train - med).abs() / mad

print("=== per-row count of extreme features (robust |z| > 5) ===")
extreme_per_row = (z > 5).sum(axis=1)
print(extreme_per_row.describe().to_string())
print(f"rows with >20 extreme features: {int((extreme_per_row > 20).sum())}")
print(f"rows with >50 extreme features: {int((extreme_per_row > 50).sum())}")

print("\n=== are the 3 huge-scale columns concentrated in few rows? ===")
for c in ["x665", "x173", "x596"]:
    col = X_train[c]
    # fraction of values above 1e6 in magnitude
    frac_huge = (col.abs() > 1e6).mean()
    print(f"  {c}: fraction |value|>1e6 = {frac_huge:.3f}  "
          f"(so the whole column is on this scale, not a few outliers)"
          if frac_huge > 0.5 else
          f"  {c}: fraction |value|>1e6 = {frac_huge:.3f}  (concentrated in few rows)")

print("\n=== correlation of features with target (top absolute) ===")
y = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"]
# use rank correlation to be robust to scale/outliers
corr = X_train.corrwith(y, method="spearman").abs().sort_values(ascending=False)
print("top-10 |spearman corr| with y:")
print(corr.head(10).to_string())
print(f"\nfeatures with |corr| > 0.1: {int((corr > 0.1).sum())}")
print(f"features with |corr| > 0.2: {int((corr > 0.2).sum())}")
