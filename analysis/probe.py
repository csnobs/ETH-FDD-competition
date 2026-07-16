"""Deeper probe into scale heterogeneity, heavy tails and constant features."""

from __future__ import annotations

import numpy as np
import pandas as pd

DATA_DIR = "data"
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"])
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv").drop(columns=["id"])

print("=== how many columns at each scale (by std) ===")
stds = X_train.std()
for thr in [1e2, 1e3, 1e4, 1e6, 1e9, 1e12, 1e18]:
    print(f"    std > {thr:.0e}: {int((stds > thr).sum())} columns")

print("\n=== the extreme-scale columns (std > 1e9) ===")
big = stds[stds > 1e9].sort_values(ascending=False)
print(f"count: {len(big)}")
for c in big.head(10).index:
    col = X_train[c].dropna()
    print(f"  {c}: std={col.std():.2e} min={col.min():.2e} "
          f"med={col.median():.2e} max={col.max():.2e} skew={col.skew():.1f}")

print("\n=== constant / near-constant features ===")
nun = X_train.nunique()
for c in nun[nun <= 2].index:
    print(f"  {c}: n_unique={nun[c]}  values={X_train[c].dropna().unique()[:5]}")

print("\n=== skewness distribution (all features) ===")
skew = X_train.skew().abs()
print(f"  |skew|: median={skew.median():.2f}  mean={skew.mean():.2f}  max={skew.max():.2f}")
for thr in [1, 2, 5, 10]:
    print(f"    |skew| > {thr}: {int((skew > thr).sum())} / {len(skew)} columns")

print("\n=== do negative values exist? (affects log transform) ===")
has_neg = (X_train.min() < 0)
print(f"  features with any negative value: {int(has_neg.sum())} / {X_train.shape[1]}")
print(f"  features strictly >= 0: {int((~has_neg).sum())}")

print("\n=== a 'normal' scale feature for contrast (x0) ===")
print(X_train["x0"].describe().to_string())
