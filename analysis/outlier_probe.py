"""Empirical probe: Isolation Forest vs Local Outlier Factor on our data.

Checks, before committing to a method:
  (a) how many rows each flags at low contamination,
  (b) how much the two methods agree,
  (c) whether they flag the ~3 rows we already identified as corrupted
      (rows with many robust-z extreme features, cf. row_probe.py).

Uses median imputation + RobustScaler just to get a complete, comparably
scaled matrix for the detectors (Isolation Forest is scale-invariant; LOF
needs the scaling). Run with:  python analysis/outlier_probe.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

DATA_DIR = "data"
RS = 0

X = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"])
Xv = X.to_numpy()

# --- "ground truth" corrupted rows: many robust-z extreme features -----------
med = X.median()
mad = (X - med).abs().median() * 1.4826
mad = mad.replace(0, np.nan)
z = (X - med).abs() / mad
extreme_per_row = (z > 5).sum(axis=1)
known_bad = set(np.where(extreme_per_row > 20)[0])
print(f"Rows with >20 robust-z extreme features (known corrupted): "
      f"{sorted(known_bad)} (n={len(known_bad)})")
print(f"Rows with >50: {sorted(np.where(extreme_per_row > 50)[0])}")

# --- complete, scaled matrix for the detectors -------------------------------
X_imp = SimpleImputer(strategy="median").fit_transform(Xv)
X_scaled = RobustScaler().fit_transform(X_imp)


def flagged(mask: np.ndarray) -> set[int]:
    return set(np.where(mask == -1)[0])


for contamination in [0.005, 0.01, 0.02]:
    iso = IsolationForest(contamination=contamination, n_estimators=200,
                          random_state=RS)
    iso_flags = flagged(iso.fit_predict(X_scaled))

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    lof_flags = flagged(lof.fit_predict(X_scaled))

    overlap = iso_flags & lof_flags
    print(f"\n=== contamination = {contamination} ===")
    print(f"IsolationForest flagged: {len(iso_flags)} rows")
    print(f"LOF flagged:             {len(lof_flags)} rows")
    print(f"Agreement (both):        {len(overlap)} rows")
    print(f"iForest catches known-bad: "
          f"{sorted(iso_flags & known_bad)} / {sorted(known_bad)}")
    print(f"LOF catches known-bad:     "
          f"{sorted(lof_flags & known_bad)} / {sorted(known_bad)}")
