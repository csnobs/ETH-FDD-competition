"""Exploratory data analysis for the ETH-FDD competition dataset.

Step 1: characterise missing values and outliers.
Step 2: look for further data problems (distribution shift / perturbations,
        duplicates, constant features, quantisation, target anomalies).

Run with:  python analysis/analyze_data.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATA_DIR = "data"


def load() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv").drop(columns=["id"])
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")["y"]
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv").drop(columns=["id"])
    return X_train, y_train, X_test


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def basic_shape(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> None:
    section("1. SHAPES & DTYPES")
    print(f"X_train: {X_train.shape}   y_train: {y_train.shape}   X_test: {X_test.shape}")
    print(f"feature dtypes: {X_train.dtypes.value_counts().to_dict()}")
    print(f"n features: {X_train.shape[1]}")


def missing_analysis(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    section("2. MISSING VALUES")
    for name, X in [("train", X_train), ("test", X_test)]:
        n_cells = X.size
        n_missing = int(X.isna().sum().sum())
        col_missing = X.isna().mean()
        row_missing = X.isna().mean(axis=1)
        cols_with_na = int((col_missing > 0).sum())
        print(f"\n[{name}] total missing: {n_missing} / {n_cells} "
              f"({100 * n_missing / n_cells:.2f}%)")
        print(f"[{name}] columns with any NaN: {cols_with_na} / {X.shape[1]}")
        print(f"[{name}] per-column NaN rate:  min={col_missing.min():.3f} "
              f"median={col_missing.median():.3f} max={col_missing.max():.3f}")
        print(f"[{name}] per-row    NaN rate:  min={row_missing.min():.3f} "
              f"median={row_missing.median():.3f} max={row_missing.max():.3f}")
        print(f"[{name}] rows fully complete: {int((row_missing == 0).sum())} / {X.shape[0]}")
        # worst columns
        worst = col_missing.sort_values(ascending=False).head(5)
        print(f"[{name}] top-5 columns by NaN rate:")
        for c, v in worst.items():
            print(f"    {c}: {100 * v:.1f}%")


def outlier_analysis(X_train: pd.DataFrame) -> None:
    section("3. OUTLIERS (train)")
    desc = X_train.describe().T
    # feature scale spread
    print("Feature value ranges vary massively across columns:")
    print(f"    column means:  min={desc['mean'].min():.2f}  max={desc['mean'].max():.2f}")
    print(f"    column stds:   min={desc['std'].min():.2f}  max={desc['std'].max():.2f}")
    print(f"    global min={X_train.min().min():.2f}  global max={X_train.max().max():.2f}")

    # IQR-based outlier count per column
    q1 = X_train.quantile(0.25)
    q3 = X_train.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (X_train < lower) | (X_train > upper)
    out_per_col = outlier_mask.mean()  # fraction flagged per column
    print(f"\nIQR outliers (1.5*IQR rule), fraction of values flagged per column:")
    print(f"    mean={out_per_col.mean():.3f}  median={out_per_col.median():.3f} "
          f"max={out_per_col.max():.3f}")
    print(f"    columns with >5% flagged: {int((out_per_col > 0.05).sum())} / {X_train.shape[1]}")

    # extreme z-scores
    z = ((X_train - X_train.mean()) / X_train.std()).abs()
    extreme = (z > 5).sum().sum()
    print(f"\nValues with |z| > 5: {int(extreme)} "
          f"({100 * extreme / X_train.notna().sum().sum():.3f}% of non-missing)")
    print(f"max |z| observed: {z.max().max():.1f}")


def further_problems(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> None:
    section("4. FURTHER PROBLEMS")

    # --- duplicates ---
    dup_rows_tr = X_train.duplicated().sum()
    dup_rows_te = X_test.duplicated().sum()
    print(f"Duplicate rows: train={dup_rows_tr}, test={dup_rows_te}")

    # --- constant / near-constant features ---
    nunique = X_train.nunique()
    constant = int((nunique <= 1).sum())
    near_const = int((nunique <= 2).sum())
    print(f"Constant features: {constant}; features with <=2 unique values: {near_const}")

    # --- duplicate / perfectly correlated columns ---
    # (cheap check: identical columns)
    dup_cols = X_train.T.duplicated().sum()
    print(f"Exactly duplicated feature columns (train): {dup_cols}")

    # --- target analysis ---
    section("4b. TARGET (y_train)")
    print(y_train.describe().to_string())
    frac_int = np.mean(np.isclose(y_train.dropna(), np.round(y_train.dropna())))
    print(f"fraction of y that are integers: {frac_int:.3f}")
    print(f"y missing: {int(y_train.isna().sum())}")
    print(f"y negative values: {int((y_train < 0).sum())}")

    # --- train vs test distribution shift (possible perturbation) ---
    section("4c. TRAIN vs TEST DISTRIBUTION SHIFT / PERTURBATIONS")
    tr_mean, te_mean = X_train.mean(), X_test.mean()
    tr_std = X_train.std()
    # standardised difference of means per feature
    shift = (te_mean - tr_mean).abs() / (tr_std.replace(0, np.nan))
    shift = shift.dropna()
    print(f"Per-feature |mean(test)-mean(train)| / std(train):")
    print(f"    median={shift.median():.3f}  mean={shift.mean():.3f}  max={shift.max():.3f}")
    print(f"    features shifted > 0.5 std: {int((shift > 0.5).sum())} / {len(shift)}")
    print(f"    features shifted > 1.0 std: {int((shift > 1.0).sum())} / {len(shift)}")
    worst_shift = shift.sort_values(ascending=False).head(8)
    print("    top-8 shifted features (std units):")
    for c, v in worst_shift.items():
        print(f"        {c}: {v:.2f}   train_mean={tr_mean[c]:.2f} test_mean={te_mean[c]:.2f}")

    # std ratio test/train (scale perturbation)
    std_ratio = (X_test.std() / tr_std.replace(0, np.nan)).dropna()
    print(f"\nPer-feature std(test)/std(train):")
    print(f"    median={std_ratio.median():.3f}  min={std_ratio.min():.3f}  max={std_ratio.max():.3f}")
    print(f"    features with ratio >2 or <0.5: "
          f"{int(((std_ratio > 2) | (std_ratio < 0.5)).sum())} / {len(std_ratio)}")

    # --- quantisation / discreteness (sign of injected noise or rounding) ---
    section("4d. DISCRETENESS / QUANTISATION")
    frac_unique = (X_train.nunique() / len(X_train))
    print(f"Per-feature (unique values / n_rows):")
    print(f"    min={frac_unique.min():.3f} median={frac_unique.median():.3f} "
          f"max={frac_unique.max():.3f}")
    print(f"    likely-continuous features (>0.9 unique): "
          f"{int((frac_unique > 0.9).sum())} / {X_train.shape[1]}")
    print(f"    likely-categorical/discrete (<0.05 unique): "
          f"{int((frac_unique < 0.05).sum())} / {X_train.shape[1]}")

    # --- infinities ---
    section("4e. INFINITIES / SANITY")
    inf_tr = np.isinf(X_train.to_numpy(dtype=float, na_value=np.nan)).sum()
    inf_te = np.isinf(X_test.to_numpy(dtype=float, na_value=np.nan)).sum()
    print(f"Infinite values: train={int(inf_tr)}, test={int(inf_te)}")


def main() -> None:
    X_train, y_train, X_test = load()
    basic_shape(X_train, y_train, X_test)
    missing_analysis(X_train, X_test)
    outlier_analysis(X_train)
    further_problems(X_train, y_train, X_test)
    print("\nDone.")


if __name__ == "__main__":
    main()
