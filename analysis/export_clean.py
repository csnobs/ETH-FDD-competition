"""Export cleaned data files to the output/ folder.

Applies the leak-safe cleaning pipeline (median impute -> drop constant cols ->
winsorize -> robust scale), FIT ON TRAIN ONLY, then writes:
    output/X_train_clean.csv, output/X_test_clean.csv, output/y_train_clean.csv

Run with:  python analysis/export_clean.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import build_cleaning_pipeline

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    X_train_df = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test_df = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")

    train_ids = X_train_df["id"]
    test_ids = X_test_df["id"]
    feature_cols = [c for c in X_train_df.columns if c != "id"]

    X_train = X_train_df[feature_cols].to_numpy()
    X_test = X_test_df[feature_cols].to_numpy()
    y = y_train_df["y"].to_numpy()

    pipe = build_cleaning_pipeline()
    X_train_clean = pipe.fit_transform(X_train, y)  # fit on train only
    X_test_clean = pipe.transform(X_test)

    # Column names that survived the constant-column drop.
    kept_mask = pipe.named_steps["drop_constant"].get_support()
    kept_cols = [c for c, keep in zip(feature_cols, kept_mask) if keep]
    dropped = [c for c, keep in zip(feature_cols, kept_mask) if not keep]

    train_out = pd.DataFrame(X_train_clean, columns=kept_cols)
    train_out.insert(0, "id", train_ids.values)
    test_out = pd.DataFrame(X_test_clean, columns=kept_cols)
    test_out.insert(0, "id", test_ids.values)

    train_out.to_csv(OUTPUT_DIR / "X_train_clean.csv", index=False)
    test_out.to_csv(OUTPUT_DIR / "X_test_clean.csv", index=False)
    y_train_df.to_csv(OUTPUT_DIR / "y_train_clean.csv", index=False)

    print(f"Dropped {len(dropped)} constant columns: {dropped}")
    print(f"[scaled]  X_train_clean: {train_out.shape}  X_test_clean: {test_out.shape}")

    # --- Tree-friendly version: constant cols dropped, NaN kept, NOT scaled ---
    # Tree models (GB/XGB) use raw values and handle missing values natively, so
    # winsorising/scaling is unnecessary (and can slightly hurt). We keep NaN.
    kept = kept_cols
    tree_train = X_train_df[["id"] + kept].copy()
    tree_test = X_test_df[["id"] + kept].copy()
    tree_train.to_csv(OUTPUT_DIR / "X_train_clean_trees.csv", index=False)
    tree_test.to_csv(OUTPUT_DIR / "X_test_clean_trees.csv", index=False)
    print(f"[trees]   X_train_clean_trees: {tree_train.shape}  "
          f"X_test_clean_trees: {tree_test.shape}  (NaN kept, unscaled)")
    print(f"Wrote cleaned files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
