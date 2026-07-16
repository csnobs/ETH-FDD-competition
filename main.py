"""main.py - Submission pipeline for the ETH-FDD competition.

Uses the model selected by the experiments in ``analysis/`` (see
``analysis/DATA_ANALYSIS.md``):

    median impute -> SelectKBest(mutual_info, k=500)
                  -> Voting(GradientBoosting + XGBoost + ExtraTrees)

Robust 5x3 CV R^2 = 0.504 +/- 0.043 (single split ~0.52), versus the original
mean-impute + linear-regression baseline at ~0.18.

HOW TO GET THE DATA:
  Download from https://www.kaggle.com/competitions/eth-fdd-competition/data
  and place X_train.csv, y_train.csv, X_test.csv in DATA_DIR.

Run with:  python main.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.model import build_model_pipeline

DATA_DIR = "data"
OUTPUT_DIR = "output"


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Load features/target, dropping the id column from the feature matrices."""
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv")).drop(columns=["id"])
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["y"]
    X_test_df = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    test_ids = X_test_df["id"]
    X_test = X_test_df.drop(columns=["id"])
    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), test_ids


def main() -> None:
    X_train, y_train, X_test, test_ids = load_data()
    print(f"X_train {X_train.shape}  y_train {y_train.shape}  X_test {X_test.shape}")

    # Quick holdout to report a validation score (final model is fit on all data).
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    model = build_model_pipeline()
    model.fit(X_tr, y_tr)
    val_score = r2_score(y_val, model.predict(X_val))
    print(f"Holdout R^2: {val_score:.4f}")

    # Refit on the full training set before predicting the test set.
    final_model = build_model_pipeline()
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids.astype(int).to_numpy(), "y": y_test_pred})
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print("Submission saved to:", out_path)


if __name__ == "__main__":
    main()
