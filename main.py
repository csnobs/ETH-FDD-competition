"""
submission.py - Exemplar solution for AML 2025 Project 1 (eth-fdd-competition).

This is the baseline pipeline shown in the project slides ("Basic Examples"):
  1. Data loading
  2. Splitting and imputing (mean imputation)
  3. Feature selection (SelectKBest with mutual information)
  4. Training and validating a Linear Regression (R^2 score)
  5. Export the predictions in the submission format

HOW TO GET THE DATA:
  Download the files from the Kaggle competition page and put them in DATA_DIR:
    https://www.kaggle.com/competitions/eth-fdd-competition/data
    -> X_train.csv, y_train.csv, X_test.csv, sample.csv

Run with:  python3 submission.py or python submission.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATA_DIR = "data"   # folder containing the csv files
OUTPUT_DIR = "output"

# --- 1. Data loading --------------------------------------------------------
# skiprows=1 skips the header row, header=None keeps all columns positional.
# The first column (index 0) is the id, so we drop it with [:, 1:].
X_train_df = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"), skiprows=1, header=None)
y_train_df = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"), skiprows=1, header=None)
X_test_df = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"), skiprows=1, header=None)

X_train = X_train_df.values[:, 1:]
y_train = y_train_df.values[:, 1:]
X_test = X_test_df.values[:, 1:]

print(X_train.shape, y_train.shape, X_test.shape)

# --- 2. Splitting and imputing (subtask 0) ----------------------------------------------
# Randomly split the data into training and validation sets with 80-20 ratio
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Impute missing values with mean of each column (computed on the training set)
X_mean = np.nanmean(X_train, axis=0, keepdims=True)
X_train = np.where(np.isnan(X_train), X_mean, X_train)
X_val = np.where(np.isnan(X_val), X_mean, X_val)
X_test = np.where(np.isnan(X_test), X_mean, X_test)

# --- 3. Feature selection (subtask 2) ---------------------------------------------------
# Select the top 100 features with highest mutual information with the target.
selection = SelectKBest(mutual_info_regression, k=100).fit(X_train, y_train.ravel())
X_train = selection.transform(X_train)
X_val = selection.transform(X_val)
X_test = selection.transform(X_test)

# --- 4. Training and validating ---------------------------------------------
# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_val_pred = regressor.predict(X_val)

# Evaluate the model on training and validation sets
train_score = r2_score(y_train, y_train_pred)
val_score = r2_score(y_val, y_val_pred)

print(train_score, val_score)

# --- 5. Export and submit ---------------------------------------------------
# Predict on test set
y_test_pred = regressor.predict(X_test)

# Save predictions to submission file with the given format (id, y)
os.makedirs(OUTPUT_DIR, exist_ok=True)
table = pd.DataFrame(
    {"id": np.arange(0, y_test_pred.shape[0]), "y": y_test_pred.flatten()}
)
table.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
print("Submission saved to:", os.path.join(OUTPUT_DIR, "submission.csv"))