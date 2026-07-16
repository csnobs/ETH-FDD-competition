"""Fast unit tests for the cleaning pipeline (synthetic data, no training)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from src.cleaning import Winsorizer, build_cleaning_pipeline, build_model_pipeline


def test_winsorizer_clips_to_percentiles() -> None:
    X = np.arange(100, dtype=float).reshape(-1, 1)  # 0..99
    w = Winsorizer(lower_percentile=10, upper_percentile=90).fit(X)
    out = w.transform(X)
    assert out.min() >= np.percentile(X, 10)
    assert out.max() <= np.percentile(X, 90)


def test_cleaning_pipeline_removes_nan_and_constant_cols() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    X[:, 3] = 5.0  # constant column -> should be dropped
    X[0, 0] = np.nan  # missing value -> should be imputed
    out = build_cleaning_pipeline().fit_transform(X)

    assert not np.isnan(out).any()  # no missing values remain
    assert out.shape[0] == 50
    assert out.shape[1] == 3  # constant column removed


def test_model_pipeline_selects_k_and_predicts() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 20))
    y = X[:, 0] * 3.0 + rng.normal(scale=0.1, size=80)  # first feature is signal
    pipe = build_model_pipeline(LinearRegression(), k=5)
    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (80,)
    assert np.isfinite(preds).all()
    assert pipe.named_steps["select"].k == 5


def test_transforms_are_fit_only_on_training_data() -> None:
    """Winsorizer bounds come from fit data, not transform data."""
    X_fit = np.arange(100, dtype=float).reshape(-1, 1)
    w = Winsorizer(lower_percentile=0, upper_percentile=100).fit(X_fit)
    X_new = np.array([[1000.0]])  # far above fit range
    assert w.transform(X_new)[0, 0] == X_fit.max()  # clipped to fit-time max
