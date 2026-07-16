"""Fast unit tests for the production model pipeline (synthetic data)."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from src.model import build_estimator, build_model_pipeline


def test_pipeline_structure() -> None:
    pipe = build_model_pipeline(k=10)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["impute", "select", "model"]
    assert pipe.named_steps["select"].k == 10


def test_estimator_has_three_members() -> None:
    ens = build_estimator()
    names = [name for name, _ in ens.estimators]
    assert names == ["gb", "xgb", "et"]


def test_pipeline_fits_and_predicts_with_nan() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 30))
    X[0, 0] = np.nan  # imputer must handle missing values
    y = X[:, 1] * 2.0 + rng.normal(scale=0.1, size=60)

    pipe = build_model_pipeline(k=10)
    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == (60,)
    assert np.isfinite(preds).all()
