"""Leak-safe data-cleaning pipeline for the ETH-FDD competition.

Implements the plan from ``analysis/DATA_ANALYSIS.md`` as scikit-learn
transformers so every step is fit on the training fold only (no leakage when
used inside ``Pipeline`` / cross-validation):

    median impute -> drop constant columns -> winsorize outliers -> robust scale

``build_cleaning_pipeline`` returns just the preprocessing; ``build_model_pipeline``
appends feature selection and an estimator.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to per-column percentile bounds learned on ``fit``.

    Tames point outliers without dropping rows (see analysis: only a handful of
    extreme values per column). Bounds are learned on training data only.
    """

    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X: npt.ArrayLike, y: object = None) -> "Winsorizer":
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.lower_percentile, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper_percentile, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)


def build_cleaning_pipeline(
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Pipeline:
    """Preprocessing only: impute -> drop constant cols -> winsorize -> scale."""
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("drop_constant", VarianceThreshold(threshold=0.0)),
            ("winsorize", Winsorizer(lower_percentile, upper_percentile)),
            ("scale", RobustScaler()),
        ]
    )


def build_model_pipeline(
    estimator: BaseEstimator,
    k: int = 100,
    random_state: int = 0,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Pipeline:
    """Full pipeline: cleaning -> SelectKBest(mutual info) -> estimator."""
    mi = partial(mutual_info_regression, random_state=random_state)
    cleaning = build_cleaning_pipeline(lower_percentile, upper_percentile)
    return Pipeline(
        steps=[
            *cleaning.steps,
            ("select", SelectKBest(mi, k=k)),
            ("model", estimator),
        ]
    )
