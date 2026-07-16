"""Production model for the ETH-FDD competition.

Selected by the experiments in ``analysis/`` (see ``DATA_ANALYSIS.md``):
an averaging ensemble of three tree models on the top-500 mutual-information
features, with median imputation. Robust (5x3) CV R^2 = 0.504 +/- 0.043;
single-split ~0.52. Trees need no scaling, so this pipeline is intentionally
simpler than the linear-model cleaning pipeline in ``src/cleaning.py``.

    median impute -> SelectKBest(mutual_info, k=500) -> Voting(GB + XGB + ExtraTrees)
"""

from __future__ import annotations

from functools import partial

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

RANDOM_STATE = 0
N_FEATURES = 500


def build_estimator(random_state: int = RANDOM_STATE) -> VotingRegressor:
    """The averaging ensemble of the three best-performing tree models."""
    gb = GradientBoostingRegressor(
        learning_rate=0.05, n_estimators=400, max_depth=3,
        subsample=0.8, random_state=random_state,
    )
    xgb = XGBRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.3, reg_lambda=2.0,
        random_state=random_state, verbosity=0,
    )
    et = ExtraTreesRegressor(
        n_estimators=400, random_state=random_state, n_jobs=-1,
    )
    return VotingRegressor([("gb", gb), ("xgb", xgb), ("et", et)])


def build_model_pipeline(
    k: int = N_FEATURES, random_state: int = RANDOM_STATE
) -> Pipeline:
    """Full submission pipeline (leak-safe: every step fits on training data only)."""
    mi = partial(mutual_info_regression, random_state=random_state)
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(mi, k=k)),
            ("model", build_estimator(random_state)),
        ]
    )
