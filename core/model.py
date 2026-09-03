"""LightGBM classifier — CPU only, fixed seed, fully deterministic.

WHY GRADIENT-BOOSTED TREES AND NOT A NEURAL NETWORK
---------------------------------------------------
This is an explicit engineering-judgment call, made under a zero-GPU, zero-cloud-budget
constraint. GBDTs remain the strongest known family on heterogeneous tabular data of this
shape, train on CPU in minutes, handle NaN and categoricals natively, and are
deterministic given a seed. A tabular deep model or a GNN would cost far more compute to
train, be harder to reproduce, and — on the published IEEE-CIS leaderboard — perform no
better. Spending the compute budget on a graph *algorithm* rather than a graph *network*
buys more signal per watt here.

DETERMINISM
-----------
Fixed seed and `deterministic=True`, paired with `force_row_wise=True` because LightGBM's
deterministic mode only holds when one of the force_* options is set. Two runs on the same
data produce byte-identical predictions, asserted by `tests/test_model.py`
(`test_training_is_deterministic`, with `test_predictions_are_not_constant` to stop that
passing vacuously on a degenerate single-leaf model).

This matters because the whole submission rests on an ablation of small differences: if
the model were nondeterministic, the measured -0.0064 for k-core could be nothing more
than two different runs of the same configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from core.data import TIME_COL

RANDOM_SEED = 42

# Early stopping needs a validation set. It CANNOT be the test set — selecting the number
# of boosting rounds against the held-out data is a form of leakage that would inflate the
# reported score. So the training period is itself split temporally, and the final
# 20% of *train* becomes validation. The test set stays untouched until evaluation.
INNER_VALIDATION_QUANTILE = 0.80

BASE_PARAMS = {
    "objective": "binary",
    # 'average_precision' is AUC-PR. Optimising and early-stopping on the same metric the
    # project reports, rather than on AUC-ROC, which is optimistic under class imbalance.
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 96,
    "min_child_samples": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_jobs": -1,
    "seed": RANDOM_SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100


@dataclass
class TrainedModel:
    booster: lgb.Booster
    feature_names: list[str]
    best_iteration: int
    validation_ap: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Fraud probability in [0, 1]. Deterministic given the booster."""
        return self.booster.predict(
            X[self.feature_names], num_iteration=self.best_iteration
        )


def inner_temporal_validation_split(
    train: pd.DataFrame, quantile: float = INNER_VALIDATION_QUANTILE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set off the END of the training period, chronologically.

    Same reasoning as the outer split: a random inner split would let the early-stopping
    decision see the future.
    """
    cutoff = train[TIME_COL].quantile(quantile)
    fit = train[train[TIME_COL] < cutoff]
    validation = train[train[TIME_COL] >= cutoff]
    if fit.empty or validation.empty:
        raise ValueError("inner validation split produced an empty side")
    return fit, validation


def train_model(
    train: pd.DataFrame,
    feature_names: list[str],
    target: str = "isFraud",
    params: dict | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> TrainedModel:
    """Fit LightGBM with early stopping on a temporally held-out slice of train.

    `sample_weight` is optional and defaults to None, which reproduces the uniform-weight
    behaviour every previously published score was produced with. When supplied it must be
    indexed like `train`, so it can be split alongside the frame -- weights that stayed in
    the original row order while the data was split would silently pair each row with some
    other row's cost, and the model would still train, just on nonsense.
    """
    fit_df, val_df = inner_temporal_validation_split(train)

    merged_params = {**BASE_PARAMS, **(params or {})}

    fit_weight = val_weight = None
    if sample_weight is not None:
        weights = pd.Series(np.asarray(sample_weight, dtype=np.float64), index=train.index)
        fit_weight = weights.loc[fit_df.index].to_numpy()
        val_weight = weights.loc[val_df.index].to_numpy()

    fit_set = lgb.Dataset(fit_df[feature_names], label=fit_df[target], weight=fit_weight)
    val_set = lgb.Dataset(
        val_df[feature_names], label=val_df[target], weight=val_weight, reference=fit_set
    )

    evals: dict = {}
    booster = lgb.train(
        merged_params,
        fit_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_set],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.record_evaluation(evals),
        ],
    )

    best_iteration = booster.best_iteration
    validation_ap = evals["validation"]["average_precision"][best_iteration - 1]

    return TrainedModel(
        booster=booster,
        feature_names=feature_names,
        best_iteration=best_iteration,
        validation_ap=float(validation_ap),
    )
