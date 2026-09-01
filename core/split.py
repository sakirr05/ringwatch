"""Temporal train/test split.

WHY THIS IS NOT A RANDOM SPLIT
------------------------------
IEEE-CIS is time-ordered transaction data spanning 182 days. A random split would place
transactions from the same card, the same day, and often the same *fraud ring* on both
sides of the boundary, letting the model learn from the future to predict the past. The
resulting score would be inflated and meaningless — it would measure interpolation within
a known period, not the thing a deployed fraud system actually does, which is score
transactions it has never seen from a period that has not happened yet.

This matters doubly here because of the graph layer: an entity graph built across a
randomly-split dataset would connect training transactions to test transactions directly,
turning graph features into an explicit leakage channel.

So: sort by TransactionDT, cut at a quantile, everything before is train and everything
at-or-after is test. The cut uses strict inequality on the boundary value, which
guarantees no timestamp appears on both sides even when many transactions share a second.
"""

from __future__ import annotations

import pandas as pd

from core.data import TARGET, TIME_COL

DEFAULT_SPLIT_QUANTILE = 0.80


def temporal_split(
    df: pd.DataFrame, quantile: float = DEFAULT_SPLIT_QUANTILE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically at `quantile` of TransactionDT.

    Returns (train, test) where every train timestamp is strictly less than every test
    timestamp. Because the cut is applied to the *value* rather than to row positions,
    the realised train fraction may differ slightly from `quantile` when timestamps are
    tied at the boundary — correctness of the separation is preferred over hitting the
    ratio exactly.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    cutoff = df[TIME_COL].quantile(quantile)
    train = df[df[TIME_COL] < cutoff].copy()
    test = df[df[TIME_COL] >= cutoff].copy()

    if train.empty or test.empty:
        raise ValueError(f"split at quantile={quantile} produced an empty side")

    return train, test


def split_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Stats for the Phase 2 milestone output.

    The train/test fraud rates are reported separately on purpose: fraud rate drifts over
    time in this dataset, and a large gap between the two is itself a finding that shapes
    how much the held-out score can be trusted.
    """
    total = len(train) + len(test)
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_frac": len(train) / total,
        "boundary_dt": int(test[TIME_COL].min()),
        "boundary_day": float(test[TIME_COL].min() / 86_400),
        "train_fraud_rate": float(train[TARGET].mean()),
        "test_fraud_rate": float(test[TARGET].mean()),
        "train_fraud_count": int(train[TARGET].sum()),
        "test_fraud_count": int(test[TARGET].sum()),
        "train_span_days": float(
            (train[TIME_COL].max() - train[TIME_COL].min()) / 86_400
        ),
        "test_span_days": float((test[TIME_COL].max() - test[TIME_COL].min()) / 86_400),
    }
