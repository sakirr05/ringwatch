"""Feature preparation for the tabular baseline.

Kept deliberately thin. This is the *baseline* the graph layer must beat, so it gets no
special cleverness — hand-tuning it would make the later ablation flattering rather than
informative. LightGBM consumes NaN and pandas `category` dtype natively, so there is no
imputation and no one-hot expansion.
"""

from __future__ import annotations

import pandas as pd

from core.data import ID_COL, TARGET, TIME_COL

# Raw TransactionDT is excluded as a feature on purpose. It is a monotonically increasing
# clock, and the test set lies entirely to the right of the train set, so every test value
# is outside the range the trees ever saw. A split on it cannot generalise; it can only
# memorise the training period. Cyclical derivatives of it (hour, day-of-week) DO
# generalise and are engineered below.
# `g_component` is a connected-component LABEL, not a quantity. Its numeric value is an
# arbitrary node id, so letting a tree split on it would be memorising which specific
# components happened to contain fraud in the training period -- an identifier leak
# dressed up as a feature. It is carried in the frame for cluster grouping and never fed
# to the model.
EXCLUDED = {ID_COL, TARGET, TIME_COL, "g_component", "uid"}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cyclical time features from the raw seconds-offset clock.

    TransactionDT is seconds from an unspecified reference point. The absolute origin is
    unknown, so 'hour' here is hour-relative-to-that-origin rather than true local time —
    it is still a consistent 24-cycle, which is what matters for capturing the well-known
    nocturnal skew of card-testing traffic.
    """
    out = df.copy()
    out["tx_hour"] = (df[TIME_COL] // 3_600) % 24
    out["tx_dayofweek"] = (df[TIME_COL] // 86_400) % 7
    out["tx_day"] = df[TIME_COL] // 86_400
    return out


def feature_columns(df: pd.DataFrame, extra_excluded: set[str] | None = None) -> list[str]:
    """Every column the model is allowed to see."""
    excluded = EXCLUDED | (extra_excluded or set())
    # tx_day is an absolute day index -- same unbounded-clock problem as TransactionDT.
    # It is retained in the frame because the graph layer needs it to build the uid
    # fingerprint, but it is never handed to the model.
    excluded = excluded | {"tx_day"}
    return [c for c in df.columns if c not in excluded]


def prepare(
    df: pd.DataFrame, extra_excluded: set[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) ready for LightGBM."""
    framed = add_time_features(df)
    cols = feature_columns(framed, extra_excluded)
    return framed[cols], framed[TARGET], cols
