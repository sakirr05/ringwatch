"""Tests for the temporal split.

These run on small synthetic frames rather than the 590k-row dataset so they stay fast,
with one opt-in integration test against the real data at the bottom.

The property that matters most is NO TEMPORAL LEAKAGE: max(train time) < min(test time),
strictly. Everything else is secondary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data import MERGED_PARQUET, TARGET, TIME_COL
from core.split import DEFAULT_SPLIT_QUANTILE, split_summary, temporal_split


def make_frame(n: int = 1000, seed: int = 0, tied: bool = False) -> pd.DataFrame:
    """Synthetic frame shaped like the real one: a time column and a rare positive class."""
    rng = np.random.default_rng(seed)
    if tied:
        # Heavy timestamp ties: only 10 distinct values across n rows.
        times = np.repeat(np.arange(10) * 86_400, n // 10)[:n]
    else:
        times = np.sort(rng.integers(86_400, 15_811_131, size=n))
    return pd.DataFrame(
        {
            TIME_COL: times,
            TARGET: (rng.random(n) < 0.035).astype(int),
            "feature": rng.normal(size=n),
        }
    )


def test_no_temporal_leakage():
    """The defining property: nothing in train may occur at or after anything in test."""
    train, test = temporal_split(make_frame())
    assert train[TIME_COL].max() < test[TIME_COL].min()


def test_no_temporal_leakage_with_heavy_ties():
    """Tied timestamps at the boundary must not straddle the split."""
    train, test = temporal_split(make_frame(tied=True), quantile=0.5)
    assert train[TIME_COL].max() < test[TIME_COL].min()


def test_split_is_a_partition():
    """No row is lost and no row is duplicated across the two sides."""
    df = make_frame()
    train, test = temporal_split(df)
    assert len(train) + len(test) == len(df)
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(df.index)


def test_split_ratio_is_approximately_the_quantile():
    train, _ = temporal_split(make_frame(n=10_000), quantile=0.8)
    assert 0.78 <= len(train) / 10_000 <= 0.82


def test_split_is_deterministic():
    """Same input, same output — no randomness anywhere in the split."""
    df = make_frame()
    a_train, a_test = temporal_split(df)
    b_train, b_test = temporal_split(df)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_split_is_insensitive_to_input_row_order():
    """Shuffling the input must not change the resulting split, only its row order."""
    df = make_frame()
    ordered_train, ordered_test = temporal_split(df)
    shuffled = df.sample(frac=1.0, random_state=7)
    shuffled_train, shuffled_test = temporal_split(shuffled)
    assert set(ordered_train.index) == set(shuffled_train.index)
    assert set(ordered_test.index) == set(shuffled_test.index)


def test_both_sides_contain_both_classes():
    """A test set with no positives would make AUC-PR undefined."""
    train, test = temporal_split(make_frame(n=20_000))
    for side in (train, test):
        assert side[TARGET].nunique() == 2


@pytest.mark.parametrize("bad_quantile", [0.0, 1.0, -0.1, 1.5])
def test_rejects_out_of_range_quantile(bad_quantile):
    with pytest.raises(ValueError):
        temporal_split(make_frame(), quantile=bad_quantile)


def test_summary_reports_consistent_counts():
    df = make_frame()
    train, test = temporal_split(df)
    summary = split_summary(train, test)
    assert summary["train_rows"] + summary["test_rows"] == len(df)
    assert summary["train_fraud_count"] + summary["test_fraud_count"] == int(
        df[TARGET].sum()
    )


@pytest.mark.skipif(
    not MERGED_PARQUET.exists(), reason="real dataset cache not built yet"
)
def test_real_dataset_split_has_no_leakage():
    """Integration check against the actual 590k-row dataset."""
    from core.data import load_merged

    df = load_merged()
    train, test = temporal_split(df, DEFAULT_SPLIT_QUANTILE)
    assert train[TIME_COL].max() < test[TIME_COL].min()
    assert len(train) + len(test) == len(df)
    assert train[TARGET].nunique() == 2 and test[TARGET].nunique() == 2
