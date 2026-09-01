"""Loading for the IEEE-CIS Fraud Detection dataset.

Deterministic and fully offline once the raw CSVs are present. The raw files are ~710 MB
of CSV; naive float64 loading costs roughly 1.8 GB of RAM, so everything is downcast on
ingest and cached to Parquet. Every later phase reads the cache, which makes repeated
runs fast and byte-identical.

No network access happens here. See `scripts/fetch_data.py` for acquisition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
CACHE_DIR = REPO_ROOT / "data" / "cache"

TRANSACTION_CSV = RAW_DIR / "train_transaction.csv"
IDENTITY_CSV = RAW_DIR / "train_identity.csv"
MERGED_PARQUET = CACHE_DIR / "train_merged.parquet"

TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"

# Published characteristics of IEEE-CIS train_transaction. Asserted on load so that a
# truncated download or a silently-substituted mirror fails loudly here rather than
# surfacing as a mysteriously wrong metric three phases later.
EXPECTED_ROWS = 590_540
EXPECTED_FRAUD_RATE = 0.0350  # to 3 d.p.; actual 0.034990


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric columns in place-ish without changing values.

    float64 -> float32 is safe for this data: the features are transaction amounts,
    counts and day-offsets, none of which need 15 significant digits. Integer columns
    go to the smallest type that holds their observed range.
    """
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(dtype):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif dtype == object or pd.api.types.is_string_dtype(dtype):
            # pandas 3.0 stores text as the new `str` dtype rather than `object`, so an
            # `== object` check alone silently misses every categorical column and hands
            # LightGBM a dtype it refuses. See FAILURE_LOG.md.
            df[col] = df[col].astype("category")
    return df


def load_merged(use_cache: bool = True) -> pd.DataFrame:
    """Load transactions left-joined with identity, downcast and cached.

    The join is a LEFT join: identity covers only ~24% of transactions, and dropping the
    uncovered majority would throw away most of the dataset. Missing identity columns
    stay NaN, which LightGBM handles natively.
    """
    if use_cache and MERGED_PARQUET.exists():
        return pd.read_parquet(MERGED_PARQUET)

    if not TRANSACTION_CSV.exists():
        raise FileNotFoundError(
            f"{TRANSACTION_CSV} not found. Run: python scripts/fetch_data.py"
        )

    transactions = pd.read_csv(TRANSACTION_CSV)
    identity = pd.read_csv(IDENTITY_CSV)

    _assert_dataset_integrity(transactions)

    df = transactions.merge(identity, on=ID_COL, how="left")
    df = _downcast(df)
    df = df.sort_values(TIME_COL, kind="mergesort").reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MERGED_PARQUET, index=False)
    return df


def _assert_dataset_integrity(transactions: pd.DataFrame) -> None:
    """Fail loudly on a wrong or truncated dataset rather than silently modelling it."""
    n = len(transactions)
    if n != EXPECTED_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ROWS} rows in train_transaction.csv, got {n}. "
            "The download may be truncated or the mirror may not be IEEE-CIS."
        )

    fraud_rate = transactions[TARGET].mean()
    if abs(fraud_rate - EXPECTED_FRAUD_RATE) > 0.001:
        raise ValueError(
            f"fraud rate {fraud_rate:.4f} differs from the published "
            f"{EXPECTED_FRAUD_RATE:.4f}; this may not be the IEEE-CIS training set."
        )


def dataset_summary(df: pd.DataFrame) -> dict:
    """Descriptive stats used in the Phase 2 milestone output and the README."""
    span_days = (df[TIME_COL].max() - df[TIME_COL].min()) / 86_400
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "fraud_count": int(df[TARGET].sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "span_days": float(span_days),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
    }
