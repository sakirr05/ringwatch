"""Scoring a Razorpay payload with the IEEE-CIS model — explicitly a demonstration.

READ THIS BEFORE TRUSTING ANY NUMBER THIS MODULE RETURNS
--------------------------------------------------------
The classifier was trained on IEEE-CIS, a US e-commerce dataset assembled by Vesta. It
expects **433 features**. A Razorpay payment payload can supply **3** of them.

Why so few. `card1`, `card2` and `card5` are Vesta-internal identifiers, not card network
or type. `addr1`/`addr2` and `dist1`/`dist2` likewise. `C1`-`C14`, `D1`-`D15`, `M1`-`M9`
and `V1`-`V339` — roughly 400 columns — are Vesta's proprietary engineered features and
have no counterpart in any payment processor's webhook. No mapping exists; there is nothing
to be clever about.

LightGBM will accept 430 missing values and return a number. That number comes from a model
operating almost entirely outside its training distribution, and it is **not** an assessment
of the transaction. This module exists to demonstrate that the ingestion -> scoring path
works end to end, and to quantify exactly how little transfers.

WHY THE CATEGORICAL FIELDS ARE DELIBERATELY *NOT* MAPPED
---------------------------------------------------------
`card4` (visa/mastercard/amex) and `card6` (credit/debit) look like clean matches for
Razorpay's `card.network` and `card.type`, and `P_emaildomain` looks like a match for the
email domain. They are excluded anyway.

LightGBM encodes categoricals as integer codes fixed at training time. Feeding a category
whose code ordering differs from training's silently maps "visa" onto whatever category
happened to occupy that code in the training data. That is a wrong number that looks
plausible — strictly worse than an honest missing value. Mapping them correctly would mean
reconstructing the training-time category ordering from the model file, which is fragile
machinery in service of a score that is a demonstration anyway.

So only unambiguous numeric features are mapped, and coverage is reported as a measured
figure rather than a rhetorical caveat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Committed alongside the code so the deployed instance has it; data/cache/ is gitignored.
MODEL_PATHS = [
    REPO_ROOT / "artifacts" / "model_baseline.txt",
    REPO_ROOT / "data" / "cache" / "model_baseline.txt",
]

# IEEE-CIS TransactionAmt is denominated in USD. Razorpay amounts arrive in paise. The
# conversion at least puts the magnitude in the neighbourhood the model was trained on;
# it does not make the feature vector valid.
USD_TO_INR = 88.0


@dataclass
class DemoScore:
    """A demonstration score, inseparable from the caveat that qualifies it."""

    score: float | None
    features_present: int
    features_total: int
    mapped: dict[str, float] = field(default_factory=dict)
    available: bool = True
    reason: str | None = None

    @property
    def coverage_pct(self) -> float:
        if not self.features_total:
            return 0.0
        return 100.0 * self.features_present / self.features_total

    @property
    def caveat(self) -> str:
        return (
            f"{self.features_present} of {self.features_total} features present; "
            f"{self.features_total - self.features_present} imputed as missing. This model "
            "was trained on IEEE-CIS US e-commerce data and has never seen a Razorpay "
            "payload. This demonstrates the ingestion path, not an assessment of this "
            "transaction."
        )

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "features_present": self.features_present,
            "features_total": self.features_total,
            "coverage_pct": self.coverage_pct,
            "mapped": self.mapped,
            "available": self.available,
            "reason": self.reason,
            "caveat": self.caveat,
        }


_BOOSTER = None
_FEATURE_NAMES: list[str] = []


def _load_booster():
    """Load the committed booster once. Absence is a degraded state, not an error."""
    global _BOOSTER, _FEATURE_NAMES
    if _BOOSTER is not None:
        return _BOOSTER

    import lightgbm as lgb  # imported lazily so the web layer can start without it

    for path in MODEL_PATHS:
        if path.exists():
            _BOOSTER = lgb.Booster(model_file=str(path))
            _FEATURE_NAMES = _BOOSTER.feature_name()
            return _BOOSTER
    return None


def extract_features(payment: dict) -> dict[str, float]:
    """Map the handful of Razorpay fields that genuinely correspond to model features."""
    mapped: dict[str, float] = {}

    amount_paise = payment.get("amount")
    if isinstance(amount_paise, (int, float)) and amount_paise > 0:
        mapped["TransactionAmt"] = (float(amount_paise) / 100.0) / USD_TO_INR

    created_at = payment.get("created_at")
    if isinstance(created_at, (int, float)) and created_at > 0:
        # Genuine 24-hour and 7-day cycles. Note the training column counts seconds from an
        # unpublished origin, so the phase alignment is not guaranteed to match.
        mapped["tx_hour"] = float(int(created_at // 3_600) % 24)
        mapped["tx_dayofweek"] = float(int(created_at // 86_400) % 7)

    return mapped


def score_payment(payment: dict) -> DemoScore:
    """Run the booster on a Razorpay payment entity, with coverage measured."""
    booster = _load_booster()
    if booster is None:
        return DemoScore(
            score=None,
            features_present=0,
            features_total=0,
            available=False,
            reason=(
                "artifacts/model_baseline.txt not present; the demonstration scorer is "
                "unavailable. Nothing else is affected."
            ),
        )

    mapped = extract_features(payment)

    # A plain numpy row rather than a DataFrame: this sidesteps LightGBM's pandas
    # categorical handling entirely, which is the machinery that could silently mis-encode
    # a category. NaN means "missing" and is handled natively.
    row = np.full((1, len(_FEATURE_NAMES)), np.nan, dtype=np.float64)
    index = {name: i for i, name in enumerate(_FEATURE_NAMES)}
    present = 0
    for name, value in mapped.items():
        if name in index:
            row[0, index[name]] = value
            present += 1

    score = float(booster.predict(row)[0])
    return DemoScore(
        score=score,
        features_present=present,
        features_total=len(_FEATURE_NAMES),
        mapped=mapped,
    )
