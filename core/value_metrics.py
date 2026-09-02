"""Value-weighted evaluation: does the graph layer matter where the money is?

WHY THIS MODULE EXISTS
----------------------
AUC-PR is count-uniform. It treats a small fraud and a large one as equally important, which
is not how a payments business experiences fraud. PayPal's engineers state they "optimize for
higher accuracy of dollar-weighted fraud detection", and that raises a fair objection to this
project's headline negative result: perhaps the graph layer looks useless only because the
metric ignores value.

This module answers that with a measurement. See `PLAN_VALUE_WEIGHTED.md` for the predictions
recorded before it was run.

TERMINOLOGY
-----------
"Dollar-weighted fraud detection" is PayPal's own phrase. The formalised metric name **Value
Detection Rate (VDR)** comes from a different source — Dervovic, Amiri and Cashmore (JP Morgan
AI Research, FinPlan 2023) — and is not PayPal's term. The distinction is kept because
misattributing an acronym is exactly the kind of small dishonesty this project tries not to
commit.

WHY RAW AMOUNTS, NOT RUPEE-CONVERTED
------------------------------------
VDR is a **ratio** of value caught to value present, so any scale factor cancels. Weighting by
raw `TransactionAmt` therefore needs **no exchange-rate assumption at all** — strictly fewer
premises than this project's rupee-denominated insult costing, which carries a documented
USD_TO_INR figure. `test_value_metrics.py` asserts that scale invariance directly.

WHAT THESE NUMBERS ARE NOT
--------------------------
IEEE-CIS `TransactionAmt` is anonymised and possibly transformed. Everything here is evidence
about **value structure within the dataset**, never a claim about literal money saved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score


@dataclass
class ValueConcentration:
    """Is fraud in some subgroup disproportionately valuable?

    The mechanism check. Value weighting can only change a conclusion if the subgroup it
    favours actually carries more value than its share of the count.
    """

    subgroup: str
    n_rows: int
    n_fraud: int
    count_share: float
    value_share: float
    mean_amount_in: float
    mean_amount_out: float
    median_amount_in: float
    median_amount_out: float

    @property
    def enrichment(self) -> float:
        """Value share divided by count share. 1.0 means no enrichment whatsoever."""
        return self.value_share / self.count_share if self.count_share else float("nan")

    def summary_lines(self) -> list[str]:
        return [
            f"  subgroup                {self.subgroup}",
            f"  rows / fraud rows       {self.n_rows:,} / {self.n_fraud:,}",
            f"  share of fraud COUNT    {100 * self.count_share:.2f}%",
            f"  share of fraud VALUE    {100 * self.value_share:.2f}%",
            f"  value enrichment        {self.enrichment:.3f}x  "
            f"({'no' if abs(self.enrichment - 1) < 0.25 else 'some'} concentration)",
            f"  mean fraud amount       {self.mean_amount_in:.2f} in-group vs "
            f"{self.mean_amount_out:.2f} out",
            f"  median fraud amount     {self.median_amount_in:.2f} in-group vs "
            f"{self.median_amount_out:.2f} out",
        ]


def value_concentration(
    y_true: np.ndarray,
    amounts: np.ndarray,
    mask: np.ndarray,
    subgroup: str = "subgroup",
) -> ValueConcentration:
    """Compare a subgroup's share of fraud value against its share of fraud count.

    This runs *before* any value-weighted model comparison, because it decides whether the
    comparison can possibly change anything. If a subgroup holds the same share of value as
    of count, no amount of reweighting will make features about that subgroup matter more.
    """
    y_true = np.asarray(y_true).astype(int)
    amounts = np.asarray(amounts, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    fraud = y_true == 1
    in_fraud = mask & fraud
    out_fraud = (~mask) & fraud

    total_fraud_count = int(fraud.sum())
    total_fraud_value = float(amounts[fraud].sum())

    return ValueConcentration(
        subgroup=subgroup,
        n_rows=int(mask.sum()),
        n_fraud=int(in_fraud.sum()),
        count_share=(in_fraud.sum() / total_fraud_count) if total_fraud_count else 0.0,
        value_share=(
            float(amounts[in_fraud].sum()) / total_fraud_value if total_fraud_value else 0.0
        ),
        mean_amount_in=float(amounts[in_fraud].mean()) if in_fraud.any() else 0.0,
        mean_amount_out=float(amounts[out_fraud].mean()) if out_fraud.any() else 0.0,
        median_amount_in=float(np.median(amounts[in_fraud])) if in_fraud.any() else 0.0,
        median_amount_out=float(np.median(amounts[out_fraud])) if out_fraud.any() else 0.0,
    )


def value_weighted_average_precision(
    y_true: np.ndarray, y_score: np.ndarray, amounts: np.ndarray
) -> float:
    """AUC-PR with each transaction weighted by its value.

    Deliberately a thin wrapper over scikit-learn's `sample_weight` rather than a
    reimplementation: it is the same estimator the project already reports, with the
    uniform weights replaced by amounts, so the two numbers are directly comparable and
    any difference is attributable to the weighting alone.
    """
    y_true = np.asarray(y_true).astype(int)
    amounts = np.asarray(amounts, dtype=np.float64)
    if np.any(amounts < 0):
        raise ValueError("amounts must be non-negative to be used as weights")
    return float(average_precision_score(y_true, y_score, sample_weight=amounts))


def value_detection_rate(
    y_true: np.ndarray, y_score: np.ndarray, amounts: np.ndarray, threshold: float
) -> float:
    """Share of total fraud VALUE that would be caught at this threshold.

    VDR = (value of fraud scoring at or above the threshold) / (value of all fraud).

    The count-weighted analogue is recall. The difference between the two is the whole
    question: recall asks how many frauds you stopped, VDR asks how much of the money.
    """
    y_true = np.asarray(y_true).astype(int)
    amounts = np.asarray(amounts, dtype=np.float64)

    fraud = y_true == 1
    total_fraud_value = amounts[fraud].sum()
    if total_fraud_value <= 0:
        return 0.0

    caught = fraud & (np.asarray(y_score) >= threshold)
    return float(amounts[caught].sum() / total_fraud_value)


@dataclass
class ValueReport:
    """Value-weighted view of one model, alongside its count-weighted counterpart."""

    name: str
    auc_pr: float
    value_weighted_ap: float
    vdr_at_cap: float
    recall_at_cap: float
    threshold: float
    insult_rate: float

    def summary_lines(self) -> list[str]:
        return [
            f"--- {self.name} ---",
            f"  AUC-PR (count-weighted)   {self.auc_pr:.4f}",
            f"  AUC-PR (value-weighted)   {self.value_weighted_ap:.4f}",
            f"  at the <=1% insult cap (threshold {self.threshold:.4f}, "
            f"insult {100 * self.insult_rate:.3f}%):",
            f"    recall (count)          {self.recall_at_cap:.4f}",
            f"    VDR    (value)          {self.vdr_at_cap:.4f}",
        ]
