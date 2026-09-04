"""Tests for the retrospective per-cluster outcomes behind the small-multiples grid.

Two properties matter more than the arithmetic.

**The label must not reach the narrative layer.** `ClusterOutcome` exists as a separate
object precisely so the held-out fraud label stays out of `ClusterEvidence`. A narrator
handed ground truth would produce narratives that look accurate for a reason unrelated to
the evidence it was given, and the project's central claim would be hollow.

**`all_fraud` must not become a ring claim.** It means every transaction in the cluster
carries a fraud label. Three unrelated fraudsters sharing an address satisfy that and
coordinate nothing. The distinction is the one this project refuses to blur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.contract import ClusterEvidence
from core.clusters import ClusterOutcome, build_cluster_evidence, cluster_outcomes
from core.graph import UID_COL


def frame(rows: list[dict]) -> pd.DataFrame:
    """A minimal test-set shape: whatever build_cluster_evidence and outcomes both read."""
    return pd.DataFrame(
        [
            {
                UID_COL: row["uid"],
                "g_component": row["component"],
                "g_component_size": row.get("component_size", 4),
                "g_core_number": row.get("core", 2),
                "g_degree": row.get("degree", 2),
                "card1": row.get("card1", 1000),
                "addr1": row.get("addr1", 200),
                "P_emaildomain": row.get("email", "a.com"),
                "TransactionDT": row.get("dt", 0),
                "TransactionAmt": row.get("amt", 100.0),
            }
            for row in rows
        ]
    )


# --------------------------------------------------------------------------
# the label stays out of the contract
# --------------------------------------------------------------------------


def test_cluster_evidence_carries_no_ground_truth_field():
    """The load-bearing boundary test for this feature."""
    fields = set(ClusterEvidence.__dataclass_fields__)
    for forbidden in (
        "fraud_transactions", "all_fraud", "is_fraud", "fraud_share",
        "label", "y_true", "caught", "missed", "false_alarms",
    ):
        assert forbidden not in fields, f"ClusterEvidence leaks ground truth: {forbidden}"


def test_outcomes_are_a_separate_object_from_evidence():
    assert ClusterOutcome is not ClusterEvidence
    assert "fraud_transactions" in ClusterOutcome.__dataclass_fields__


def test_a_cluster_outcome_is_immutable():
    outcome = ClusterOutcome(0, 2, 2, 1.0, True, 2, 0, 0)
    with pytest.raises(Exception):
        outcome.all_fraud = False  # type: ignore[misc]


# --------------------------------------------------------------------------
# arithmetic
# --------------------------------------------------------------------------


def test_all_fraud_is_true_only_when_every_transaction_is_labelled_fraud():
    test = frame([
        {"uid": "a", "component": 1}, {"uid": "b", "component": 1},
        {"uid": "c", "component": 2}, {"uid": "d", "component": 2},
    ])
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    y_true = np.array([1, 1, 1, 0])

    outcomes = cluster_outcomes(test, scores, y_true, 0.5, [1, 2])
    assert outcomes[0].all_fraud is True
    assert outcomes[1].all_fraud is False
    assert outcomes[0].fraud_share == pytest.approx(1.0)
    assert outcomes[1].fraud_share == pytest.approx(0.5)


def test_an_empty_cluster_is_not_vacuously_all_fraud():
    """`all(x)` over nothing is True, and that would badge a phantom cluster."""
    test = frame([{"uid": "a", "component": 1}])
    outcomes = cluster_outcomes(test, np.array([0.9]), np.array([1]), 0.5, [99])
    assert outcomes[0].transaction_count == 0
    assert outcomes[0].all_fraud is False
    assert outcomes[0].fraud_share == 0.0


def test_caught_and_missed_partition_the_fraud():
    test = frame([{"uid": c, "component": 1} for c in "abcd"])
    scores = np.array([0.9, 0.1, 0.9, 0.1])
    y_true = np.array([1, 1, 0, 0])

    outcome = cluster_outcomes(test, scores, y_true, 0.5, [1])[0]
    assert outcome.fraud_transactions == 2
    assert outcome.caught == 1
    assert outcome.missed == 1
    assert outcome.caught + outcome.missed == outcome.fraud_transactions
    assert outcome.false_alarms == 1  # one non-fraud row scored above the threshold


def test_a_cluster_with_no_fraud_is_reported_as_such():
    """A pure false alarm has to be visible, or the grid is a highlight reel."""
    test = frame([{"uid": c, "component": 1} for c in "ab"])
    outcome = cluster_outcomes(test, np.array([0.9, 0.9]), np.array([0, 0]), 0.5, [1])[0]
    assert outcome.fraud_transactions == 0
    assert outcome.all_fraud is False
    assert outcome.false_alarms == 2


def test_the_threshold_is_inclusive_matching_the_rest_of_the_engine():
    test = frame([{"uid": "a", "component": 1}])
    outcome = cluster_outcomes(test, np.array([0.5]), np.array([1]), 0.5, [1])[0]
    assert outcome.caught == 1


# --------------------------------------------------------------------------
# alignment with the selection
# --------------------------------------------------------------------------


def test_outcomes_follow_the_order_of_selected_components():
    test = frame([
        {"uid": "a", "component": 7}, {"uid": "b", "component": 7},
        {"uid": "c", "component": 3}, {"uid": "d", "component": 3},
        {"uid": "e", "component": 3},
    ])
    scores = np.full(5, 0.9)
    y_true = np.array([1, 1, 0, 0, 0])

    outcomes = cluster_outcomes(test, scores, y_true, 0.5, [3, 7])
    assert [o.cluster_id for o in outcomes] == [0, 1]
    assert outcomes[0].transaction_count == 3  # component 3 first, as ordered
    assert outcomes[1].transaction_count == 2


def test_outcomes_and_evidence_describe_the_same_rows():
    """The guard against two independent derivations of 'which cluster' drifting apart."""
    test = frame([
        {"uid": "a", "component": 1, "dt": 0},
        {"uid": "b", "component": 1, "dt": 86_400},
        {"uid": "c", "component": 2, "dt": 0},
        {"uid": "d", "component": 2, "dt": 0},
        {"uid": "e", "component": 2, "dt": 0},
    ])
    scores = np.array([0.9, 0.8, 0.95, 0.7, 0.6])
    y_true = np.array([1, 0, 1, 1, 0])

    selected: list[int] = []
    evidence = build_cluster_evidence(test, scores, 0.5, selected_components=selected)
    outcomes = cluster_outcomes(test, scores, y_true, 0.5, selected)

    assert len(evidence) == len(outcomes)
    for ev, out in zip(evidence, outcomes):
        assert ev.cluster_id == out.cluster_id
        assert ev.transaction_count == out.transaction_count


def test_no_selected_components_yields_no_outcomes():
    test = frame([{"uid": "a", "component": 1}])
    assert cluster_outcomes(test, np.array([0.9]), np.array([1]), 0.5, []) == []
