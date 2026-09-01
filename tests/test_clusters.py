"""Tests for cluster selection and evidence packaging.

Cluster selection is the last decision made on the deterministic side of the boundary, so
it must be reproducible and must never hand the narrative layer a number that the layer
is then forbidden to mention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai.contract import extract_numbers
from core.clusters import build_cluster_evidence
from core.graph import UID_COL


def make_test_frame() -> pd.DataFrame:
    """Two multi-entity clusters plus one isolated entity."""
    return pd.DataFrame(
        {
            UID_COL: ["a", "a", "b", "c", "c", "d", "e"],
            "g_component": [10, 10, 10, 20, 20, 20, 30],
            "g_component_size": [2, 2, 2, 2, 2, 2, 1],
            "g_core_number": [1, 1, 1, 2, 2, 2, 0],
            "g_degree": [1, 1, 1, 2, 2, 2, 0],
            "TransactionDT": [86_400, 172_800, 259_200, 86_400, 86_400, 86_400, 86_400],
            "TransactionAmt": [100.0, 200.0, 50.0, 10.0, 20.0, 30.0, 500.0],
            "card1": [1000, 1000, 1000, 2000, 2001, 2002, 3000],
            "addr1": [204, 204, 204, 299, 299, 299, 325],
            "P_emaildomain": ["a.com"] * 3 + ["b.com"] * 3 + ["c.com"],
        }
    )


def test_selects_only_multi_entity_clusters():
    frame = make_test_frame()
    scores = np.array([0.9, 0.8, 0.7, 0.95, 0.4, 0.3, 0.99])
    evidence = build_cluster_evidence(frame, scores, threshold=0.5)

    # Component 30 is a single isolated entity and must not be offered to the analyst,
    # despite carrying the highest score in the frame.
    assert len(evidence) == 2
    assert all(item.entity_count >= 2 for item in evidence)


def test_clusters_below_threshold_are_not_flagged():
    frame = make_test_frame()
    scores = np.full(len(frame), 0.01)
    assert build_cluster_evidence(frame, scores, threshold=0.5) == []


def test_clusters_ranked_by_peak_score():
    frame = make_test_frame()
    scores = np.array([0.6, 0.6, 0.6, 0.95, 0.4, 0.3, 0.1])
    evidence = build_cluster_evidence(frame, scores, threshold=0.5)
    assert evidence[0].max_risk_score > evidence[1].max_risk_score


def test_evidence_counts_are_accurate():
    frame = make_test_frame()
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1])
    evidence = build_cluster_evidence(frame, scores, threshold=0.5)

    cluster = evidence[0]
    assert cluster.transaction_count == 3
    assert cluster.flagged_transaction_count == 2  # only 0.9 and 0.8 clear 0.5
    assert cluster.entity_count == 2  # uids 'a' and 'b'
    assert cluster.span_days == 2


def test_shared_attributes_are_detected():
    frame = make_test_frame()
    scores = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    evidence = build_cluster_evidence(frame, scores, threshold=0.5)
    assert "card1" in evidence[0].shared_attributes
    assert "addr1" in evidence[0].shared_attributes


def test_every_evidence_number_is_quotable():
    """The layer must never be handed a figure it is then forbidden to mention.

    If a number appears in the prompt facts but not in allowed_numbers, a model that
    faithfully quotes the evidence would be rejected for hallucinating -- the worst
    possible failure, because it punishes correct behaviour.
    """
    frame = make_test_frame()
    scores = np.array([0.9, 0.8, 0.7, 0.95, 0.4, 0.3, 0.99])
    for cluster in build_cluster_evidence(frame, scores, threshold=0.5):
        shown = extract_numbers(cluster.as_prompt_facts())
        allowed = cluster.allowed_numbers()
        assert shown <= allowed, f"unquotable figures in prompt: {sorted(shown - allowed)}"


def test_selection_is_deterministic():
    frame = make_test_frame()
    scores = np.array([0.9, 0.8, 0.7, 0.95, 0.4, 0.3, 0.99])
    first = build_cluster_evidence(frame, scores, threshold=0.5)
    second = build_cluster_evidence(frame, scores, threshold=0.5)
    assert first == second
