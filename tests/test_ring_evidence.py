"""Tests for the ring-concentration permutation test.

The test must fire on planted structure and stay quiet on noise. A statistic that reports
significance on randomly-scattered labels would invalidate the project's central claim,
so both directions are asserted.
"""

from __future__ import annotations

import numpy as np

from core.ring_evidence import ring_concentration_test


def build_components(sizes: list[int]) -> np.ndarray:
    """Component label array for components of the given sizes."""
    labels = []
    for component_id, size in enumerate(sizes):
        labels.extend([component_id] * size)
    return np.array(labels, dtype=np.int64)


def test_scattered_labels_are_not_significant():
    """Fraud spread evenly across components must NOT look like rings."""
    rng = np.random.default_rng(0)
    labels = build_components([3] * 300)
    fraud = (rng.random(len(labels)) < 0.1).astype(np.int64)
    linked = np.ones(len(labels), dtype=bool)

    evidence = ring_concentration_test(labels, fraud, linked)
    assert abs(evidence.z_score) < 3.0


def test_planted_rings_are_significant():
    """Fraud confined to whole components must be detected."""
    labels = build_components([4] * 200)
    fraud = np.zeros(len(labels), dtype=np.int64)
    # Make the first 15 components entirely fraudulent.
    fraud[: 15 * 4] = 1
    linked = np.ones(len(labels), dtype=bool)

    evidence = ring_concentration_test(labels, fraud, linked)
    assert evidence.all_fraud_components == 15
    assert evidence.z_score > 5.0


def test_singleton_components_are_excluded():
    """Components of size 1 are trivially 'all fraud' and must not be counted."""
    labels = build_components([1] * 50 + [3])
    fraud = np.ones(len(labels), dtype=np.int64)
    linked = np.ones(len(labels), dtype=bool)

    evidence = ring_concentration_test(labels, fraud, linked)
    assert evidence.components == 1


def test_counts_partition_the_components():
    rng = np.random.default_rng(3)
    labels = build_components([3] * 100)
    fraud = (rng.random(len(labels)) < 0.3).astype(np.int64)
    linked = np.ones(len(labels), dtype=bool)

    evidence = ring_concentration_test(labels, fraud, linked)
    total = (
        evidence.all_fraud_components
        + evidence.mixed_components
        + evidence.clean_components
    )
    assert total == evidence.components


def test_is_deterministic_given_seed():
    rng = np.random.default_rng(1)
    labels = build_components([4] * 100)
    fraud = (rng.random(len(labels)) < 0.2).astype(np.int64)
    linked = np.ones(len(labels), dtype=bool)

    first = ring_concentration_test(labels, fraud, linked, seed=7)
    second = ring_concentration_test(labels, fraud, linked, seed=7)
    assert first.z_score == second.z_score


def test_unlinked_entities_are_ignored():
    """Isolated entities carry no topology and must not enter the statistic."""
    labels = build_components([3, 3])
    fraud = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    linked = np.array([True, True, True, False, False, False])

    evidence = ring_concentration_test(labels, fraud, linked)
    assert evidence.linked_entities == 3
    assert evidence.components == 1
    assert evidence.all_fraud_components == 1
