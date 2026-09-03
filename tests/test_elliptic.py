"""Tests for the Elliptic replication and the homophily statistic it required.

The interesting one is `test_component_statistic_is_degenerate_on_a_percolated_graph`.
Running the existing concentration test on Elliptic returned z = +0.0, which reads like a
clean null and is not one — on components averaging thousands of members, "every member is
illicit" is impossible, so observed and null are both zero and the test has no power at all.
Reporting that as "no concentration" would have been a false negative dressed as a finding.
These tests pin both the degeneracy and the statistic that replaced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.elliptic import (
    CLASSES_CSV,
    EDGELIST_CSV,
    EXPECTED_EDGES,
    EXPECTED_NODES,
    describe_graph,
    project_bipartite,
)
from core.ring_evidence import label_homophily_test, ring_concentration_test


# --------------------------------------------------------------------------
# homophily: known answers
# --------------------------------------------------------------------------


def clique(members: list[int], adjacency: list[list[int]]) -> None:
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            adjacency[a].append(b)
            adjacency[b].append(a)


def test_perfect_clustering_scores_high():
    """Illicit nodes wired only to each other must be far above the shuffle null."""
    n = 200
    adjacency: list[list[int]] = [[] for _ in range(n)]
    illicit = list(range(0, 40))
    licit = list(range(40, n))
    # Two separate cliques: all illicit together, all licit together.
    for group in (illicit, licit):
        for i in range(0, len(group) - 1):
            adjacency[group[i]].append(group[i + 1])
            adjacency[group[i + 1]].append(group[i])

    labels = np.zeros(n, dtype=int)
    labels[illicit] = 1
    evidence = label_homophily_test(adjacency, labels, np.ones(n, dtype=bool))

    assert evidence.z_score > 5
    assert evidence.observed_rate > evidence.null_mean


def test_randomly_scattered_labels_score_near_zero():
    """No clustering must produce no signal, or the statistic is not measuring clustering."""
    rng = np.random.default_rng(0)
    n = 400
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for _ in range(800):
        a, b = rng.integers(0, n, size=2)
        if a != b and b not in adjacency[a]:
            adjacency[a].append(int(b))
            adjacency[b].append(int(a))

    labels = (rng.random(n) < 0.2).astype(int)
    evidence = label_homophily_test(adjacency, labels, np.ones(n, dtype=bool))
    assert abs(evidence.z_score) < 3


def test_homophily_counts_only_edges_between_testable_nodes():
    """Unlabelled nodes must not enter the statistic, in either direction."""
    adjacency = [[1, 2], [0], [0]]
    labels = np.array([1, 1, 0])
    testable = np.array([True, False, True])  # node 1 is unlabelled

    evidence = label_homophily_test(adjacency, labels, testable)
    assert evidence.labelled_edges == 1  # only 0-2 qualifies
    assert evidence.illicit_illicit_edges == 0  # node 2 is licit


def test_homophily_handles_a_graph_with_no_testable_edges():
    """Returns nan rather than raising -- exactly what the bipartite graph first did."""
    evidence = label_homophily_test([[], [], []], np.array([1, 0, 1]), np.ones(3, bool))
    assert evidence.labelled_edges == 0
    assert np.isnan(evidence.observed_rate)


def test_homophily_is_deterministic():
    rng = np.random.default_rng(1)
    n = 150
    adjacency: list[list[int]] = [[] for _ in range(n)]
    clique(list(range(20)), adjacency)
    labels = (rng.random(n) < 0.3).astype(int)
    testable = np.ones(n, dtype=bool)

    first = label_homophily_test(adjacency, labels, testable, seed=7)
    second = label_homophily_test(adjacency, labels, testable, seed=7)
    assert first.z_score == second.z_score


# --------------------------------------------------------------------------
# why the component statistic had to be replaced
# --------------------------------------------------------------------------


def test_component_statistic_is_degenerate_on_a_percolated_graph():
    """The failure that motivated the homophily test, pinned so it cannot be forgotten.

    One giant component containing both illicit and licit nodes: no component can be
    entirely illicit, the permutation null also produces none, and the z-score is 0/0.
    That reads as "no concentration" while actually meaning the test could not have
    detected any. The homophily statistic, on the same graph, does have power.
    """
    n = 600
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n - 1):  # one long chain -> a single component
        adjacency[i].append(i + 1)
        adjacency[i + 1].append(i)

    # Illicit nodes deliberately bunched at one end: real clustering exists.
    labels = np.zeros(n, dtype=int)
    labels[:120] = 1
    components = np.zeros(n, dtype=np.int64)  # all one component
    testable = np.ones(n, dtype=bool)

    degenerate = ring_concentration_test(components, labels, testable)
    assert degenerate.all_fraud_components == 0
    assert degenerate.null_mean == 0.0
    assert abs(degenerate.z_score) < 1e-6  # zero from having no power, not from no effect

    # The same graph, measured with a statistic that works at this scale.
    powered = label_homophily_test(adjacency, labels, testable)
    assert powered.z_score > 3, "the replacement statistic must detect the real clustering"


# --------------------------------------------------------------------------
# bipartite projection
# --------------------------------------------------------------------------


def test_projection_links_entities_that_share_an_attribute():
    """Entities 0,1,2 all touch attribute node 3, so all pairs must be linked."""
    adjacency = [[3], [3], [3], [0, 1, 2]]
    projected = project_bipartite(adjacency, n_left=3)

    assert projected[0] == [1, 2]
    assert projected[1] == [0, 2]
    assert projected[2] == [0, 1]


def test_projection_leaves_unshared_entities_isolated():
    adjacency = [[2], [3], [0], [1]]  # two entities, two separate attributes
    projected = project_bipartite(adjacency, n_left=2)
    assert projected == [[], []]


def test_projection_skips_oversized_groups():
    """Guard against the quadratic blow-up if pointed at an unsuppressed graph."""
    members = list(range(10))
    adjacency: list[list[int]] = [[10] for _ in members] + [members]
    assert project_bipartite(adjacency, n_left=10, max_group=5) == [[] for _ in members]
    assert project_bipartite(adjacency, n_left=10, max_group=20)[0] != []


def test_projection_produces_no_self_loops():
    adjacency = [[2], [2], [0, 1]]
    for node, neighbours in enumerate(project_bipartite(adjacency, n_left=2)):
        assert node not in neighbours


# --------------------------------------------------------------------------
# graph shape
# --------------------------------------------------------------------------


def test_describe_graph_on_a_known_shape():
    """A triangle plus one isolated node: every figure checkable by hand."""
    adjacency = [[1, 2], [0, 2], [0, 1], []]
    shape = describe_graph("triangle", adjacency)

    assert shape.n_nodes == 4
    assert shape.n_edges == 3
    assert shape.mean_degree == pytest.approx(6 / 4)
    assert shape.isolated_fraction == pytest.approx(0.25)
    assert shape.largest_component == 3
    assert shape.max_core_number == 2  # every triangle member has degree 2 within it


# --------------------------------------------------------------------------
# the real dataset, when present
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (CLASSES_CSV.exists() and EDGELIST_CSV.exists()),
    reason="Elliptic not downloaded (run scripts/fetch_elliptic.py)",
)
def test_real_elliptic_matches_published_dimensions():
    from core.elliptic import load_elliptic

    graph = load_elliptic()
    assert graph.n_nodes == EXPECTED_NODES
    assert graph.n_edges == EXPECTED_EDGES
    # Roughly 2% illicit, 21% licit, 77% unknown.
    assert 0.01 < graph.is_illicit.mean() < 0.04
    assert 0.15 < graph.is_labelled.mean() < 0.30


@pytest.mark.skipif(
    not (CLASSES_CSV.exists() and EDGELIST_CSV.exists()),
    reason="Elliptic not downloaded (run scripts/fetch_elliptic.py)",
)
def test_illicit_activity_clusters_on_the_observed_graph():
    """The replication itself: clustering on a graph nobody had to infer."""
    from core.elliptic import load_elliptic

    graph = load_elliptic()
    degrees = np.array([len(n) for n in graph.adjacency], dtype=np.int64)
    testable = graph.is_labelled & (degrees > 0)

    evidence = label_homophily_test(
        graph.adjacency, graph.is_illicit.astype(int), testable
    )
    assert evidence.z_score > 5
