"""Correctness of the hand-written PageRank and Brandes betweenness.

Same oracle pattern as `tests/test_graph.py` uses for k-core: networkx is an independent
implementation, so agreement with it is real evidence rather than a restatement of my own
assumptions. Known-answer cases come first, because a bug that networkx happens to share
would slip past comparison alone.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from core.centrality import (
    betweenness,
    component_centrality_variance,
    pagerank,
)


def from_networkx(graph: nx.Graph) -> list[list[int]]:
    graph = nx.convert_node_labels_to_integers(graph)
    return [sorted(graph.neighbors(node)) for node in range(graph.number_of_nodes())]


# --------------------------------------------------------------------------
# PageRank: known answers
# --------------------------------------------------------------------------


def test_pagerank_empty_graph():
    assert len(pagerank([])) == 0


def test_pagerank_sums_to_one():
    """The defining property: it is a probability distribution over vertices."""
    for graph in (nx.gnm_random_graph(40, 90, seed=1), nx.karate_club_graph()):
        assert pagerank(from_networkx(graph)).sum() == pytest.approx(1.0)


def test_pagerank_uniform_on_a_symmetric_graph():
    """Every node of a cycle is structurally identical, so all ranks must be equal."""
    ranks = pagerank(from_networkx(nx.cycle_graph(8)))
    assert np.allclose(ranks, ranks[0])


def test_pagerank_isolated_nodes_still_sum_to_one():
    """Degree-0 nodes leak probability mass unless the dangling term is handled.

    Without redistributing dangling mass the vector decays below 1 every iteration — the
    single most common PageRank implementation bug.
    """
    adjacency = [[1], [0], [], []]
    ranks = pagerank(adjacency)
    assert ranks.sum() == pytest.approx(1.0)
    assert (ranks > 0).all()


def test_pagerank_hub_outranks_leaves():
    """In a star, the centre must rank highest."""
    ranks = pagerank(from_networkx(nx.star_graph(6)))
    assert ranks[0] == max(ranks)
    assert ranks[0] > ranks[1] * 2


# --------------------------------------------------------------------------
# PageRank: against networkx
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_pagerank_matches_networkx_on_random_graphs(seed):
    graph = nx.convert_node_labels_to_integers(nx.gnm_random_graph(50, 150, seed=seed))
    # weight=None is essential, not cosmetic. networkx's pagerank defaults to
    # weight="weight", and some built-in graphs (karate club especially) carry edge
    # weights -- so the default oracle would compute WEIGHTED PageRank and disagree with
    # this unweighted implementation by ~0.01. The entity graph is unweighted (an edge
    # exists or it does not), so unweighted is the correct comparison. Caught exactly this
    # way: every random graph passed, karate alone failed.
    expected = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=200, weight=None)
    actual = pagerank(from_networkx(graph))
    for node in range(graph.number_of_nodes()):
        assert actual[node] == pytest.approx(expected[node], abs=1e-6)


@pytest.mark.parametrize(
    "graph",
    [
        nx.karate_club_graph(),
        nx.barbell_graph(5, 2),
        nx.wheel_graph(9),
        nx.lollipop_graph(5, 4),
        nx.complete_bipartite_graph(3, 5),
        nx.path_graph(10),
    ],
    ids=["karate", "barbell", "wheel", "lollipop", "bipartite", "path"],
)
def test_pagerank_matches_networkx_on_pathological_graphs(graph):
    graph = nx.convert_node_labels_to_integers(graph)
    # weight=None is essential, not cosmetic. networkx's pagerank defaults to
    # weight="weight", and some built-in graphs (karate club especially) carry edge
    # weights -- so the default oracle would compute WEIGHTED PageRank and disagree with
    # this unweighted implementation by ~0.01. The entity graph is unweighted (an edge
    # exists or it does not), so unweighted is the correct comparison. Caught exactly this
    # way: every random graph passed, karate alone failed.
    expected = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=200, weight=None)
    actual = pagerank(from_networkx(graph))
    for node in range(graph.number_of_nodes()):
        assert actual[node] == pytest.approx(expected[node], abs=1e-6)


def test_pagerank_is_deterministic():
    adjacency = from_networkx(nx.gnm_random_graph(30, 80, seed=2))
    assert np.array_equal(pagerank(adjacency), pagerank(adjacency))


# --------------------------------------------------------------------------
# Betweenness: known answers
# --------------------------------------------------------------------------


def test_betweenness_path_graph_centre_is_highest():
    """On a path, the middle vertex lies on the most shortest paths."""
    scores = betweenness(from_networkx(nx.path_graph(5)))
    assert scores[2] == max(scores)
    assert scores[0] == 0.0 and scores[4] == 0.0


def test_betweenness_complete_graph_is_all_zero():
    """Everyone is directly connected, so nobody is ever an intermediary."""
    assert np.allclose(betweenness(from_networkx(nx.complete_graph(6))), 0.0)


def test_betweenness_star_centre_is_one():
    """Every shortest path between leaves goes through the hub; normalised, that is 1."""
    scores = betweenness(from_networkx(nx.star_graph(5)))
    assert scores[0] == pytest.approx(1.0)
    assert np.allclose(scores[1:], 0.0)


def test_betweenness_tiny_graphs_are_zero():
    """Fewer than three vertices means nothing can lie between anything."""
    assert np.allclose(betweenness([[1], [0]]), 0.0)
    assert len(betweenness([])) == 0


# --------------------------------------------------------------------------
# Betweenness: against networkx
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_betweenness_matches_networkx_on_random_graphs(seed):
    graph = nx.convert_node_labels_to_integers(nx.gnm_random_graph(40, 110, seed=seed))
    expected = nx.betweenness_centrality(graph, normalized=True, weight=None)
    actual = betweenness(from_networkx(graph))
    for node in range(graph.number_of_nodes()):
        assert actual[node] == pytest.approx(expected[node], abs=1e-9)


@pytest.mark.parametrize(
    "graph",
    [
        nx.karate_club_graph(),
        nx.barbell_graph(5, 3),
        nx.wheel_graph(9),
        nx.lollipop_graph(5, 4),
        nx.complete_bipartite_graph(3, 5),
        nx.circular_ladder_graph(6),
    ],
    ids=["karate", "barbell", "wheel", "lollipop", "bipartite", "ladder"],
)
def test_betweenness_matches_networkx_on_pathological_graphs(graph):
    graph = nx.convert_node_labels_to_integers(graph)
    expected = nx.betweenness_centrality(graph, normalized=True, weight=None)
    actual = betweenness(from_networkx(graph))
    for node in range(graph.number_of_nodes()):
        assert actual[node] == pytest.approx(expected[node], abs=1e-9)


def test_betweenness_matches_networkx_on_a_disconnected_graph():
    """Shaped like the real entity graph: many small components, many isolated nodes."""
    graph = nx.convert_node_labels_to_integers(nx.gnm_random_graph(200, 90, seed=4))
    expected = nx.betweenness_centrality(graph, normalized=True, weight=None)
    actual = betweenness(from_networkx(graph))
    for node in range(200):
        assert actual[node] == pytest.approx(expected[node], abs=1e-9)


# --------------------------------------------------------------------------
# the variance diagnostic
# --------------------------------------------------------------------------


def test_variance_diagnostic_reports_near_zero_on_uniform_components():
    """Symmetric components: PageRank is identical within each, so CV must be ~0.

    This is the shape the real entity graph is predicted to have.
    """
    labels = np.array([0, 0, 0, 1, 1, 1])
    ranks = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    between = np.zeros(6)
    linked = np.ones(6, dtype=bool)

    result = component_centrality_variance(labels, ranks, between, linked)
    assert result.n_components == 2
    assert result.median_pagerank_cv == pytest.approx(0.0, abs=1e-9)
    assert result.zero_betweenness_fraction == 1.0


def test_variance_diagnostic_detects_real_variation():
    """A component with a genuine hub must show a non-trivial coefficient of variation."""
    labels = np.array([0, 0, 0, 0])
    ranks = np.array([0.70, 0.10, 0.10, 0.10])
    between = np.array([1.0, 0.0, 0.0, 0.0])
    linked = np.ones(4, dtype=bool)

    result = component_centrality_variance(labels, ranks, between, linked)
    assert result.median_pagerank_cv > 0.5
    assert result.zero_betweenness_fraction == 0.75


def test_variance_diagnostic_ignores_unlinked_entities():
    labels = np.array([0, 0, 1, 2])
    ranks = np.array([0.3, 0.3, 0.2, 0.2])
    between = np.zeros(4)
    linked = np.array([True, True, False, False])

    result = component_centrality_variance(labels, ranks, between, linked)
    assert result.n_components == 1
