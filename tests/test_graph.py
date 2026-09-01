"""Tests for the deterministic graph layer.

The k-core implementation is hand-written Batagelj-Zaversnik peeling, so it is validated
against networkx.core_number as an independent oracle -- on hand-built graphs with known
answers, on pathological shapes, and on randomised graphs.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from core.graph import (
    MAX_GROUP_SIZE,
    UID_COL,
    build_graph,
    build_uid,
    connected_components,
    graph_features,
    k_core_numbers,
)


def to_networkx(adjacency: list[list[int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(adjacency)))
    for node, neighbours in enumerate(adjacency):
        for neighbour in neighbours:
            graph.add_edge(node, neighbour)
    return graph


def from_networkx(graph: nx.Graph) -> list[list[int]]:
    return [sorted(graph.neighbors(node)) for node in range(graph.number_of_nodes())]


# --------------------------------------------------------------------------
# k-core: known answers
# --------------------------------------------------------------------------


def test_kcore_empty_graph():
    assert len(k_core_numbers([])) == 0


def test_kcore_isolated_nodes_are_zero_core():
    assert k_core_numbers([[], [], []]).tolist() == [0, 0, 0]


def test_kcore_single_edge_is_one_core():
    assert k_core_numbers([[1], [0]]).tolist() == [1, 1]


def test_kcore_triangle_is_two_core():
    """Every node of K3 has degree 2 within the subgraph, so core number 2."""
    triangle = [[1, 2], [0, 2], [0, 1]]
    assert k_core_numbers(triangle).tolist() == [2, 2, 2]


def test_kcore_complete_graph_is_n_minus_one():
    for n in (2, 4, 7):
        adjacency = from_networkx(nx.complete_graph(n))
        assert k_core_numbers(adjacency).tolist() == [n - 1] * n


def test_kcore_star_is_all_ones():
    """A hub with leaves: everything peels to core 1, no dense region exists."""
    adjacency = from_networkx(nx.star_graph(6))
    assert set(k_core_numbers(adjacency).tolist()) == {1}


def test_kcore_path_is_all_ones():
    adjacency = from_networkx(nx.path_graph(5))
    assert k_core_numbers(adjacency).tolist() == [1, 1, 1, 1, 1]


def test_kcore_triangle_with_pendant_tail():
    """The classic discriminating case: a dense core plus a chain that peels away.

    Nodes 0,1,2 form a triangle (core 2); node 3 hangs off node 2 and node 4 off node 3,
    so both peel to core 1. A degree-based heuristic would wrongly call node 2 special;
    core number correctly separates the triangle from the tail.
    """
    adjacency = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3]]
    assert k_core_numbers(adjacency).tolist() == [2, 2, 2, 1, 1]


# --------------------------------------------------------------------------
# k-core: validated against networkx on random and structured graphs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
def test_kcore_matches_networkx_on_random_graphs(seed):
    graph = nx.gnm_random_graph(60, 180, seed=seed)
    adjacency = from_networkx(graph)
    expected = nx.core_number(graph)
    actual = k_core_numbers(adjacency)
    assert actual.tolist() == [expected[i] for i in range(graph.number_of_nodes())]


@pytest.mark.parametrize(
    "graph",
    [
        nx.barbell_graph(6, 3),
        nx.karate_club_graph(),
        nx.complete_bipartite_graph(4, 7),
        nx.wheel_graph(10),
        nx.lollipop_graph(5, 5),
        nx.circular_ladder_graph(8),
    ],
    ids=["barbell", "karate", "bipartite", "wheel", "lollipop", "ladder"],
)
def test_kcore_matches_networkx_on_pathological_graphs(graph):
    graph = nx.convert_node_labels_to_integers(graph)
    adjacency = from_networkx(graph)
    expected = nx.core_number(graph)
    actual = k_core_numbers(adjacency)
    assert actual.tolist() == [expected[i] for i in range(graph.number_of_nodes())]


def test_kcore_matches_networkx_on_sparse_disconnected_graph():
    """Shaped like the real entity graph: many tiny components, mostly isolated nodes."""
    graph = nx.gnm_random_graph(500, 120, seed=3)
    adjacency = from_networkx(graph)
    expected = nx.core_number(graph)
    assert k_core_numbers(adjacency).tolist() == [expected[i] for i in range(500)]


def test_kcore_is_deterministic():
    graph = nx.gnm_random_graph(80, 240, seed=11)
    adjacency = from_networkx(graph)
    first = k_core_numbers(adjacency)
    second = k_core_numbers(adjacency)
    assert np.array_equal(first, second)


# --------------------------------------------------------------------------
# connected components
# --------------------------------------------------------------------------


def test_components_on_disjoint_pairs():
    labels = connected_components([[1], [0], [3], [2]])
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_components_isolated_nodes_are_singletons():
    labels = connected_components([[], [], []])
    assert len(set(labels.tolist())) == 3


def test_components_match_networkx():
    graph = nx.gnm_random_graph(200, 150, seed=5)
    adjacency = from_networkx(graph)
    labels = connected_components(adjacency)

    expected = {frozenset(c) for c in nx.connected_components(graph)}
    grouped: dict[int, set[int]] = {}
    for node, label in enumerate(labels):
        grouped.setdefault(int(label), set()).add(node)
    assert {frozenset(v) for v in grouped.values()} == expected


def test_components_label_is_smallest_member():
    """Canonical labelling makes runs comparable regardless of union order."""
    labels = connected_components([[1, 2], [0], [0], [4], [3]])
    assert labels[0] == labels[1] == labels[2] == 0
    assert labels[3] == labels[4] == 3


# --------------------------------------------------------------------------
# uid construction and graph building
# --------------------------------------------------------------------------


def make_entity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UID_COL: [f"u{i}" for i in range(6)],
            "card1": [1000, 1000, 2000, 3000, 4000, 5000],
            "addr1": [204, 204, 204, 299, 299, 325],
            "P_emaildomain": ["a.com", "a.com", "b.com", "c.com", "c.com", "d.com"],
            "card3": [150] * 6,
            "card5": [226] * 6,
        }
    )


def test_build_uid_is_na_when_components_missing():
    df = pd.DataFrame(
        {
            "TransactionDT": [86_400, 86_400, 86_400],
            "card1": [1000, np.nan, 1000],
            "addr1": [204, 204, np.nan],
            "D1": [0.0, 0.0, 0.0],
        }
    )
    uid = build_uid(df)
    assert pd.notna(uid.iloc[0])
    assert pd.isna(uid.iloc[1])
    assert pd.isna(uid.iloc[2])


def test_build_uid_same_account_across_days_is_stable():
    """Two transactions from one card on different days must yield the SAME uid.

    D1 counts days since the card was first seen, so as the day advances D1 advances with
    it and (day - D1) stays put. This invariant is the whole basis of the fingerprint.
    """
    df = pd.DataFrame(
        {
            "TransactionDT": [10 * 86_400, 40 * 86_400],
            "card1": [1000, 1000],
            "addr1": [204, 204],
            "D1": [5.0, 35.0],
        }
    )
    uid = build_uid(df)
    assert uid.iloc[0] == uid.iloc[1]


def test_build_uid_differs_for_different_start_days():
    df = pd.DataFrame(
        {
            "TransactionDT": [10 * 86_400, 10 * 86_400],
            "card1": [1000, 1000],
            "addr1": [204, 204],
            "D1": [5.0, 6.0],
        }
    )
    uid = build_uid(df)
    assert uid.iloc[0] != uid.iloc[1]


def test_build_graph_links_entities_sharing_an_attribute():
    graph = build_graph(make_entity_frame())
    labels = connected_components(graph.adjacency)
    index = graph.uid_index()
    # u0 and u1 share card1, addr1 and email domain -- must land together.
    assert labels[index["u0"]] == labels[index["u1"]]
    # u5 shares nothing with them.
    assert labels[index["u5"]] != labels[index["u0"]]


def test_hub_suppression_drops_oversized_groups():
    """A value shared by more entities than the cap is generic, and must create no edges."""
    n = MAX_GROUP_SIZE + 3
    frame = pd.DataFrame(
        {
            UID_COL: [f"u{i}" for i in range(n)],
            "card1": [7000] * n,  # one hub value held by everyone
            "addr1": [204] * n,
            "P_emaildomain": ["hub.com"] * n,
            "card3": [150] * n,
            "card5": [226] * n,
        }
    )
    graph = build_graph(frame)
    assert graph.n_edges == 0
    assert graph.dropped_hub_values > 0
    # With every group suppressed, no entity is linked to any other.
    labels = connected_components(graph.adjacency)
    assert len(set(labels.tolist())) == n


def test_graph_is_independent_of_input_row_order():
    """Shuffling the entity frame must not change any entity's features."""
    frame = make_entity_frame()
    ordered = graph_features(build_graph(frame)).set_index(UID_COL).sort_index()
    shuffled = (
        graph_features(build_graph(frame.sample(frac=1.0, random_state=3)))
        .set_index(UID_COL)
        .sort_index()
    )
    pd.testing.assert_frame_equal(ordered, shuffled)


def test_graph_build_is_deterministic():
    frame = make_entity_frame()
    first = graph_features(build_graph(frame))
    second = graph_features(build_graph(frame))
    pd.testing.assert_frame_equal(first, second)


def test_graph_features_contain_no_label_derived_column():
    """Guard against the leakage trap: no graph feature may depend on isFraud."""
    features = graph_features(build_graph(make_entity_frame()))
    for column in features.columns:
        assert "fraud" not in column.lower()
        assert "label" not in column.lower()


def test_isolated_entity_has_zero_graph_features():
    features = graph_features(build_graph(make_entity_frame())).set_index(UID_COL)
    assert features.loc["u5", "g_degree"] == 0
    assert features.loc["u5", "g_core_number"] == 0
    assert features.loc["u5", "g_is_linked"] == 0
