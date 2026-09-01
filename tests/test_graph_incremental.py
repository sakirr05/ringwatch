"""Correctness of incremental k-core maintenance.

The requirement is exact equality with a full batch rebuild, at every step — not
approximate, not eventually. `core/graph.py` is the oracle, and it is itself validated
against networkx in `tests/test_graph.py`, so the chain of trust is:

    networkx  ->  core/graph.py (batch)  ->  core/graph_incremental.py

If incremental and batch ever disagree, that is a bug in the incremental implementation.
The assertion does not get loosened.
"""

from __future__ import annotations

import networkx as nx
import pytest

from core.graph import k_core_numbers
from core.graph_incremental import IncrementalCore, build_incrementally


def edges_of(graph: nx.Graph) -> list[tuple[int, int]]:
    graph = nx.convert_node_labels_to_integers(graph)
    return [(int(u), int(v)) for u, v in graph.edges()]


# --------------------------------------------------------------------------
# tier 1: random graphs, asserted after EVERY insertion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_matches_batch_after_every_single_insertion(seed):
    """The strict version: not just the final state, but every intermediate one."""
    source = nx.gnm_random_graph(40, 110, seed=seed)
    graph = IncrementalCore(adjacency=[[] for _ in range(40)], core=[0] * 40)

    for step, (u, v) in enumerate(edges_of(source)):
        graph.insert_edge(u, v)
        assert graph.core == graph.batch_core(), (
            f"diverged from batch after insertion {step} of edge ({u}, {v})"
        )


@pytest.mark.parametrize("seed", range(6))
def test_matches_batch_on_denser_graphs(seed):
    """Denser graphs exercise deeper cores and longer eviction cascades."""
    source = nx.gnm_random_graph(60, 420, seed=seed)
    graph = build_incrementally(60, edges_of(source))
    assert graph.core == graph.batch_core()
    assert max(graph.core) > 3, "graph too shallow to exercise the cascade"


def test_insertion_order_does_not_affect_the_result():
    """Core numbers are a property of the graph, not of arrival order."""
    source = nx.gnm_random_graph(45, 200, seed=3)
    edges = edges_of(source)

    forward = build_incrementally(45, edges)
    backward = build_incrementally(45, list(reversed(edges)))

    assert forward.core == backward.core
    assert forward.core == list(k_core_numbers(forward.adjacency))


# --------------------------------------------------------------------------
# tier 2: pathological structures (reusing the batch suite's fixtures)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "graph",
    [
        nx.barbell_graph(6, 3),
        nx.karate_club_graph(),
        nx.complete_bipartite_graph(4, 7),
        nx.wheel_graph(10),
        nx.lollipop_graph(5, 5),
        nx.circular_ladder_graph(8),
        nx.complete_graph(8),
        nx.star_graph(12),
        nx.path_graph(15),
    ],
    ids=[
        "barbell", "karate", "bipartite", "wheel", "lollipop",
        "ladder", "complete", "star", "path",
    ],
)
def test_matches_batch_on_pathological_structures(graph):
    graph = nx.convert_node_labels_to_integers(graph)
    incremental = build_incrementally(graph.number_of_nodes(), edges_of(graph))

    expected = nx.core_number(graph)
    assert incremental.core == [expected[i] for i in range(graph.number_of_nodes())]


def test_barbell_bridge_insertion_promotes_nobody():
    """A bridge joins two dense blobs but creates no new dense region.

    Both cliques are already 4-cores; the connecting edge gives each endpoint one extra
    neighbour but cannot lift either into a 5-core, because the neighbour is not itself
    promoted. A naive implementation that counted every same-core neighbour would wrongly
    promote here.
    """
    graph = nx.convert_node_labels_to_integers(nx.barbell_graph(5, 0))
    incremental = build_incrementally(graph.number_of_nodes(), edges_of(graph))
    assert incremental.core == list(k_core_numbers(incremental.adjacency))


# --------------------------------------------------------------------------
# tier 3: the real IEEE-CIS entity graph
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_matches_batch_on_the_real_entity_graph():
    """Replay the real graph's edges and assert the final state matches a batch build.

    Skipped unless the dataset cache exists, since it needs the 683 MB download.
    """
    from core.data import MERGED_PARQUET

    if not MERGED_PARQUET.exists():
        pytest.skip("dataset cache not built")

    from core.data import load_merged
    from core.graph import UID_COL, build_graph, build_uid, entity_frame

    df = load_merged()
    df[UID_COL] = build_uid(df)
    batch = build_graph(entity_frame(df))

    edges: list[tuple[int, int]] = []
    for node, neighbours in enumerate(batch.adjacency):
        for neighbour in neighbours:
            if node < neighbour:
                edges.append((node, neighbour))

    incremental = build_incrementally(len(batch.adjacency), edges)
    assert incremental.core == list(k_core_numbers(batch.adjacency))


# --------------------------------------------------------------------------
# structural guarantees
# --------------------------------------------------------------------------


def test_isolated_nodes_have_core_zero():
    graph = IncrementalCore(adjacency=[[] for _ in range(5)], core=[0] * 5)
    assert graph.core == [0, 0, 0, 0, 0]


def test_single_edge_creates_a_one_core():
    graph = IncrementalCore(adjacency=[[], []], core=[0, 0])
    assert graph.insert_edge(0, 1) is True
    assert graph.core == [1, 1]


def test_triangle_reaches_core_two_on_the_closing_edge():
    """The first two edges build a path; only the third creates a 2-core."""
    graph = IncrementalCore(adjacency=[[], [], []], core=[0, 0, 0])
    graph.insert_edge(0, 1)
    graph.insert_edge(1, 2)
    assert graph.core == [1, 1, 1]
    assert graph.insert_edge(0, 2) is True
    assert graph.core == [2, 2, 2]


def test_duplicate_edge_is_ignored():
    """Re-inserting an existing edge must not inflate degrees."""
    graph = IncrementalCore(adjacency=[[], []], core=[0, 0])
    graph.insert_edge(0, 1)
    assert graph.insert_edge(0, 1) is False
    assert graph.adjacency[0] == [1]
    assert graph.core == graph.batch_core()


def test_self_loop_is_ignored():
    graph = IncrementalCore(adjacency=[[], []], core=[0, 0])
    assert graph.insert_edge(0, 0) is False
    assert graph.core == [0, 0]


def test_nodes_are_created_on_demand():
    """Streaming graphs grow; inserting an edge to an unseen vertex must extend the graph."""
    graph = IncrementalCore()
    graph.insert_edge(0, 1)
    graph.insert_edge(1, 5)
    assert len(graph.adjacency) == 6
    assert graph.core == graph.batch_core()


def test_adding_a_node_changes_nothing_else():
    graph = build_incrementally(4, [(0, 1), (1, 2), (2, 0)])
    before = list(graph.core)
    graph.add_node()
    assert graph.core[:4] == before
    assert graph.core[-1] == 0


def test_from_adjacency_seeds_correctly():
    """Seeding mid-stream must agree with the oracle before any insertion happens."""
    adjacency = [[1, 2], [0, 2], [0, 1], []]
    graph = IncrementalCore.from_adjacency(adjacency)
    assert graph.core == list(k_core_numbers(adjacency))
    assert graph.matches_batch()


def test_seeded_graph_stays_correct_under_further_insertions():
    source = nx.gnm_random_graph(30, 90, seed=11)
    adjacency = [sorted(source.neighbors(n)) for n in range(30)]
    graph = IncrementalCore.from_adjacency(adjacency)

    for u, v in [(0, 15), (3, 22), (7, 28), (11, 19)]:
        graph.insert_edge(u, v)
        assert graph.matches_batch()


def test_core_numbers_never_decrease_on_insertion():
    """Insertion is monotone: adding an edge cannot make a vertex less deeply embedded."""
    source = nx.gnm_random_graph(35, 120, seed=5)
    graph = IncrementalCore(adjacency=[[] for _ in range(35)], core=[0] * 35)

    previous = list(graph.core)
    for u, v in edges_of(source):
        graph.insert_edge(u, v)
        assert all(new >= old for new, old in zip(graph.core, previous))
        previous = list(graph.core)


def test_a_single_insertion_raises_core_by_at_most_one():
    """The theoretical bound the whole algorithm rests on."""
    source = nx.gnm_random_graph(35, 140, seed=9)
    graph = IncrementalCore(adjacency=[[] for _ in range(35)], core=[0] * 35)

    previous = list(graph.core)
    for u, v in edges_of(source):
        graph.insert_edge(u, v)
        assert all(new - old <= 1 for new, old in zip(graph.core, previous))
        previous = list(graph.core)
