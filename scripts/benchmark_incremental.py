"""Benchmark incremental k-core maintenance against full batch rebuild.

Reports whichever way the result falls. The prediction was recorded in
PLAN_INCREMENTAL.md before this script was written, so the comparison between predicted and
measured is itself part of the output.

    python scripts/benchmark_incremental.py
"""

from __future__ import annotations

import random
import sys
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.graph import k_core_numbers  # noqa: E402
from core.graph_incremental import IncrementalCore  # noqa: E402

SIZES = [
    ("small", 1_000, 3_000),
    ("medium", 5_000, 15_000),
    ("large", 20_000, 60_000),
    ("sparse (entity-graph-like)", 50_000, 15_000),
]

# How many edges to insert incrementally when measuring per-insertion latency.
PROBE_EDGES = 500


def random_edges(n_nodes: int, n_edges: int, seed: int = 0) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    edges = set()
    while len(edges) < n_edges:
        u = rng.randrange(n_nodes)
        v = rng.randrange(n_nodes)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def adjacency_from(n_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    return adjacency


def bench_case(label: str, n_nodes: int, n_edges: int) -> dict:
    edges = random_edges(n_nodes, n_edges)
    seed_edges, probe = edges[:-PROBE_EDGES], edges[-PROBE_EDGES:]

    # --- full batch rebuild on the complete graph ---------------------
    full_adjacency = adjacency_from(n_nodes, edges)
    started = time.perf_counter()
    k_core_numbers(full_adjacency)
    rebuild_s = time.perf_counter() - started

    # --- incremental: seed, then insert the probe edges one at a time --
    graph = IncrementalCore.from_adjacency(adjacency_from(n_nodes, seed_edges))

    started = time.perf_counter()
    for u, v in probe:
        graph.insert_edge(u, v)
    incremental_s = time.perf_counter() - started
    per_insert_us = 1e6 * incremental_s / len(probe)

    # Correctness is not optional even in the benchmark.
    correct = graph.core == list(k_core_numbers(graph.adjacency))

    # --- crossover -----------------------------------------------------
    # Incremental is cheaper while  n_updates * per_insert < one rebuild.
    crossover = rebuild_s / (incremental_s / len(probe)) if incremental_s else float("inf")

    # --- memory overhead of the maintained state -----------------------
    tracemalloc.start()
    IncrementalCore.from_adjacency(adjacency_from(n_nodes, seed_edges))
    _, peak_incremental = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    k_core_numbers(adjacency_from(n_nodes, seed_edges))
    _, peak_batch = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "label": label,
        "nodes": n_nodes,
        "edges": n_edges,
        "rebuild_ms": rebuild_s * 1e3,
        "per_insert_us": per_insert_us,
        "crossover_edges": crossover,
        "crossover_pct": 100.0 * crossover / n_edges,
        "correct": correct,
        "peak_incremental_mb": peak_incremental / 1e6,
        "peak_batch_mb": peak_batch / 1e6,
        "candidates_per_insert": graph.candidates_examined / max(graph.insertions, 1),
    }


def bench_bulk_replay(n_nodes: int, n_edges: int) -> dict:
    """Insert every edge one at a time, versus one rebuild at the end."""
    edges = random_edges(n_nodes, n_edges, seed=7)

    graph = IncrementalCore(adjacency=[[] for _ in range(n_nodes)], core=[0] * n_nodes)
    started = time.perf_counter()
    for u, v in edges:
        graph.insert_edge(u, v)
    incremental_s = time.perf_counter() - started

    adjacency = adjacency_from(n_nodes, edges)
    started = time.perf_counter()
    k_core_numbers(adjacency)
    rebuild_s = time.perf_counter() - started

    return {
        "incremental_s": incremental_s,
        "rebuild_s": rebuild_s,
        "ratio": incremental_s / rebuild_s if rebuild_s else float("inf"),
        "correct": graph.core == list(k_core_numbers(graph.adjacency)),
    }


def bench_real_graph() -> dict | None:
    """The actual IEEE-CIS entity graph, if the dataset cache is present."""
    from core.data import MERGED_PARQUET

    if not MERGED_PARQUET.exists():
        return None

    from core.data import load_merged
    from core.graph import UID_COL, build_graph, build_uid, entity_frame

    df = load_merged()
    df[UID_COL] = build_uid(df)
    batch = build_graph(entity_frame(df))

    edges = [
        (node, neighbour)
        for node, neighbours in enumerate(batch.adjacency)
        for neighbour in neighbours
        if node < neighbour
    ]

    started = time.perf_counter()
    k_core_numbers(batch.adjacency)
    rebuild_s = time.perf_counter() - started

    graph = IncrementalCore(
        adjacency=[[] for _ in range(len(batch.adjacency))],
        core=[0] * len(batch.adjacency),
    )
    started = time.perf_counter()
    for u, v in edges:
        graph.insert_edge(u, v)
    incremental_s = time.perf_counter() - started

    return {
        "nodes": len(batch.adjacency),
        "edges": len(edges),
        "rebuild_ms": rebuild_s * 1e3,
        "incremental_s": incremental_s,
        "per_insert_us": 1e6 * incremental_s / max(len(edges), 1),
        "ratio": incremental_s / rebuild_s if rebuild_s else float("inf"),
        "crossover_edges": rebuild_s / (incremental_s / max(len(edges), 1)),
        "correct": graph.core == list(k_core_numbers(batch.adjacency)),
        "hub_values_suppressed": batch.dropped_hub_values,
    }


def main() -> int:
    print("=" * 78)
    print("INCREMENTAL K-CORE vs FULL REBUILD")
    print("=" * 78)

    print("\n--- per-insertion latency vs one full rebuild ---")
    print(
        f"{'graph':<28} {'nodes':>7} {'edges':>7} {'rebuild':>10} "
        f"{'per-insert':>11} {'crossover':>11} {'ok':>4}"
    )
    results = []
    for label, nodes, edges in SIZES:
        row = bench_case(label, nodes, edges)
        results.append(row)
        print(
            f"{row['label']:<28} {row['nodes']:>7,} {row['edges']:>7,} "
            f"{row['rebuild_ms']:>8.1f}ms {row['per_insert_us']:>9.1f}us "
            f"{row['crossover_edges']:>8.0f}ed {str(row['correct']):>5}"
        )

    print("\n  crossover = how many single-edge insertions cost the same as one rebuild")
    for row in results:
        print(
            f"    {row['label']:<28} {row['crossover_edges']:>8.0f} edges "
            f"= {row['crossover_pct']:>5.1f}% of the graph  "
            f"({row['candidates_per_insert']:.1f} candidates examined per insert)"
        )

    print("\n--- memory: peak during maintenance vs during a rebuild ---")
    for row in results:
        overhead = row["peak_incremental_mb"] - row["peak_batch_mb"]
        print(
            f"    {row['label']:<28} incremental {row['peak_incremental_mb']:>6.1f} MB  "
            f"batch {row['peak_batch_mb']:>6.1f} MB  overhead {overhead:>+6.1f} MB"
        )

    print("\n--- bulk replay: every edge inserted one at a time vs one rebuild ---")
    bulk = bench_bulk_replay(5_000, 15_000)
    print(
        f"    5,000 nodes / 15,000 edges: incremental {bulk['incremental_s']:.3f}s  "
        f"rebuild {bulk['rebuild_s']:.3f}s  ->  incremental is "
        f"{bulk['ratio']:.1f}x SLOWER" if bulk['ratio'] > 1 else
        f"rebuild {bulk['rebuild_s']:.3f}s  ->  incremental is "
        f"{1/bulk['ratio']:.1f}x faster"
        f"   (correct: {bulk['correct']})"
    )

    print("\n--- the real IEEE-CIS entity graph ---")
    real = bench_real_graph()
    if real is None:
        print("    dataset cache absent; skipped")
    else:
        print(f"    {real['nodes']:,} nodes, {real['edges']:,} edges after hub suppression")
        print(f"    full rebuild            : {real['rebuild_ms']:.1f} ms")
        print(f"    incremental full replay : {real['incremental_s']:.3f} s "
              f"({real['per_insert_us']:.1f} us per edge)")
        if real["ratio"] > 1:
            verdict = f"{real['ratio']:.1f}x SLOWER"
        else:
            verdict = f"{1 / real['ratio']:.1f}x FASTER"
        print(f"    ratio                   : incremental is {verdict} for a full replay")
        print(f"    crossover               : {real['crossover_edges']:.0f} edges "
              f"({100 * real['crossover_edges'] / real['edges']:.2f}% of the graph)")
        print(f"    core numbers match batch: {real['correct']}")
        print(f"    hub values suppressed   : {real['hub_values_suppressed']:,} "
              "(each is a cap-crossing a streaming build would have to handle as a deletion)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
