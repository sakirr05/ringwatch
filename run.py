"""RingWatch pipeline entrypoint.

    python run.py --stage data       # build the Parquet cache, print dataset stats
    python run.py --stage baseline   # train + evaluate the tabular-only model
    python run.py --stage graph      # build the entity graph, print its statistics
    python run.py --stage ablation   # baseline vs graph-augmented, the honest comparison
    python run.py --stage narrate    # LLM narratives for flagged clusters
    python run.py --stage all

Every stage is deterministic and every expensive artifact is cached under data/cache/, so
re-running is cheap and reproduces byte-identical numbers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.data import CACHE_DIR, TARGET, dataset_summary, load_merged
from core.evaluate import evaluate
from core.features import add_time_features, feature_columns
from core.graph import (
    GRAPH_FEATURE_COLUMNS,
    UID_COL,
    build_features_for_split,
    build_graph,
    build_uid,
    connected_components,
    entity_frame,
    graph_features,
    graph_summary,
)
from core.clusters import build_cluster_evidence
from core.model import train_model
from core.ring_evidence import ring_concentration_test
from core.split import split_summary, temporal_split

from ai.narrate import narrate_all

STAGES = ("data", "baseline", "graph", "ablation", "narrate", "all")


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def stage_data() -> pd.DataFrame:
    banner("STAGE: data")
    started = time.time()
    df = load_merged()
    print(f"loaded in {time.time() - started:.1f}s")

    for key, value in dataset_summary(df).items():
        print(f"  {key:12s} {value}")

    train, test = temporal_split(df)
    print("\ntemporal split (80th percentile of TransactionDT):")
    for key, value in split_summary(train, test).items():
        print(f"  {key:20s} {value}")

    leak_free = train.TransactionDT.max() < test.TransactionDT.min()
    print(f"  {'leakage check':20s} max(train) < min(test) -> {leak_free}")
    if not leak_free:
        raise AssertionError("temporal leakage detected")
    return df


def _fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    label: str,
    cache_key: str,
) -> np.ndarray:
    """Train (or load a cached model) and return held-out scores."""
    model_path = CACHE_DIR / f"model_{cache_key}.txt"
    score_path = CACHE_DIR / f"scores_{cache_key}.npy"

    if score_path.exists() and model_path.exists():
        print(f"  [{label}] using cached scores ({score_path.name})")
        return np.load(score_path)

    print(f"  [{label}] training on {len(cols)} features ...", flush=True)
    started = time.time()
    model = train_model(train, cols)
    print(
        f"  [{label}] {time.time() - started:.0f}s, "
        f"best_iter={model.best_iteration}, validation AUC-PR={model.validation_ap:.4f}"
    )

    scores = model.predict(test)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(model_path), num_iteration=model.best_iteration)
    np.save(score_path, scores)
    return scores


def stage_baseline(df: pd.DataFrame | None = None) -> None:
    banner("STAGE: baseline (tabular only)")
    df = df if df is not None else load_merged()
    train, test = temporal_split(df)
    train, test = add_time_features(train), add_time_features(test)
    cols = feature_columns(train)

    scores = _fit_and_score(train, test, cols, "baseline", "baseline")
    report = evaluate(
        "BASELINE (tabular only)",
        test[TARGET].values,
        scores,
        test["TransactionAmt"].values,
    )
    print()
    for line in report.summary_lines():
        print(line)


def stage_graph(df: pd.DataFrame | None = None) -> pd.DataFrame:
    banner("STAGE: graph")
    df = df if df is not None else load_merged()
    df = df.copy()
    df[UID_COL] = build_uid(df)

    started = time.time()
    graph = build_graph(entity_frame(df))
    features = graph_features(graph)
    print(f"built entity graph in {time.time() - started:.1f}s")
    for key, value in graph_summary(graph, features).items():
        print(f"  {key:28s} {value}")

    print("\nring concentration test (is fraud bunched into components?):")
    labels = connected_components(graph.adjacency)[: graph.n_uids]
    entity_fraud = (
        df.dropna(subset=[UID_COL])
        .groupby(UID_COL, observed=True)[TARGET]
        .max()
        .reindex(graph.uids)
        .fillna(0)
        .to_numpy()
    )
    degrees = np.array(
        [len(graph.adjacency[i]) for i in range(graph.n_uids)], dtype=np.int64
    )
    evidence = ring_concentration_test(labels, entity_fraud, degrees > 0)
    for line in evidence.summary_lines():
        print(line)

    if evidence.z_score > 3:
        print(
            f"\n  -> fraud concentrates in components {evidence.z_score:+.1f} sigma "
            "beyond chance."
        )
        print(
            "     This is evidence of ring STRUCTURE, not a validated ring count -- "
            "IEEE-CIS has no ring labels."
        )
    return df


def _honest_negative_case(
    test: pd.DataFrame,
    baseline_scores: np.ndarray,
    augmented_scores: np.ndarray,
) -> None:
    """Find and print a case where the graph layer made things WORSE.

    Required output, not optional. A submission that reports only the favourable half of
    an ablation is not reporting an ablation.
    """
    frame = pd.DataFrame(
        {
            "is_fraud": test[TARGET].to_numpy(),
            "baseline": baseline_scores,
            "augmented": augmented_scores,
            "component_size": test["g_component_size"].to_numpy(),
            "core": test["g_core_number"].to_numpy(),
            "amount": test["TransactionAmt"].to_numpy(),
        }
    )
    frame["delta"] = frame["augmented"] - frame["baseline"]

    legit = frame[(frame.is_fraud == 0) & (frame.component_size.notna())]
    hurt = legit.nlargest(5, "delta")

    print("\n  Legitimate transactions the graph layer pushed MOST toward being declined:")
    print(
        f"    {'component':>10} {'core':>5} {'amount':>10} "
        f"{'baseline':>9} {'augmented':>10} {'delta':>8}"
    )
    for _, row in hurt.iterrows():
        print(
            f"    {row.component_size:10.0f} {row.core:5.0f} {row.amount:10.2f} "
            f"{row.baseline:9.4f} {row.augmented:10.4f} {row.delta:+8.4f}"
        )

    fraud = frame[(frame.is_fraud == 1) & (frame.component_size.notna())]
    missed = fraud.nsmallest(3, "delta")
    print("\n  Fraudulent transactions the graph layer pushed toward being ACCEPTED:")
    for _, row in missed.iterrows():
        print(
            f"    {row.component_size:10.0f} {row.core:5.0f} {row.amount:10.2f} "
            f"{row.baseline:9.4f} {row.augmented:10.4f} {row.delta:+8.4f}"
        )


def stage_ablation(df: pd.DataFrame | None = None) -> None:
    banner("STAGE: ablation (tabular vs graph-augmented)")
    df = df if df is not None else load_merged()
    if UID_COL not in df.columns:
        df = df.copy()
        df[UID_COL] = build_uid(df)

    train, test = temporal_split(df)
    train, test, graph_stats = build_features_for_split(train, test)
    print("graph built separately per split to keep the future out of training:")
    for name, stats in graph_stats.items():
        print(
            f"  {name:12s} linked={stats['linked_uids']:,} "
            f"largest_component={stats['largest_component_entities']}"
        )

    train, test = add_time_features(train), add_time_features(test)

    base_cols = feature_columns(train, extra_excluded=set(GRAPH_FEATURE_COLUMNS) | {UID_COL})
    full_cols = feature_columns(train, extra_excluded={UID_COL})

    variants = {
        "tabular only": (base_cols, "baseline"),
        "+ components": (
            base_cols + ["g_component_size", "g_component_size_total", "g_is_linked"],
            "components",
        ),
        "+ k-core": (base_cols + ["g_core_number", "g_degree"], "kcore"),
        "+ full graph": (full_cols, "graph_full"),
    }

    reports = {}
    scores_by_name = {}
    for name, (cols, cache_key) in variants.items():
        scores = _fit_and_score(train, test, cols, name, cache_key)
        scores_by_name[name] = scores
        reports[name] = evaluate(
            name, test[TARGET].values, scores, test["TransactionAmt"].values
        )

    print(f"\n{'variant':<16} {'features':>9} {'AUC-PR':>9} {'vs base':>10} {'AUC-ROC':>9}")
    base_ap = reports["tabular only"].auc_pr
    for name, report in reports.items():
        delta = report.auc_pr - base_ap
        marker = "" if name == "tabular only" else f"{delta:+.4f}"
        print(
            f"{name:<16} {len(variants[name][0]):>9} {report.auc_pr:>9.4f} "
            f"{marker:>10} {report.auc_roc:>9.4f}"
        )

    print()
    for report in reports.values():
        for line in report.summary_lines():
            print(line)

    banner("HONEST NEGATIVE CASE: where the graph layer HURTS")
    _honest_negative_case(
        test, scores_by_name["tabular only"], scores_by_name["+ full graph"]
    )


def stage_narrate(df: pd.DataFrame | None = None) -> None:
    banner("STAGE: narrate (LLM writes about already-flagged clusters)")
    df = df if df is not None else load_merged()
    if UID_COL not in df.columns:
        df = df.copy()
        df[UID_COL] = build_uid(df)

    train, test = temporal_split(df)
    _, test, _ = build_features_for_split(train, test)
    test = add_time_features(test)

    score_path = CACHE_DIR / "scores_graph_full.npy"
    if not score_path.exists():
        print("No graph-augmented scores cached. Run: python run.py --stage ablation")
        return
    scores = np.load(score_path)

    report = evaluate(
        "graph-augmented", test[TARGET].values, scores, test["TransactionAmt"].values
    )
    threshold = report.constrained_operating_point.threshold
    print(f"using the insult-constrained threshold: {threshold:.4f}")

    evidence_items = build_cluster_evidence(test, scores, threshold)
    print(f"deterministic engine flagged {len(evidence_items)} multi-entity clusters\n")

    if not evidence_items:
        print("no clusters met the criteria")
        return

    narratives = narrate_all(evidence_items)

    unavailable_count = 0
    for evidence, narrative in zip(evidence_items, narratives):
        print(f"--- cluster {evidence.cluster_id} ---")
        print(
            f"  entities={evidence.entity_count} transactions={evidence.transaction_count} "
            f"flagged={evidence.flagged_transaction_count} core={evidence.core_number} "
            f"span={evidence.span_days}d"
        )
        print(f"  shared: {', '.join(evidence.shared_attributes)}")
        print(f"  max_risk={evidence.max_risk_score:.4f} (computed by core/, not the LLM)")
        if narrative.status != "OK":
            unavailable_count += 1
            print(f"  NARRATIVE_UNAVAILABLE -- {narrative.rejection_reason}")
        else:
            print(f"  cause: {narrative.probable_cause}  confidence: {narrative.confidence}")
            print(f"  summary: {narrative.human_summary}")
            print(f"  action:  {narrative.suggested_action}")
        print()

    print(
        f"{len(narratives) - unavailable_count}/{len(narratives)} narratives validated; "
        f"{unavailable_count} fell back to NARRATIVE_UNAVAILABLE."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="RingWatch pipeline")
    parser.add_argument("--stage", choices=STAGES, default="all")
    args = parser.parse_args()

    if args.stage == "data":
        stage_data()
    elif args.stage == "baseline":
        stage_baseline()
    elif args.stage == "graph":
        stage_graph()
    elif args.stage == "ablation":
        stage_ablation()
    elif args.stage == "narrate":
        stage_narrate()
    else:
        df = stage_data()
        stage_baseline(df)
        stage_graph(df)
        stage_ablation(df)
        stage_narrate(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
