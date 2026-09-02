"""Export every dashboard-facing number to a single committed JSON artifact.

WHY THIS EXISTS
---------------
Until now every number RingWatch reports existed only as stdout from `run.py`. The demo
needs those numbers, and the demo must not compute them -- a web request must never be able
to trigger a model evaluation, a bootstrap, or an LLM call.

So computation and presentation are split at a file boundary:

    LOCAL (has the 683 MB dataset)              DEPLOYED (has neither dataset nor cache)
    --------------------------------            ----------------------------------------
    python run.py --stage ablation
    python scripts/export_results.py  ------->   docs/results.json  ---->  FastAPI renders

`data/raw/` and `data/cache/` are gitignored, so the deployed instance genuinely cannot
recompute any of this even if the code tried to. The constraint enforces the architecture.

This script introduces NO new analysis. Every figure below comes from a function that
already existed and is already tested; this is serialisation, not computation.

    python scripts/export_results.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai.narrate import narrate_all  # noqa: E402
from core.calibration import calibration_report  # noqa: E402
from core.clusters import build_cluster_evidence  # noqa: E402
from core.data import CACHE_DIR, TARGET, load_merged  # noqa: E402
from core.evaluate import (  # noqa: E402
    MAX_ACCEPTABLE_INSULT_RATE,
    USD_TO_INR,
    GROSS_MARGIN_RATE,
    CHARGEBACK_FEE_INR,
    bootstrap_auc_pr_delta,
    evaluate,
)
from core.features import add_time_features  # noqa: E402
from core.graph import (  # noqa: E402
    MAX_GROUP_SIZE,
    UID_COL,
    build_features_for_split,
    build_graph,
    build_uid,
    connected_components,
    entity_frame,
    graph_features,
    graph_summary,
)
from core.ring_evidence import ring_concentration_test  # noqa: E402
from core.value_metrics import (  # noqa: E402
    value_concentration,
    value_detection_rate,
    value_weighted_average_precision,
)
from core.split import DEFAULT_SPLIT_QUANTILE, split_summary, temporal_split  # noqa: E402

OUTPUT = REPO_ROOT / "docs" / "results.json"

# Cache key -> display label. Order is the order the dashboard shows them in.
VARIANTS = [
    ("baseline", "tabular only"),
    ("components", "+ components"),
    ("kcore", "+ k-core"),
    ("graph_full", "+ full graph"),
]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def threshold_to_dict(report) -> dict:
    return {
        "threshold": report.threshold,
        "precision": report.precision,
        "recall": report.recall,
        "true_positives": report.true_positives,
        "false_positives": report.false_positives,
        "true_negatives": report.true_negatives,
        "false_negatives": report.false_negatives,
        "insult_rate": report.insult_rate,
        "fraud_caught_inr": report.fraud_caught_inr,
        "fraud_missed_inr": report.fraud_missed_inr,
        "insult_cost_inr": report.insult_cost_inr,
        "total_cost_inr": report.total_cost_inr,
    }


def main() -> int:
    print("Loading data and rebuilding the temporal split ...", flush=True)
    df = load_merged()
    df[UID_COL] = build_uid(df)
    train, test = temporal_split(df)

    missing = [k for k, _ in VARIANTS if not (CACHE_DIR / f"scores_{k}.npy").exists()]
    if missing:
        print(f"Missing cached scores for {missing}.")
        print("Run: python run.py --stage ablation")
        return 1

    scores = {key: np.load(CACHE_DIR / f"scores_{key}.npy") for key, _ in VARIANTS}
    y_true = test[TARGET].to_numpy().astype(int)
    amounts_usd = test["TransactionAmt"].to_numpy()

    # ---- split -----------------------------------------------------------
    summary = split_summary(train, test)

    # ---- ablation + operating points -------------------------------------
    print("Evaluating variants ...", flush=True)
    reports = {
        key: evaluate(label, y_true, scores[key], amounts_usd) for key, label in VARIANTS
    }

    print("Bootstrapping confidence intervals ...", flush=True)
    deltas = {}
    for key, label in VARIANTS:
        if key == "baseline":
            continue
        deltas[key] = bootstrap_auc_pr_delta(
            y_true, scores["baseline"], scores[key], name=label
        )

    ablation = []
    for key, label in VARIANTS:
        report = reports[key]
        delta = deltas.get(key)
        ablation.append(
            {
                "key": key,
                "label": label,
                "auc_pr": report.auc_pr,
                "auc_roc": report.auc_roc,
                "lift_over_random": report.auc_pr_lift_over_random,
                "delta": delta.delta if delta else None,
                "ci_low": delta.ci_low if delta else None,
                "ci_high": delta.ci_high if delta else None,
                "significant": delta.significant if delta else None,
                "verdict": delta.verdict() if delta else "baseline",
            }
        )

    # ---- graph + ring concentration --------------------------------------
    print("Building the entity graph and running the permutation test ...", flush=True)
    graph = build_graph(entity_frame(df))
    features = graph_features(graph)
    gsummary = graph_summary(graph, features)

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
    ring = ring_concentration_test(labels, entity_fraud, degrees > 0)

    # ---- value-weighted view ---------------------------------------------
    # Reuses the cached scores; adds the centrality variant when it exists.
    print("Computing the value-weighted comparison ...", flush=True)
    value_variants = list(VARIANTS)
    if (CACHE_DIR / "scores_centrality.npy").exists():
        scores["centrality"] = np.load(CACHE_DIR / "scores_centrality.npy")
        value_variants = value_variants + [("centrality", "+ centrality")]

    n_comparisons = (len(value_variants) - 1) * 2
    value_rows = []
    for key, label in value_variants:
        row = {
            "key": key,
            "label": label,
            "auc_pr": float(average_precision_score(y_true, scores[key])),
            "value_weighted_ap": value_weighted_average_precision(
                y_true, scores[key], amounts_usd
            ),
        }
        if key != "baseline":
            for weight_name, weight in (("count", None), ("value", amounts_usd)):
                raw = bootstrap_auc_pr_delta(
                    y_true, scores["baseline"], scores[key], name=label,
                    sample_weight=weight,
                )
                corrected = bootstrap_auc_pr_delta(
                    y_true, scores["baseline"], scores[key], name=label,
                    sample_weight=weight, n_comparisons=n_comparisons,
                )
                row[f"{weight_name}_delta"] = raw.delta
                row[f"{weight_name}_ci"] = [raw.ci_low, raw.ci_high]
                row[f"{weight_name}_ci_corrected"] = [corrected.ci_low, corrected.ci_high]
                row[f"{weight_name}_significant"] = raw.significant
                row[f"{weight_name}_survives_correction"] = corrected.significant
        value_rows.append(row)

    # Reuse the full-graph features already computed above rather than rebuilding.
    linked_mask = (
        test[[UID_COL]]
        .merge(features[[UID_COL, "g_is_linked"]], on=UID_COL, how="left")["g_is_linked"]
        .fillna(0)
        .to_numpy()
        == 1
    )
    concentration = value_concentration(
        y_true, amounts_usd, linked_mask, "graph-linked rows"
    )

    # ---- calibration -----------------------------------------------------
    print("Computing calibration ...", flush=True)
    calibration = []
    for key in ("baseline", "graph_full"):
        report = calibration_report(dict(VARIANTS)[key], y_true, scores[key])
        calibration.append(
            {
                "key": key,
                "label": report.name,
                "brier": report.brier,
                "ece": report.ece,
                "max_bin_error": report.max_bin_error,
                "n_bins": report.n_bins,
                "strategy": report.strategy,
                "curve": [
                    {"predicted": float(p), "observed": float(o), "rows": int(c)}
                    for p, o, c in zip(
                        report.prob_pred, report.prob_true, report.bin_counts
                    )
                ],
            }
        )

    # ---- flagged clusters + narratives -----------------------------------
    print("Selecting flagged clusters and fetching narratives ...", flush=True)
    _, test_with_graph, _ = build_features_for_split(train, test)
    test_with_graph = add_time_features(test_with_graph)
    threshold = reports["graph_full"].constrained_operating_point.threshold
    evidence_items = build_cluster_evidence(test_with_graph, scores["graph_full"], threshold)
    narratives = narrate_all(evidence_items)

    clusters = []
    for evidence, narrative in zip(evidence_items, narratives):
        clusters.append(
            {
                # Everything under "evidence" was computed by core/. Everything under
                # "narrative" was written by the language model. The dashboard renders
                # that distinction visually; keeping it in the data model is what makes
                # that possible without the template having to guess.
                "evidence": {
                    "cluster_id": evidence.cluster_id,
                    "entity_count": evidence.entity_count,
                    "transaction_count": evidence.transaction_count,
                    "flagged_transaction_count": evidence.flagged_transaction_count,
                    "component_size": evidence.component_size,
                    "core_number": evidence.core_number,
                    "max_degree": evidence.max_degree,
                    "shared_attributes": list(evidence.shared_attributes),
                    "distinct_cards": evidence.distinct_cards,
                    "distinct_addresses": evidence.distinct_addresses,
                    "distinct_email_domains": evidence.distinct_email_domains,
                    "span_days": evidence.span_days,
                    "total_amount_inr": evidence.total_amount_inr,
                    "max_risk_score": evidence.max_risk_score,
                    "mean_risk_score": evidence.mean_risk_score,
                },
                "narrative": {
                    "status": narrative.status,
                    "probable_cause": narrative.probable_cause,
                    "confidence": narrative.confidence,
                    "human_summary": narrative.human_summary,
                    "suggested_action": narrative.suggested_action,
                    "rejection_reason": narrative.rejection_reason,
                },
            }
        )

    validated = sum(1 for c in clusters if c["narrative"]["status"] == "OK")

    # ---- assemble --------------------------------------------------------
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "note": (
                "Produced by scripts/export_results.py from cached model scores. The web "
                "app renders this file and computes nothing."
            ),
        },
        "dataset": {
            "name": "IEEE-CIS Fraud Detection",
            "rows": int(len(df)),
            "fraud_rate": float(df[TARGET].mean()),
            "split_quantile": DEFAULT_SPLIT_QUANTILE,
            **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in summary.items()},
        },
        "ablation": ablation,
        "ring_evidence": {
            "linked_entities": ring.linked_entities,
            "components": ring.components,
            "entity_fraud_rate": ring.entity_fraud_rate,
            "all_fraud_components": ring.all_fraud_components,
            "mixed_components": ring.mixed_components,
            "clean_components": ring.clean_components,
            "null_mean": ring.null_mean,
            "null_std": ring.null_std,
            "z_score": ring.z_score,
        },
        "graph": {
            **{k: (float(v) if isinstance(v, float) else int(v)) for k, v in gsummary.items()},
            "hub_cap": MAX_GROUP_SIZE,
            "test_rows_linked": int((test_with_graph["g_is_linked"].fillna(0) == 1).sum()),
            "test_rows_linked_pct": float(
                (test_with_graph["g_is_linked"].fillna(0) == 1).mean() * 100
            ),
        },
        "operating_points": {
            "cost_minimising": threshold_to_dict(reports["baseline"].operating_point),
            "insult_constrained": threshold_to_dict(
                reports["baseline"].constrained_operating_point
            ),
            "insult_cap": MAX_ACCEPTABLE_INSULT_RATE,
        },
        "assumptions": {
            "usd_to_inr": USD_TO_INR,
            "gross_margin_rate": GROSS_MARGIN_RATE,
            "chargeback_fee_inr": CHARGEBACK_FEE_INR,
        },
        "value_weighted": {
            "n_comparisons": n_comparisons,
            "variants": value_rows,
            "concentration": {
                "count_share": concentration.count_share,
                "value_share": concentration.value_share,
                "enrichment": concentration.enrichment,
                "mean_amount_in": concentration.mean_amount_in,
                "mean_amount_out": concentration.mean_amount_out,
                "n_fraud": concentration.n_fraud,
            },
        },
        "calibration": calibration,
        "clusters": clusters,
        "narratives_validated": validated,
        "narratives_total": len(clusters),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2))
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"\nwrote {OUTPUT} ({size_kb:.1f} KB)")
    print(f"  ablation variants : {len(ablation)}")
    print(f"  clusters          : {len(clusters)} ({validated} narratives validated)")
    print(f"  ring z-score      : {ring.z_score:+.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
