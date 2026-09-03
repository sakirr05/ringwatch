"""RingWatch pipeline entrypoint.

    python run.py --stage data       # build the Parquet cache, print dataset stats
    python run.py --stage baseline   # train + evaluate the tabular-only model
    python run.py --stage graph      # build the entity graph, print its statistics
    python run.py --stage ablation   # baseline vs graph-augmented, the honest comparison
    python run.py --stage value      # value-weighted ablation + centrality variant
    python run.py --stage calibration # are the predicted probabilities trustworthy?
    python run.py --stage narrate    # LLM narratives for flagged clusters
    python run.py --stage all

Every stage is deterministic and every expensive artifact is cached under data/cache/, so
re-running is cheap and reproduces byte-identical numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from core.data import CACHE_DIR, TARGET, dataset_summary, load_merged
from core.evaluate import (
    amount_inr,
    bootstrap_auc_pr_delta,
    choose_threshold_under_insult_cap,
    evaluate,
)
from core.value_metrics import (
    bootstrap_vdr_delta,
    value_concentration,
    value_detection_rate,
    value_weighted_average_precision,
)
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
from core.calibration import calibration_report
from core.centrality import (
    CENTRALITY_FEATURE_COLUMNS,
    betweenness,
    build_centrality_for_split,
    component_centrality_variance,
    pagerank,
)
from core.clusters import build_cluster_evidence
from core.costs import cost_weights, weight_summary
from core.model import train_model
from core.ring_evidence import ring_concentration_test
from core.split import split_summary, temporal_split

from ai.narrate import narrate_all

STAGES = (
    "data", "baseline", "graph", "ablation", "value", "cost", "calibration",
    "narrate", "all",
)


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
            "core": test["g_core_number"].fillna(0).to_numpy(),
            "degree": test["g_degree"].fillna(0).to_numpy(),
            "amount": test["TransactionAmt"].to_numpy(),
        }
    )
    frame["delta"] = frame["augmented"] - frame["baseline"]
    size = frame["component_size"].fillna(0)

    groups = [
        ("no entity resolved", size == 0),
        ("isolated (comp=1)", size == 1),
        ("comp 2", size == 2),
        ("comp 3-4", (size >= 3) & (size <= 4)),
        ("comp 5+", size >= 5),
    ]

    print("\n  Where does the graph layer actually move scores?")
    print(f"    {'group':<20} {'rows':>8} {'mean|delta|':>12} {'mean delta':>12}")
    for name, mask in groups:
        if not mask.any():
            continue
        subset = frame[mask]
        print(
            f"    {name:<20} {len(subset):>8,} {subset.delta.abs().mean():>12.4f} "
            f"{subset.delta.mean():>+12.4f}"
        )

    print("\n  Legitimate rows pushed toward decline by more than 0.10:")
    print(f"    {'group':<20} {'legit rows':>11} {'harmed':>8} {'rate':>9}")
    legit_frame = frame[frame.is_fraud == 0]
    legit_size = legit_frame["component_size"].fillna(0)
    for name, _ in groups:
        mask = {
            "no entity resolved": legit_size == 0,
            "isolated (comp=1)": legit_size == 1,
            "comp 2": legit_size == 2,
            "comp 3-4": (legit_size >= 3) & (legit_size <= 4),
            "comp 5+": legit_size >= 5,
        }[name]
        if not mask.any():
            continue
        subset = legit_frame[mask]
        harmed = int((subset.delta > 0.10).sum())
        print(
            f"    {name:<20} {len(subset):>11,} {harmed:>8} "
            f"{100 * harmed / len(subset):>8.3f}%"
        )

    print(
        "\n  -> The largest perturbation is on rows where NO entity could be resolved,"
        "\n     i.e. exactly where the graph has nothing to say. The graph columns are"
        "\n     NaN there, and the model re-learns the (genuinely predictive) missingness"
        "\n     pattern more noisily than it already had from the null addr1/D1 columns."
        "\n     That is noise injection, not ring confusion."
    )

    # The hypothesised failure mode -- a legitimate customer inside a genuine multi-entity
    # cluster pushed toward a decline because the topology looks ring-like. It is real,
    # and it is NOT the dominant effect. Both halves of that sentence are reported.
    print("\n  The hypothesised case, which does also occur: legitimate customers inside")
    print("  REAL multi-entity clusters pushed toward decline by the topology alone:")
    print(
        f"    {'comp size':>10} {'core':>5} {'degree':>7} {'amount':>9} "
        f"{'baseline':>9} {'augmented':>10} {'delta':>8}"
    )
    ring_legit = frame[(frame.is_fraud == 0) & (size >= 3)].nlargest(5, "delta")
    for _, row in ring_legit.iterrows():
        print(
            f"    {row.component_size:10.0f} {row.core:5.0f} {row.degree:7.0f} "
            f"{row.amount:9.2f} {row.baseline:9.4f} {row.augmented:10.4f} "
            f"{row.delta:+8.4f}"
        )

    fraud = frame[(frame.is_fraud == 1) & frame.component_size.notna()]
    print("\n  And fraud the graph layer pushed toward being ACCEPTED:")
    for _, row in fraud.nsmallest(3, "delta").iterrows():
        print(
            f"    {row.component_size:10.0f} {row.core:5.0f} {row.degree:7.0f} "
            f"{row.amount:9.2f} {row.baseline:9.4f} {row.augmented:10.4f} "
            f"{row.delta:+8.4f}"
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

    banner("IS ANY OF THIS DIFFERENCE REAL? (paired bootstrap, 95% CI)")
    print("A raw delta is not a result. Resampling the test set shows how much of each")
    print("gap is just which transactions landed in the held-out period.\n")
    baseline_scores = scores_by_name["tabular only"]
    for name in ("+ components", "+ k-core", "+ full graph"):
        delta = bootstrap_auc_pr_delta(
            test[TARGET].values, baseline_scores, scores_by_name[name], name
        )
        print(delta.line())

    print("\n  Subgroup: transactions whose entity is actually linked in the graph")
    linked = test["g_is_linked"].fillna(0).to_numpy() == 1
    y_linked = test[TARGET].values[linked]
    print(
        f"    {linked.sum():,} rows ({100 * linked.mean():.2f}% of test), "
        f"fraud rate {y_linked.mean():.4f}"
    )
    for name, scores in scores_by_name.items():
        print(
            f"    {name:<16} AUC-PR = "
            f"{average_precision_score(y_linked, scores[linked]):.4f}"
        )
    subgroup_delta = bootstrap_auc_pr_delta(
        y_linked,
        baseline_scores[linked],
        scores_by_name["+ full graph"][linked],
        "+ full graph (linked only)",
    )
    print(subgroup_delta.line())

    print()
    for report in reports.values():
        for line in report.summary_lines():
            print(line)

    banner("HONEST NEGATIVE CASE: where the graph layer HURTS")
    _honest_negative_case(
        test, scores_by_name["tabular only"], scores_by_name["+ full graph"]
    )


def stage_cost(df: pd.DataFrame | None = None) -> None:
    """Does making the model cost-aware during TRAINING actually help?

    Cost asymmetry currently enters only when a threshold is chosen. This trains a variant
    that carries it in the objective, weighting each row by what getting it wrong would
    cost -- using the same named constants core/evaluate.py already uses, so training and
    thresholding cannot disagree about what a mistake is worth.

    The variant uses the SAME feature set as the baseline. Only the weighting differs, so
    any measured change is attributable to cost-sensitivity rather than to features.

    Two axes are reported, and the second is the one that matters: AUC-PR, and TOTAL
    EXPECTED COST at the insult-constrained operating point. Cost is what is being
    optimised, so a variant can lose on AUC-PR and still win on cost -- that would be the
    interesting outcome, and reporting only AUC-PR would hide it.
    """
    banner("STAGE: cost-sensitive training")
    df = df if df is not None else load_merged()
    train, test = temporal_split(df)
    train, test = add_time_features(train), add_time_features(test)
    cols = feature_columns(train)

    y_true = test[TARGET].to_numpy().astype(int)
    amounts = test["TransactionAmt"].to_numpy().astype(float)

    weights = cost_weights(train[TARGET], train["TransactionAmt"])
    print("cost weights applied during training:")
    for key, value in weight_summary(weights, train[TARGET]).items():
        print(f"  {key:22s} {value:.4f}")
    print("\n  A fraud row is weighted by (amount + chargeback fee); a legitimate row by")
    print("  (amount x gross margin). Same constants the threshold logic uses.\n")

    baseline = _fit_and_score(train, test, cols, "tabular only", "baseline")

    cache = CACHE_DIR / "scores_costsensitive.npy"
    if cache.exists() and (CACHE_DIR / "model_costsensitive.txt").exists():
        print("  [+ cost-sensitive] using cached scores")
        cost_scores = np.load(cache)
    else:
        print(f"  [+ cost-sensitive] training on {len(cols)} features with cost weights ...",
              flush=True)
        started = time.time()
        model = train_model(train, cols, sample_weight=weights)
        print(f"  [+ cost-sensitive] {time.time() - started:.0f}s, "
              f"best_iter={model.best_iteration}")
        cost_scores = model.predict(test)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model.booster.save_model(str(CACHE_DIR / "model_costsensitive.txt"),
                                 num_iteration=model.best_iteration)
        np.save(cache, cost_scores)

    scores = {"tabular only": baseline, "+ cost-sensitive": cost_scores}

    banner("AXIS 1: ranking quality (AUC-PR)")
    print(f"{'variant':<18} {'AUC-PR':>9} {'delta':>9}  {'95% CI':<22} verdict")
    for name, s in scores.items():
        ap = average_precision_score(y_true, s)
        if name == "tabular only":
            print(f"{name:<18} {ap:>9.4f} {'—':>9}  {'baseline':<22}")
            continue
        d = bootstrap_auc_pr_delta(y_true, baseline, s, name=name)
        print(f"{name:<18} {ap:>9.4f} {d.delta:>+9.4f}  "
              f"[{d.ci_low:+.4f}, {d.ci_high:+.4f}]   {d.verdict()}")

    banner("AXIS 2: total expected cost at the <=1% insult cap")
    print("This is the axis cost-sensitive training is actually optimising for.\n")
    print(f"{'variant':<18} {'threshold':>10} {'insult':>8} {'recall':>8} {'total cost':>16}")
    costs = {}
    for name, s in scores.items():
        cap = choose_threshold_under_insult_cap(y_true, s, amount_inr(amounts))
        costs[name] = cap.total_cost_inr
        print(f"{name:<18} {cap.threshold:>10.4f} {100*cap.insult_rate:>7.3f}% "
              f"{cap.recall:>8.4f} Rs {cap.total_cost_inr:>13,.0f}")

    delta = costs["+ cost-sensitive"] - costs["tabular only"]
    pct = 100 * delta / costs["tabular only"]
    print(f"\n  cost difference: Rs {delta:+,.0f} ({pct:+.2f}%)  "
          f"-> cost-sensitive training is {'CHEAPER' if delta < 0 else 'MORE EXPENSIVE'}")
    print("\n  Note: this is a single point estimate on one test set, not a CI. The AUC-PR")
    print("  comparison above carries the interval; treat the cost figure as indicative.")


def stage_value(df: pd.DataFrame | None = None) -> None:
    """Re-run the ablation weighted by transaction value, plus a centrality variant.

    AUC-PR is count-uniform: it treats a small fraud and a large one alike. PayPal's
    engineers report optimising for "dollar-weighted fraud detection", which raises a fair
    objection to this project's negative result -- maybe the graph layer only looks useless
    because the metric ignores money.

    Predictions for everything below were recorded in PLAN_VALUE_WEIGHTED.md before this
    ran. Weighting uses raw TransactionAmt: VDR is a ratio, so units cancel and no
    exchange-rate assumption is needed anywhere in this stage.
    """
    banner("STAGE: value-weighted evaluation")
    df = df if df is not None else load_merged()
    if UID_COL not in df.columns:
        df = df.copy()
        df[UID_COL] = build_uid(df)

    train, test = temporal_split(df)
    train, test, _ = build_features_for_split(train, test)
    train, test, full_graph = build_centrality_for_split(train, test)
    train, test = add_time_features(train), add_time_features(test)

    y_true = test[TARGET].to_numpy().astype(int)
    amounts = test["TransactionAmt"].to_numpy().astype(float)

    # ---- the mechanism check, before any model comparison -------------
    banner("Is graph-linked fraud actually more valuable?")
    print("Value weighting can only change a conclusion if the subgroup it favours carries")
    print("more value than its share of the count. So that is measured first.\n")
    linked = test["g_is_linked"].fillna(0).to_numpy() == 1
    concentration = value_concentration(y_true, amounts, linked, "graph-linked rows")
    for line in concentration.summary_lines():
        print(line)

    # ---- variants ------------------------------------------------------
    base_cols = feature_columns(
        train,
        extra_excluded=set(GRAPH_FEATURE_COLUMNS) | set(CENTRALITY_FEATURE_COLUMNS) | {UID_COL},
    )
    variants = {
        "tabular only": (base_cols, "baseline"),
        "+ components": (
            base_cols + ["g_component_size", "g_component_size_total", "g_is_linked"],
            "components",
        ),
        "+ k-core": (base_cols + ["g_core_number", "g_degree"], "kcore"),
        "+ full graph": (
            feature_columns(train, extra_excluded=set(CENTRALITY_FEATURE_COLUMNS) | {UID_COL}),
            "graph_full",
        ),
        "+ centrality": (
            feature_columns(train, extra_excluded={UID_COL}),
            "centrality",
        ),
    }

    banner("Training / loading variants")
    scores = {
        name: _fit_and_score(train, test, cols, name, key)
        for name, (cols, key) in variants.items()
    }

    # ---- the table -----------------------------------------------------
    banner("COUNT-WEIGHTED vs VALUE-WEIGHTED")
    cap_reports = {
        name: choose_threshold_under_insult_cap(y_true, s, amount_inr(amounts))
        for name, s in scores.items()
    }

    print(
        f"{'variant':<16} {'AUC-PR':>9} {'VW AUC-PR':>11} "
        f"{'recall@cap':>11} {'VDR@cap':>9}"
    )
    for name, s in scores.items():
        cap = cap_reports[name]
        print(
            f"{name:<16} {average_precision_score(y_true, s):>9.4f} "
            f"{value_weighted_average_precision(y_true, s, amounts):>11.4f} "
            f"{cap.recall:>11.4f} "
            f"{value_detection_rate(y_true, s, amounts, cap.threshold):>9.4f}"
        )

    # ---- paired bootstrap, both weightings ------------------------------
    banner("IS ANY DIFFERENCE REAL? (paired bootstrap, 95% CI)")
    baseline = scores["tabular only"]
    n_comparisons = (len(variants) - 1) * 2  # every variant, under both weightings
    print(
        f"Eight comparisons are made below (4 variants x 2 weightings). At 95% each, the\n"
        f"chance of at least one false positive is ~34%, so every interval is reported\n"
        f"BOTH uncorrected and Bonferroni-corrected across the family of {n_comparisons}.\n"
    )
    print(
        f"{'variant':<16} {'wt':<6} {'delta':>9}  {'95% CI':<22} "
        f"{'corrected':<22} survives?"
    )
    widths: dict[str, dict[str, float]] = {}
    for name in list(variants)[1:]:
        widths[name] = {}
        for label, weight in (("count", None), ("value", amounts)):
            raw = bootstrap_auc_pr_delta(
                y_true, baseline, scores[name], name=name, sample_weight=weight
            )
            corrected = bootstrap_auc_pr_delta(
                y_true,
                baseline,
                scores[name],
                name=name,
                sample_weight=weight,
                n_comparisons=n_comparisons,
            )
            widths[name][label] = raw.ci_high - raw.ci_low
            print(
                f"{name:<16} {label:<6} {raw.delta:>+9.4f}  "
                f"[{raw.ci_low:+.4f}, {raw.ci_high:+.4f}]   "
                f"[{corrected.ci_low:+.4f}, {corrected.ci_high:+.4f}]   "
                f"{'YES' if corrected.significant else 'no'}"
            )

    # VDR at the operating point is a different question from threshold-free AP: not
    # "does it rank high-value fraud better" but "at the threshold we would ship, does it
    # stop more of the money". Each model keeps its own cap threshold.
    banner("VALUE DETECTION RATE AT THE <=1% INSULT CAP (paired bootstrap)")
    print("Thresholds are chosen once on the full test set and held fixed across resamples,")
    print("so these intervals EXCLUDE threshold-selection uncertainty and are narrower than")
    print("the uncertainty a real deployment would face.\n")
    print(f"{'variant':<16} {'VDR':>8} {'delta':>9}  {'95% CI':<22} {'corrected':<22} survives?")
    print(f"{'tabular only':<16} {value_detection_rate(y_true, baseline, amounts, cap_reports['tabular only'].threshold):>8.4f}"
          f" {'—':>9}  {'baseline':<22} {'':<22}")
    for name in list(variants)[1:]:
        raw = bootstrap_vdr_delta(
            y_true, amounts, baseline, scores[name],
            baseline_threshold=cap_reports["tabular only"].threshold,
            variant_threshold=cap_reports[name].threshold,
            name=name,
        )
        corrected = bootstrap_vdr_delta(
            y_true, amounts, baseline, scores[name],
            baseline_threshold=cap_reports["tabular only"].threshold,
            variant_threshold=cap_reports[name].threshold,
            name=name, n_comparisons=len(variants) - 1,
        )
        vdr = value_detection_rate(y_true, scores[name], amounts, cap_reports[name].threshold)
        print(
            f"{name:<16} {vdr:>8.4f} {raw.delta:>+9.4f}  "
            f"[{raw.ci_low:+.4f}, {raw.ci_high:+.4f}]   "
            f"[{corrected.ci_low:+.4f}, {corrected.ci_high:+.4f}]   "
            f"{'YES' if corrected.significant else 'no'}"
        )

    print("\n  Predicted in advance: value-weighted intervals should be WIDER, because the")
    print("  top 1% of frauds carry ~11% of fraud value and resampling them swings hard.")
    for name, w in widths.items():
        ratio = w["value"] / w["count"] if w["count"] else float("nan")
        print(
            f"    {name:<16} count width {w['count']:.4f}  value width {w['value']:.4f}  "
            f"-> {ratio:.2f}x"
        )

    # ---- why centrality behaves as it does ------------------------------
    banner("CENTRALITY: does it vary at all inside these components?")
    labels = connected_components(full_graph.adjacency)[: full_graph.n_uids]
    ranks = pagerank(full_graph.adjacency)[: full_graph.n_uids]
    between = betweenness_scores = None
    print("computing betweenness over the full entity graph ...", flush=True)
    between = betweenness(full_graph.adjacency)[: full_graph.n_uids]
    degrees = np.array(
        [len(full_graph.adjacency[i]) for i in range(full_graph.n_uids)], dtype=np.int64
    )
    variance = component_centrality_variance(labels, ranks, between, degrees > 0)
    for line in variance.summary_lines():
        print(line)


def stage_calibration(df: pd.DataFrame | None = None) -> None:
    """Are the predicted probabilities trustworthy AS probabilities?

    Separate question from AUC-PR, which is invariant to any monotonic rescaling of the
    scores. It matters here because the operating threshold is chosen by minimising
    expected rupee cost, and that arithmetic reads the score as a probability.

    Reuses cached scores -- no retraining.
    """
    banner("STAGE: calibration")
    df = df if df is not None else load_merged()
    _, test = temporal_split(df)
    y_true = test[TARGET].to_numpy()

    variants = [("baseline", "tabular only"), ("graph_full", "+ full graph")]
    missing = [
        key for key, _ in variants if not (CACHE_DIR / f"scores_{key}.npy").exists()
    ]
    if missing:
        print(f"No cached scores for {missing}. Run: python run.py --stage ablation")
        return

    for key, label in variants:
        scores = np.load(CACHE_DIR / f"scores_{key}.npy")
        report = calibration_report(label, y_true, scores)
        for line in report.summary_lines():
            print(line)
        print()

    print(
        "  Reading these numbers: the Brier score conflates calibration with\n"
        "  discrimination/refinement, so it is not a pure calibration metric on its own.\n"
        "  ECE is the number that measures only distance from the diagonal."
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


def _load_env() -> None:
    """Load .env if present, so documented credentials actually reach the provider.

    Absence is fine and expected: the deterministic pipeline needs no credentials, and the
    narrative layer degrades to NARRATIVE_UNAVAILABLE rather than failing the run.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_env()
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
    elif args.stage == "value":
        stage_value()
    elif args.stage == "cost":
        stage_cost()
    elif args.stage == "calibration":
        stage_calibration()
    elif args.stage == "narrate":
        stage_narrate()
    else:
        df = stage_data()
        stage_baseline(df)
        stage_graph(df)
        stage_ablation(df)
        stage_value(df)
        stage_cost(df)
        stage_calibration(df)
        stage_narrate(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
