"""Reproduce the README's hub-suppression coverage sweep.

WHY THIS EXISTS
---------------
The README publishes a table answering the first objection anyone raises about the graph
layer -- "did you try giving it more coverage?" -- at hub caps 5 / 20 / 50. Until the
Phase 11 audit, those numbers had no reproducible path: they were produced by editing
`MAX_GROUP_SIZE` in `core/graph.py` by hand, running the ablation, and writing the result
down. Every other figure in this project regenerates from `run.py` or
`scripts/export_results.py`; this table did not, which made it the one published claim a
reviewer had to take on trust.

It is not a new experiment. It calls the same `build_features_for_split`, the same
`train_model`, and the same `evaluate` the shipped ablation uses, with the cap threaded
through as a parameter whose default is unchanged.

    python scripts/coverage_sweep.py                 # caps 5, 20, 50
    python scripts/coverage_sweep.py --caps 5 10 20  # any caps you like

Cost: one model per cap beyond the cached baseline, several minutes of CPU each. Scores
are cached as `scores_graph_cap{N}.npy`, matching the names the original ad-hoc runs left
behind.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.data import CACHE_DIR, TARGET, load_merged  # noqa: E402
from core.evaluate import bootstrap_auc_pr_delta, evaluate  # noqa: E402
from core.features import add_time_features, feature_columns  # noqa: E402
from core.graph import (  # noqa: E402
    MAX_GROUP_SIZE,
    UID_COL,
    build_features_for_split,
    build_uid,
)
from core.model import train_model  # noqa: E402
from core.split import temporal_split  # noqa: E402

DEFAULT_CAPS = (5, 20, 50)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    args = parser.parse_args()

    baseline_path = CACHE_DIR / "scores_baseline.npy"
    if not baseline_path.exists():
        print("Missing scores_baseline.npy. Run: python run.py --stage ablation")
        return 1

    print("Loading data and rebuilding the temporal split ...", flush=True)
    df = load_merged()
    df[UID_COL] = build_uid(df)
    train, test = temporal_split(df)
    y_true = test[TARGET].to_numpy().astype(int)
    amounts = test["TransactionAmt"].to_numpy()

    baseline_scores = np.load(baseline_path)
    baseline_report = evaluate("tabular only", y_true, baseline_scores, amounts)

    rows = []
    for cap in args.caps:
        print(f"\n--- hub cap {cap} ---", flush=True)
        train_g, test_g, _ = build_features_for_split(train, test, max_group_size=cap)
        train_g = add_time_features(train_g)
        test_g = add_time_features(test_g)

        coverage = float((test_g["g_is_linked"].fillna(0) == 1).mean() * 100)
        print(f"  coverage: {coverage:.2f}% of test rows", flush=True)

        # Cap 5 IS the shipped configuration, so its scores already exist as the `+ full
        # graph` ablation variant. Retraining it here would burn several minutes to produce
        # a second model for the same configuration -- and if it disagreed with the
        # published row by even a hair, the table would contradict the ablation above it.
        cache = (
            CACHE_DIR / "scores_graph_full.npy"
            if cap == MAX_GROUP_SIZE
            else CACHE_DIR / f"scores_graph_cap{cap}.npy"
        )
        if cache.exists():
            print(f"  using cached {cache.name}", flush=True)
            scores = np.load(cache)
        else:
            print("  training ...", flush=True)
            columns = feature_columns(train_g)
            model = train_model(train_g, columns)
            print(
                f"  best_iter={model.best_iteration} "
                f"validation AUC-PR={model.validation_ap:.4f}",
                flush=True,
            )
            scores = model.predict(test_g)
            np.save(cache, scores)

        report = evaluate(f"cap {cap}", y_true, scores, amounts)
        delta = bootstrap_auc_pr_delta(
            y_true, baseline_scores, scores, name=f"cap {cap}"
        )
        rows.append((cap, coverage, report.auc_pr, delta))

    print(f"\n{'cap':>5} {'coverage':>10} {'AUC-PR':>9} {'delta':>9}  95% CI")
    print(f"{'base':>5} {'—':>10} {baseline_report.auc_pr:>9.4f} {'—':>9}")
    for cap, coverage, auc_pr, delta in rows:
        print(
            f"{cap:>5} {coverage:>9.2f}% {auc_pr:>9.4f} {delta.delta:>+9.4f}"
            f"  [{delta.ci_low:+.4f}, {delta.ci_high:+.4f}]  {delta.verdict()}"
        )

    print(
        "\nRead this against the bootstrap noise band, not as a ranking. A positive delta"
        "\nan order of magnitude inside the CI is not evidence that a higher cap helps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
