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
from core.model import train_model
from core.split import split_summary, temporal_split

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


def stage_graph(df: pd.DataFrame | None = None) -> None:
    banner("STAGE: graph")
    print("Not yet implemented (Phase 4).")


def stage_ablation(df: pd.DataFrame | None = None) -> None:
    banner("STAGE: ablation")
    print("Not yet implemented (Phase 5).")


def stage_narrate(df: pd.DataFrame | None = None) -> None:
    banner("STAGE: narrate")
    print("Not yet implemented (Phase 6).")


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
