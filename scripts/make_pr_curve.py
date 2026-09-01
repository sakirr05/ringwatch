"""Render the precision-recall curve from cached scores.

Writes docs/pr_curve.png for the README and the pitch video. Reads cached predictions
rather than retraining, so it is fast and reproduces the exact numbers reported.

    python scripts/make_pr_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data import CACHE_DIR, TARGET, load_merged  # noqa: E402
from core.evaluate import MAX_ACCEPTABLE_INSULT_RATE  # noqa: E402
from core.split import temporal_split  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

VARIANTS = [
    ("baseline", "tabular only", "#2d6a4f", 2.4),
    ("components", "+ components", "#3d5a99", 1.4),
    ("kcore", "+ k-core", "#b03a3a", 1.4),
    ("graph_full", "+ full graph", "#b06500", 1.4),
]


def main() -> int:
    df = load_merged()
    _, test = temporal_split(df)
    y = test[TARGET].to_numpy().astype(int)
    prevalence = y.mean()

    missing = [k for k, *_ in VARIANTS if not (CACHE_DIR / f"scores_{k}.npy").exists()]
    if missing:
        print(f"missing cached scores for {missing}; run: python run.py --stage ablation")
        return 1

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)

    for key, label, colour, width in VARIANTS:
        scores = np.load(CACHE_DIR / f"scores_{key}.npy")
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.plot(
            recall, precision, color=colour, linewidth=width, label=f"{label} (AP={ap:.4f})"
        )

    ax.axhline(
        prevalence,
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        label=f"random baseline (AP={prevalence:.4f})",
    )

    ax.set_xlabel("Recall — share of fraud caught")
    ax.set_ylabel("Precision — share of declines that were really fraud")
    ax.set_title(
        "RingWatch precision–recall, temporally held-out test set\n"
        f"{len(y):,} transactions · {y.sum():,} fraud ({100 * prevalence:.2f}%)  —  "
        "the four curves overlap: the graph layer adds nothing",
        fontsize=10,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # The dashed prevalence line is the honest reference point for AUC-PR. Annotating it
    # keeps a reader from mentally comparing against the 0.5 that belongs to AUC-ROC.
    ax.annotate(
        "a model that never predicts fraud\nscores 96.5% accuracy and sits here",
        xy=(0.62, prevalence),
        xytext=(0.30, 0.14),
        fontsize=8,
        color="#555555",
        arrowprops={"arrowstyle": "->", "color": "#888888", "linewidth": 0.9},
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "pr_curve.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
