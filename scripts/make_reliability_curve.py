"""Render the reliability diagram from cached scores.

Writes docs/reliability.png for the README and the pitch video. Reads cached predictions
rather than retraining, so it is fast and reproduces the exact numbers reported.

    python scripts/make_reliability_curve.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.calibration import calibration_report  # noqa: E402
from core.data import CACHE_DIR, TARGET, load_merged  # noqa: E402
from core.split import temporal_split  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

VARIANTS = [
    ("baseline", "tabular only", "#2d6a4f"),
    ("graph_full", "+ full graph", "#b06500"),
]


def main() -> int:
    df = load_merged()
    _, test = temporal_split(df)
    y = test[TARGET].to_numpy().astype(int)

    missing = [k for k, *_ in VARIANTS if not (CACHE_DIR / f"scores_{k}.npy").exists()]
    if missing:
        print(f"missing cached scores for {missing}; run: python run.py --stage ablation")
        return 1

    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=150)

    # Log-log axes: at a 3.44% base rate the predicted probabilities span three orders of
    # magnitude and bunch near zero, so a linear reliability plot collapses nine of the ten
    # bins into the bottom-left corner and shows nothing.
    ax.plot(
        [1e-4, 1.0],
        [1e-4, 1.0],
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        label="perfect calibration",
        zorder=1,
    )

    for key, label, colour in VARIANTS:
        scores = np.load(CACHE_DIR / f"scores_{key}.npy")
        report = calibration_report(label, y, scores)
        ax.plot(
            report.prob_pred,
            report.prob_true,
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=colour,
            label=f"{label}  (ECE={report.ece:.4f}, Brier={report.brier:.5f})",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed fraud rate (bin)")
    ax.set_title(
        "RingWatch reliability diagram — 10 quantile bins, held-out test set\n"
        "every point sits ABOVE the diagonal: the model is systematically under-confident",
        fontsize=10,
    )
    ax.grid(alpha=0.25, linewidth=0.6, which="both")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    ax.annotate(
        "points above the line mean\nreal fraud is MORE common\nthan the score claims",
        xy=(0.0178, 0.0409),
        xytext=(0.0009, 0.15),
        fontsize=8,
        color="#555555",
        arrowprops={"arrowstyle": "->", "color": "#888888", "linewidth": 0.9},
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "reliability.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
