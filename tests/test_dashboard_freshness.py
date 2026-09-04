"""Guards against dashboard prose drifting away from the artifact it renders.

This file exists because of a Phase 11 audit finding. The §2b callout — the one describing
the result that nearly became a false headline — had four confidence-interval bounds typed
into the HTML by hand. The delta beside them was right; all four bounds were stale, off in
the fourth decimal from `docs/results.json`.

Nothing caught it. Every other check passed: the clean-cache re-run reproduced the artifact
bit-identically, the README matched the artifact on 28 of 28 figures, and the page rendered
without error. A hand-typed number is invisible to all of that, because it is not derived
from anything — it just sits there being wrong.

So the guard is structural rather than numeric: a figure the artifact already carries must
be rendered *from* the artifact, not retyped beside it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = REPO_ROOT / "app" / "templates" / "dashboard.html"
RESULTS_PATH = REPO_ROOT / "docs" / "results.json"

pytestmark = pytest.mark.skipif(
    not RESULTS_PATH.exists(), reason="docs/results.json not generated"
)


@pytest.fixture(scope="module")
def flat_html() -> str:
    """Whitespace-normalised, so assertions are about figures and not line wrapping."""
    return re.sub(r"\s+", " ", TestClient(app).get("/").text)


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def centrality(results: dict) -> dict:
    return next(
        v for v in results["value_weighted"]["variants"] if v["key"] == "centrality"
    )


# --------------------------------------------------------------------------
# the figures that were stale
# --------------------------------------------------------------------------


def test_the_near_false_headline_figures_match_the_artifact(flat_html, results):
    """All five: the delta, both uncorrected bounds, both corrected bounds."""
    c = centrality(results)
    assert f"{c['value_delta']:+.4f}" in flat_html
    assert f"[{c['value_ci'][0]:+.4f}, {c['value_ci'][1]:+.4f}]" in flat_html
    assert (
        f"[{c['value_ci_corrected'][0]:+.4f}, {c['value_ci_corrected'][1]:+.4f}]"
        in flat_html
    )


def test_the_comparison_count_matches_the_artifact(flat_html, results):
    """'the eight comparisons actually run' must not outlive a change to the family."""
    n = results["value_weighted"]["n_comparisons"]
    assert f"{n} comparisons actually run" in flat_html


def test_the_corrected_confidence_level_matches_the_bonferroni_arithmetic(flat_html, results):
    """The column header states a level; it has to be the level actually applied."""
    n = results["value_weighted"]["n_comparisons"]
    assert f"corrected {100 - 5.0 / n:.1f}% CI" in flat_html


def test_the_result_is_still_reported_as_not_surviving_correction(flat_html, results):
    """The honest conclusion must not be lost while fixing the figures under it."""
    c = centrality(results)
    assert c["value_significant"] is True
    assert c["value_survives_correction"] is False
    assert "does not survive correction" in flat_html
    assert "spans zero" in flat_html


# --------------------------------------------------------------------------
# the structural guard
# --------------------------------------------------------------------------


def test_the_callout_derives_its_figures_rather_than_retyping_them():
    """A hand-typed figure is invisible to every other check in this suite."""
    html = TEMPLATE.read_text()
    start = html.index("The result that nearly became a false headline")
    callout = html[start : html.index("</div>", start)]

    assert "cen.value_delta" in callout
    assert "cen.value_ci[0]" in callout and "cen.value_ci[1]" in callout
    assert "cen.value_ci_corrected[0]" in callout
    assert "vw.n_comparisons" in callout

    # And no four-decimal literal left behind beside them.
    literals = re.findall(r"[−+-]?\d\.\d{4}\b", re.sub(r"\{\{.*?\}\}", "", callout, flags=re.S))
    assert not literals, f"hand-typed figures still in the callout: {literals}"


def test_no_headline_metric_is_hardcoded_anywhere_in_the_template(results):
    """The load-bearing figures must all come from the artifact via Jinja."""
    html = TEMPLATE.read_text()
    body = html.split("</style>", 1)[-1]
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    body = re.sub(r"\{\{.*?\}\}", "", body, flags=re.S)
    body = re.sub(r"\{%.*?%\}", "", body, flags=re.S)

    abl = {r["key"]: r for r in results["ablation"]}
    forbidden = {
        f"{abl['baseline']['auc_pr']:.4f}": "baseline AUC-PR",
        f"{results['ring_evidence']['z_score']:.1f}": "ring z-score",
        f"{results['graph']['test_rows_linked_pct']:.2f}": "graph coverage",
        f"{results['cluster_outcomes']['enrichment']:.1f}": "cluster enrichment",
    }
    for literal, label in forbidden.items():
        assert literal not in body, f"{label} ({literal}) is hardcoded in the template"
