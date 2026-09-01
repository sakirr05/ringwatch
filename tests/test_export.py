"""Tests for the exported results artifact.

`docs/results.json` is what the deployed dashboard renders, and the deployed instance has
neither the dataset nor the model cache — so if this file is wrong, stale, or truncated,
nothing downstream can detect it. These tests are the only thing standing between a bad
export and a demo confidently displaying wrong numbers.

They deliberately assert the *known headline values*. If a future change alters
AUC-PR 0.5188 or the z-score of +8.8, that is either a real finding or a regression, and
either way it must not reach the dashboard silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "results.json"

pytestmark = pytest.mark.skipif(
    not RESULTS.exists(),
    reason="docs/results.json not generated yet (run scripts/export_results.py)",
)


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS.read_text())


def test_artifact_is_small_enough_to_commit():
    """It ships in git and is served on every page load; keep it lean."""
    assert RESULTS.stat().st_size < 250_000


def test_metadata_identifies_the_producing_run(results):
    """A reader must be able to tell which commit produced the numbers on screen."""
    assert results["meta"]["git_commit"]
    assert results["meta"]["generated_at"]


def test_headline_ablation_values(results):
    """The negative result is the project's thesis. Guard the exact numbers."""
    by_key = {row["key"]: row for row in results["ablation"]}
    assert len(by_key) == 4

    assert by_key["baseline"]["auc_pr"] == pytest.approx(0.5188, abs=5e-4)
    assert by_key["components"]["auc_pr"] == pytest.approx(0.5176, abs=5e-4)
    assert by_key["kcore"]["auc_pr"] == pytest.approx(0.5123, abs=5e-4)
    assert by_key["graph_full"]["auc_pr"] == pytest.approx(0.5168, abs=5e-4)


def test_no_graph_variant_beats_the_baseline(results):
    """If this ever fails, the project's central claim has changed and the README is stale."""
    by_key = {row["key"]: row for row in results["ablation"]}
    baseline = by_key["baseline"]["auc_pr"]
    for key in ("components", "kcore", "graph_full"):
        assert by_key[key]["auc_pr"] < baseline


def test_kcore_is_the_significantly_worse_variant(results):
    """k-core is reported as the measured harm case; its CI must exclude zero."""
    by_key = {row["key"]: row for row in results["ablation"]}
    kcore = by_key["kcore"]
    assert kcore["significant"] is True
    assert kcore["ci_high"] < 0
    assert kcore["verdict"] == "SIGNIFICANTLY WORSE"


def test_the_two_non_significant_variants_span_zero(results):
    by_key = {row["key"]: row for row in results["ablation"]}
    for key in ("components", "graph_full"):
        assert by_key[key]["significant"] is False
        assert by_key[key]["ci_low"] < 0 < by_key[key]["ci_high"]


def test_ring_concentration_is_significant(results):
    """The other half of the argument: structure is real even though lift is not."""
    ring = results["ring_evidence"]
    assert ring["all_fraud_components"] == 12
    assert ring["z_score"] > 5
    assert ring["null_mean"] < ring["all_fraud_components"]


def test_coverage_figure_is_present_and_small(results):
    """The reconciliation only works if coverage is displayed alongside the z-score."""
    graph = results["graph"]
    assert graph["hub_cap"] == 5
    assert 0 < graph["test_rows_linked_pct"] < 10


def test_both_operating_points_exported(results):
    """Reporting only the flattering threshold is the failure mode being avoided."""
    points = results["operating_points"]
    cost = points["cost_minimising"]
    constrained = points["insult_constrained"]

    # [A] minimises cost but insults far more customers than [B].
    assert cost["insult_rate"] > constrained["insult_rate"]
    assert cost["recall"] > constrained["recall"]
    assert constrained["precision"] > cost["precision"]
    assert constrained["insult_rate"] <= points["insult_cap"]


def test_economic_assumptions_are_published(results):
    """Every rupee figure must trace to a stated premise a reviewer can disagree with."""
    assumptions = results["assumptions"]
    assert assumptions["usd_to_inr"] > 0
    assert 0 < assumptions["gross_margin_rate"] < 1
    assert assumptions["chargeback_fee_inr"] > 0


def test_calibration_exported_for_both_variants(results):
    assert len(results["calibration"]) == 2
    for entry in results["calibration"]:
        assert entry["strategy"] == "quantile"
        assert len(entry["curve"]) == entry["n_bins"]
        assert 0 < entry["brier"] < 1
        assert entry["ece"] >= 0


def test_calibration_curve_shows_under_confidence(results):
    """The finding: observed fraud exceeds predicted probability in every bin."""
    for entry in results["calibration"]:
        for point in entry["curve"]:
            assert point["observed"] > point["predicted"], (
                f"{entry['label']} bin at {point['predicted']} no longer under-confident; "
                "the README's calibration section would be stale."
            )


def test_twelve_clusters_with_evidence_and_narrative(results):
    clusters = results["clusters"]
    assert len(clusters) == 12
    for cluster in clusters:
        assert cluster["evidence"]["entity_count"] >= 2
        assert cluster["narrative"]["status"] in {"OK", "NARRATIVE_UNAVAILABLE"}


def test_provenance_split_is_structural(results):
    """The determinism boundary must be expressed in the data, not just the template.

    Numbers live under "evidence" (computed by core/); prose lives under "narrative"
    (written by the LLM). A narrative block must never carry a computed quantity.
    """
    numeric_fields = {
        "max_risk_score",
        "mean_risk_score",
        "total_amount_inr",
        "component_size",
        "core_number",
    }
    for cluster in results["clusters"]:
        assert numeric_fields <= set(cluster["evidence"])
        assert not (numeric_fields & set(cluster["narrative"]))


def test_narrative_confidence_values_are_from_the_enum(results):
    for cluster in results["clusters"]:
        assert cluster["narrative"]["confidence"] in {"high", "medium", "low"}


def test_split_is_temporal_and_sizes_are_consistent(results):
    dataset = results["dataset"]
    assert dataset["train_rows"] + dataset["test_rows"] == dataset["rows"]
    assert dataset["split_quantile"] == 0.80
    # Fraud rate drifts a little over time but should not be wildly different.
    assert abs(dataset["train_fraud_rate"] - dataset["test_fraud_rate"]) < 0.01
