"""Tests for the small-multiples cluster grid.

The grid introduces the one thing on this dashboard that touches held-out labels, so most
of these tests are about not overclaiming with them. In particular:

  * every card must be reachable and open a real evidence panel (no orphans),
  * the zero-fraud cluster must be shown, not quietly dropped,
  * "all N fraud" must be accompanied by the statement that it is not a ring claim,
  * the enrichment figure must not be allowed to read as predictive lift, which §1 measured
    and rejected.

`test_the_all_fraud_badge_count_matches_the_data` exists because the ring-concentration test
also reports 12 all-fraud components. Those are components of the entity graph, not these
flagged clusters, and only 2 of the 12 clusters are all-fraud. Badging all twelve would have
been a false claim that the coincidence of the two numbers makes easy to miss.
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
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def html(client) -> str:
    return client.get("/").text


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def grid_script(template: str) -> str:
    start = template.index("SMALL MULTIPLES -> EVIDENCE PANEL")
    return template[start : template.index("THRESHOLD EXPLORER", start)]


# --------------------------------------------------------------------------
# every cluster is on the grid, and every card opens something
# --------------------------------------------------------------------------


def test_one_card_per_cluster(html, results):
    assert html.count('class="cardlet"') == len(results["clusters"])


def test_every_card_has_a_matching_evidence_panel(html):
    """An orphan card would be a button that silently does nothing."""
    cards = set(re.findall(r'class="cardlet" data-cluster="(\d+)"', html))
    panels = set(re.findall(r'<details class="cluster" id="cluster-(\d+)"', html))
    assert cards, "no cards rendered"
    assert cards == panels


def test_cards_are_buttons_and_panels_are_details(html):
    """The section must still work with JavaScript off."""
    assert '<button type="button" class="cardlet"' in html
    assert '<details class="cluster" id="cluster-' in html


def test_cards_report_their_expanded_state(html):
    cards = re.findall(r'<button type="button" class="cardlet"[^>]*>', html, re.S)
    assert cards
    for card in cards:
        assert 'aria-expanded="false"' in card


def test_card_facts_match_the_artifact(html, results):
    """Each card must show that cluster's own computed figures, not a neighbour's."""
    for cluster in results["clusters"]:
        evidence = cluster["evidence"]
        cid = evidence["cluster_id"]
        match = re.search(
            rf'<button type="button" class="cardlet" data-cluster="{cid}".*?</button>',
            html, re.S,
        )
        assert match, f"no card for cluster {cid}"
        card = match.group()
        assert f"{evidence['max_risk_score']:.3f}" in card
        assert f"{evidence['entity_count']} ent" in card
        assert f"{evidence['transaction_count']} txn" in card
        assert f"core {evidence['core_number']}" in card
        assert f"{evidence['span_days']}d" in card


# --------------------------------------------------------------------------
# the outcome badges do not overclaim
# --------------------------------------------------------------------------


def test_the_all_fraud_badge_count_matches_the_data(html, results):
    """Not 12. The ring test's 12 all-fraud COMPONENTS are a different set entirely."""
    expected = results["cluster_outcomes"]["all_fraud_clusters"]
    badges = re.findall(r'class="cardlet-outcome all">\s*all \d+ fraud', html)
    assert len(badges) == expected
    assert expected < len(results["clusters"]), (
        "if every cluster were all-fraud this test would stop being meaningful"
    )


def test_the_zero_fraud_cluster_is_shown_not_dropped(html, results):
    """A grid that hides its own false alarm is a highlight reel."""
    expected = results["cluster_outcomes"]["zero_fraud_clusters"]
    if not expected:
        pytest.skip("no zero-fraud cluster in this run")
    assert html.count('class="cardlet-outcome none"') == expected
    assert "no fraud" in html


def test_all_fraud_is_explicitly_not_a_ring_claim(html):
    text = html.lower()
    assert "is not a ring" in text
    assert "coordinate nothing" in text
    assert "no ring-level ground truth" in text


def test_the_enrichment_is_not_presented_as_predictive_lift(html):
    """§1 measured lift and found none. These two results must not be allowed to blur."""
    assert "cluster-surfacing" in html
    assert "does not improve AUC-PR" in html


def test_the_page_says_the_engine_never_saw_these_labels(html):
    assert "engine never saw these labels" in html


def test_the_enrichment_figure_matches_the_artifact(html, results):
    """Whitespace-normalised, so the assertion is about the numbers and not the wrapping."""
    summary = results["cluster_outcomes"]
    flat = re.sub(r"\s+", " ", html)

    assert f"{summary['enrichment']:.1f}×" in flat
    assert (
        f"{summary['fraud_transactions']} of {summary['transactions']} transactions "
        f"({summary['fraud_share'] * 100:.1f}%)" in flat
    )
    assert f"base rate of {summary['base_rate'] * 100:.2f}%" in flat
    assert (
        f"{summary['all_fraud_clusters']} of {summary['clusters']} clusters qualify" in flat
    )


# --------------------------------------------------------------------------
# the label never reaches the narrative layer
# --------------------------------------------------------------------------


def test_no_cluster_exports_ground_truth_inside_its_evidence(results):
    """The evidence block is what the model saw. It must be label-free."""
    for cluster in results["clusters"]:
        keys = set(cluster["evidence"])
        for forbidden in ("fraud_transactions", "all_fraud", "fraud_share", "isFraud"):
            assert forbidden not in keys
        # The outcome lives in its own sibling block.
        assert "fraud_transactions" in cluster["outcome"]


def test_the_case_file_the_orchestrator_saw_carries_no_outcome(results):
    for cluster in results["clusters"]:
        keys = set(cluster["case"])
        for forbidden in ("all_fraud", "fraud_transactions", "caught", "missed"):
            assert forbidden not in keys


# --------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------


def test_clicking_a_card_opens_its_panel_and_scrolls_to_it():
    script = grid_script(TEMPLATE.read_text())
    assert "panel.open = true" in script
    assert "scrollIntoView" in script


def test_closing_a_panel_by_hand_clears_the_card_state():
    """Otherwise the card claims a panel is open when the reader has closed it."""
    script = grid_script(TEMPLATE.read_text())
    assert 'panel.addEventListener("toggle", sync)' in script


def test_the_grid_script_reaches_no_network():
    script = grid_script(TEMPLATE.read_text())
    for network in ("fetch(", "XMLHttpRequest", "http://", "https://"):
        assert network not in script


def test_the_grid_prints(html):
    """Unlike the slider, a contact sheet is useful on paper."""
    css = TEMPLATE.read_text().split("@media print")[1][:3000]
    assert ".grid-multiples" in css
    assert "display: none" not in css.split(".grid-multiples")[1][:120]
