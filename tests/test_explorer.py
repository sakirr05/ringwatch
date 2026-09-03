"""Tests for the dashboard's threshold explorer.

The claim this section makes on the page is "nothing here is computed in your browser."
That is a structural claim, so it is tested structurally: the embedded curve must reproduce
the published operating points exactly, and the script must not contain the arithmetic that
would let it derive a metric of its own.

`test_the_script_derives_no_metric_of_its_own` is the load-bearing one. A slider that
recomputed precision from tp/fp in JavaScript would be a second, unvalidated implementation
of the costing model sitting inside a page whose entire thesis is that it has none.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from core.evaluate import MAX_ACCEPTABLE_INSULT_RATE

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


def embedded_sweep(html: str) -> list[dict]:
    match = re.search(
        r'<script type="application/json" id="sweep-data">(.*?)</script>', html, re.S
    )
    assert match, "the curve is not embedded in the page"
    return json.loads(match.group(1))


def explorer_script(template: str) -> str:
    start = template.index("THRESHOLD EXPLORER")
    return template[start : template.index("FIGURE LOADING STATE", start)]


# --------------------------------------------------------------------------
# the page reads a curve it did not compute
# --------------------------------------------------------------------------


def test_the_curve_is_present_and_non_trivial(html, results):
    sweep = embedded_sweep(html)
    assert len(sweep) == len(results["operating_points"]["sweep"])
    assert len(sweep) > 50, "too few points for a slider to be meaningful"


def test_the_embedded_curve_is_byte_identical_to_the_artifact(html, results):
    """The page must not reshape, round, or filter the committed data on its way out."""
    assert embedded_sweep(html) == results["operating_points"]["sweep"]


def test_both_published_operating_points_land_exactly_on_the_curve(html, results):
    """A marked point that only *nearly* matched would contradict the panels above it."""
    sweep = embedded_sweep(html)
    ops = results["operating_points"]

    for name in ("cost_minimising", "insult_constrained"):
        published = ops[name]
        matches = [
            p for p in sweep if abs(p["threshold"] - published["threshold"]) < 1e-9
        ]
        assert matches, f"{name} threshold is not on the curve"
        point = matches[0]
        assert point["true_positives"] == published["true_positives"]
        assert point["false_positives"] == published["false_positives"]
        assert point["false_negatives"] == published["false_negatives"]
        assert point["precision"] == pytest.approx(published["precision"], abs=1e-6)
        assert point["recall"] == pytest.approx(published["recall"], abs=1e-6)
        assert point["insult_rate"] == pytest.approx(published["insult_rate"], abs=1e-7)


def test_the_curve_is_monotone_in_the_direction_the_ui_assumes(html):
    """The unshippable band is drawn as one contiguous span from the left."""
    sweep = embedded_sweep(html)
    thresholds = [p["threshold"] for p in sweep]
    assert thresholds == sorted(thresholds)
    flags = [p["insult_rate"] > MAX_ACCEPTABLE_INSULT_RATE for p in sweep]
    assert flags == sorted(flags, reverse=True), "the unshippable region is not contiguous"


def test_the_slider_range_matches_the_number_of_points(html):
    match = re.search(r'<input type="range" id="thr"[^>]*max="(\d+)"', html)
    assert match, "no threshold slider rendered"
    assert int(match.group(1)) == len(embedded_sweep(html)) - 1


def test_the_declared_cap_matches_the_engine(html):
    match = re.search(r'id="explorer"[^>]*data-cap="([\d.]+)"', html, re.S)
    assert match
    assert float(match.group(1)) == MAX_ACCEPTABLE_INSULT_RATE


# --------------------------------------------------------------------------
# the browser computes nothing
# --------------------------------------------------------------------------


def test_the_script_derives_no_metric_of_its_own():
    """Every displayed metric must be a lookup. Recomputing one would fork the cost model."""
    script = explorer_script(TEMPLATE.read_text())

    for derived in (
        "true_positives +",
        "true_positives/",
        "/ (p.true_positives",
        "false_positives)",
        "fraud_missed_inr +",
        "* 1200",
        "0.12",
    ):
        assert derived not in script, f"the explorer appears to compute: {derived}"

    # It must READ each metric it shows.
    for field in ("p.precision", "p.recall", "p.insult_rate", "p.total_cost_inr"):
        assert field in script, f"{field} is not read from the curve"


def test_the_explorer_loads_no_external_resource():
    script = explorer_script(TEMPLATE.read_text())
    for network in ("fetch(", "XMLHttpRequest", "import(", "//cdn", "http://", "https://"):
        assert network not in script, f"the explorer reaches the network: {network}"


def test_the_curve_is_embedded_as_inert_data_not_executable_js(html):
    """A non-JS script type is never parsed as code, so the blob cannot become an injection."""
    assert '<script type="application/json" id="sweep-data">' in html
    assert "var sweepData =" not in html
    assert "const sweepData =" not in html


# --------------------------------------------------------------------------
# the honest caveat survives
# --------------------------------------------------------------------------


def test_the_unshippable_region_is_labelled_as_such(html):
    assert "OPERATIONALLY UNSHIPPABLE" in html
    assert "does not price customer churn" in html


def test_the_existing_unshippable_caveat_is_still_on_the_page(html):
    """The slider must not have quietly replaced the prose explaining why [A] is wrong."""
    assert "Why [A] is unshippable" in html
    assert "unpriced cost" in html.lower()


def test_the_page_states_that_the_browser_computes_nothing(html):
    assert "Nothing here is computed in your browser" in html
    assert "cost_at_threshold" in html


def test_the_explorer_opens_on_the_shippable_point(html):
    """Opening on [A] would present an unshippable configuration as the default answer."""
    script = explorer_script(TEMPLATE.read_text())
    assert "input.value = idxB;" in script
    assert "render(idxB);" in script


def test_the_explorer_is_hidden_in_print():
    """A printed slider cannot be dragged, and a frozen readout would read as a result."""
    css = TEMPLATE.read_text()
    assert "#explorer" in css.split("@media print")[1][:2000]


def test_the_slider_is_keyboard_reachable_and_labelled(html):
    match = re.search(r'<input type="range" id="thr"[^>]*>', html, re.S)
    assert match
    tag = match.group()
    assert 'aria-label="Decision threshold"' in tag
    assert "aria-describedby" in tag
    assert "focus-visible" in TEMPLATE.read_text()
