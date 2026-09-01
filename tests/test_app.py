"""Tests for the web layer.

The load-bearing one is `test_app_layer_computes_nothing`. The project's claim is that the
dashboard displays results and never produces them; like the AI/determinism boundary, that
is enforced by walking the import graph rather than by a promise in a docstring.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.results import RESULTS_PATH, ResultsUnavailable, load_results

APP_DIR = Path(__file__).resolve().parent.parent / "app"

pytestmark = pytest.mark.skipif(
    not RESULTS_PATH.exists(),
    reason="docs/results.json not generated yet (run scripts/export_results.py)",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def app_source_files() -> list[Path]:
    return sorted(p for p in APP_DIR.glob("*.py") if p.name != "__init__.py")


def test_there_are_app_modules_to_check():
    """Guard against this file silently passing on an empty glob."""
    assert app_source_files()


@pytest.mark.parametrize("path", app_source_files(), ids=lambda p: p.name)
def test_app_layer_computes_nothing(path: Path):
    """No module under app/ may import a modelling or analysis library.

    If this fails, a web request can trigger a computation, and the numbers on the
    dashboard are no longer guaranteed to be the numbers the pipeline produced and
    committed.

    `core.graph` is permitted for the webhook's live structural analysis: those are pure
    graph algorithms over live events, not model scoring, and they produce no metric that
    appears in the reported results. Everything that yields a *reported number* is banned.
    """
    banned = {"lightgbm", "sklearn", "xgboost", "torch", "scipy"}
    offending = {m.split(".")[0] for m in imported_modules(path)} & banned
    assert not offending, (
        f"{path.name} imports {sorted(offending)}; the web layer must render precomputed "
        "results, not compute them."
    )


@pytest.mark.parametrize("path", app_source_files(), ids=lambda p: p.name)
def test_app_layer_never_imports_the_evaluation_modules(path: Path):
    """Specifically: nothing that produces a reported metric."""
    forbidden = {"core.evaluate", "core.model", "core.calibration", "core.ring_evidence"}
    offending = imported_modules(path) & forbidden
    assert not offending, f"{path.name} imports {sorted(offending)}"


def test_health_reports_the_results_commit(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["results_commit"]


def test_dashboard_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_dashboard_contains_every_section(client):
    html = client.get("/").text
    for heading in (
        "Headline result",
        "The reconciliation",
        "Operating points",
        "Calibration",
        "Flagged clusters",
    ):
        assert heading in html


def test_no_unrendered_template_syntax(client):
    """A template typo can silently ship '{{ ring.z_score }}' as literal text."""
    html = client.get("/").text
    assert "{{" not in html
    assert "{%" not in html


def test_dashboard_leads_with_the_negative_result(client):
    """Ordering is the argument: the negative finding must precede the graph evidence."""
    html = client.get("/").text
    assert html.index("Headline result") < html.index("The reconciliation")
    assert "does not work" in html


def test_dashboard_shows_the_harm_verdict(client):
    """k-core is reported as measurably harmful, not quietly omitted."""
    assert "significantly worse" in client.get("/").text


def test_dashboard_carries_the_detection_only_statement(client):
    html = client.get("/").text
    assert "Detection-only" in html
    assert "does not generate" in html


def test_dashboard_carries_the_brier_caveat(client):
    """The caveat must travel with the number wherever it is displayed."""
    html = client.get("/").text
    assert "conflates calibration with discrimination" in html


def test_dashboard_marks_provenance_on_every_cluster(client):
    """The determinism boundary must be visible, not merely true."""
    html = client.get("/").text
    results = load_results()
    n = len(results["clusters"])
    assert html.count("computed by core/") >= n
    assert html.count("written by the LLM") >= n


def test_dashboard_does_not_claim_a_ring_count(client):
    """IEEE-CIS has no ring labels, so an affirmative 'N rings caught' would be fabricated.

    A bare substring check for "rings caught" is wrong here and was the first version of
    this test: the page contains that phrase inside the sentence *disclaiming* the claim.
    What must be absent is a quantified assertion — a number followed by a detection verb.
    """
    html = client.get("/").text.lower()

    affirmative = re.compile(r"\d[\d,]*\s+(fraud\s+)?rings?\s+(caught|detected|found|identified)")
    match = affirmative.search(html)
    assert match is None, f"page asserts a ring count: {match.group()!r}"

    # And the disclaimer itself must actually be present.
    assert "not ring labels" in html or "never claims" in html


def test_api_results_matches_the_artifact_on_disk(client):
    """The page and its source must agree, so a reviewer can check one against the other."""
    served = client.get("/api/results").json()
    on_disk = json.loads(RESULTS_PATH.read_text())
    assert served == on_disk


def test_missing_artifact_degrades_honestly(tmp_path):
    """A missing export must say so, not render a page of blanks or zeroes."""
    with pytest.raises(ResultsUnavailable) as excinfo:
        load_results(tmp_path / "nope.json")
    assert "export_results" in str(excinfo.value)
