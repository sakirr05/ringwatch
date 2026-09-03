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
def test_app_modules_do_not_directly_import_modelling_libraries(path: Path):
    """No module under app/ imports a modelling library directly.

    SCOPE, STATED PRECISELY. This checks *direct* imports only. `app/main.py` does reach
    LightGBM transitively, through `core.demo_score`, in the webhook's background task —
    that is deliberate and is the whole point of the demonstration-scoring track. So this
    test does NOT prove "the app cannot compute anything"; an earlier version of it was
    named as though it did, which was an overclaim of exactly the kind this project has had
    to correct elsewhere.

    What it does prove: no request handler pulls a model into its own module scope, so the
    web layer cannot casually acquire the ability to produce a reported metric. The
    stronger guarantee — that the *dashboard rendering path* computes nothing — is asserted
    separately in `test_dashboard_render_path_touches_no_computation`.
    """
    banned = {"lightgbm", "sklearn", "xgboost", "torch", "scipy"}
    offending = {m.split(".")[0] for m in imported_modules(path)} & banned
    assert not offending, (
        f"{path.name} directly imports {sorted(offending)}; the web layer must render "
        "precomputed results, not compute them."
    )


def test_dashboard_render_path_touches_no_computation():
    """The page that displays results must have no path to producing one.

    This is the guarantee that actually matters, and it is narrow enough to be true:
    `app/results.py` is the dashboard's entire relationship with the analysis, and it reads
    a JSON file. It imports nothing from `core/`, transitively or otherwise.
    """
    results_module = APP_DIR / "results.py"
    imports = imported_modules(results_module)

    reachable_core = {m for m in imports if m == "core" or m.startswith("core.")}
    assert not reachable_core, (
        f"app/results.py imports {sorted(reachable_core)}; the dashboard's data path must "
        "read the committed artifact and nothing else."
    )

    banned = {"lightgbm", "sklearn", "xgboost", "torch", "scipy", "numpy", "pandas"}
    assert not ({m.split(".")[0] for m in imports} & banned)


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


# --------------------------------------------------------------------------
# SAR workbench and bi-directional provenance
# --------------------------------------------------------------------------


def test_sar_uses_fiu_ind_category_structure(client):
    """The draft organises evidence on the standard headings an analyst works from."""
    html = client.get("/").text
    for heading in ("Identity of client", "Nature of transactions", "Activity in accounts"):
        assert heading in html


def test_sar_is_labelled_as_a_draft_and_not_a_filing(client):
    """A Suspicious Transaction Report is a statutory PMLA filing.

    This page shows a working draft built from an unvalidated model over clusters with no
    ring-level ground truth. It must say so unmissably, or it becomes a document shaped
    like a regulatory record while containing model output nobody has verified.
    """
    # Whitespace-normalised: the disclaimer wraps across source lines, so a naive
    # substring check fails on text that is present and correct.
    html = re.sub(r"\s+", " ", client.get("/").text)
    assert "working draft" in html.lower()
    assert "not a filing" in html.lower()
    assert "not a Suspicious Transaction Report" in html
    assert "not suitable for regulatory submission" in html


def test_sar_does_not_fabricate_filing_identifiers(client):
    """No reference numbers, filing IDs or officer names that would look official."""
    html = client.get("/").text.lower()
    for invented in ("str no", "str ref", "filing reference", "reporting officer",
                     "acknowledgement number", "fiu ack"):
        assert invented not in html


def test_every_entity_chip_has_a_matching_graph_node(client):
    """Bi-directional highlighting only works if both ends exist.

    A chip pointing at a node id that is not in the SVG would hover into silence.
    """
    # Attributes are split across source lines in the template, so the pattern must be
    # whitespace-tolerant. An earlier version was not and reported 0 nodes against 80
    # chips -- a test failure describing a page that was entirely correct.
    html = re.sub(r"\s+", " ", client.get("/").text)
    chips = set(re.findall(r'class="sar-entity" data-target="([^"]+)"', html))
    nodes = set(re.findall(r'class="gn gn-entity"[^>]*?data-id="([^"]+)"', html))
    assert chips, "no entity chips rendered"
    assert nodes, "no entity nodes rendered"
    assert chips <= nodes, f"chips with no node: {sorted(chips - nodes)}"


def test_graph_nodes_carry_stable_ids_for_the_reverse_direction(client):
    """Hovering a node must be able to find its text, which needs an id and a data-id."""
    html = re.sub(r"\s+", " ", client.get("/").text)
    assert re.search(r'<circle class="gn gn-entity" id="node-\d+-e\d+"', html)
    assert "data-entity=" in html


def test_provenance_script_is_bidirectional_and_defensive(client):
    """Both directions wired, and null-guarded rather than assuming the DOM is complete."""
    html = client.get("/").text
    assert "direction 1: report text -> graph" in html
    assert "direction 2: graph -> report text" in html
    # Guards for the truncated-subgraph case.
    assert "if (!a || !b) return;" in html
    assert "if (!id) return;" in html


def test_no_external_javascript_or_css_is_loaded(client):
    """Vanilla only: no CDN, no framework, nothing for the free tier to fetch."""
    html = client.get("/").text
    assert "<script src=" not in html
    assert "cdn." not in html
    for library in ("d3.", "jquery", "react", "vue"):
        assert library not in html.lower()


def test_highlight_state_is_shared_between_text_and_svg(client):
    """One class drives both sides, so they cannot drift out of visual sync."""
    html = client.get("/").text
    assert ".highlight-provenance" in html
    assert "circle.gn.highlight-provenance" in html
    assert 'var HL = "highlight-provenance"' in html


# --------------------------------------------------------------------------
# report numbering and print output
# --------------------------------------------------------------------------


def test_sections_are_numbered_by_css_counters(client):
    """Numbering is generated, not hand-written.

    Hand-numbered sections drift the moment one is added or reordered; counters renumber
    themselves. Asserted on the mechanism rather than on rendered digits, since the numbers
    only exist at paint time.
    """
    html = client.get("/").text
    assert "counter-reset: sec" in html
    assert 'content: counter(sec) ". "' in html
    # Fields carry section.field numbering (1.1, 1.2, ...).
    assert 'counter(sec) "." counter(field)' in html


def test_no_hand_written_section_prefixes_remain(client):
    """Leftover 'A ·' next to a generated '1.' would render as '1. A · Identity'."""
    html = client.get("/").text
    for stale in ("A · Identity", "B · Nature", "C · Activity"):
        assert stale not in html


def test_print_stylesheet_exists(client):
    html = client.get("/").text
    assert "@media print" in html
    assert "@page" in html


def test_print_hides_analysis_but_keeps_the_drafts(client):
    """Paper should carry the reports, not the model-evaluation sections around them."""
    html = client.get("/").text
    print_css = html[html.index("@media print"):]
    # The analysis chrome is hidden...
    assert ".masthead" in print_css and "display: none" in print_css
    # ...and collapsed drafts are forced open, or printing would emit blank pages.
    assert "details > .cbody { display: block !important" in print_css


def test_the_disclaimer_survives_printing(client):
    """The single most important thing to get right here.

    A printed page separated from its banner is a document that looks like a regulatory
    report with no indication it is neither reviewed nor a filing. So the disclaimer is
    made MORE prominent on paper, and repeated at the top of every printed draft.
    """
    html = client.get("/").text
    print_css = html[html.index("@media print"):]

    # It is never hidden in print.
    assert ".sar-warn { border: 1.5pt solid #000 !important" in print_css
    assert ".print-only { display: block !important; }" in print_css

    # And a per-draft print header repeats the essential statement.
    results = load_results()
    assert html.count("NOT a Suspicious Transaction Report") >= len(results["clusters"])


def test_each_printed_draft_starts_on_its_own_page(client):
    html = client.get("/").text
    print_css = html[html.index("@media print"):]
    assert "page-break-before: always" in print_css
    assert "page-break-inside: avoid" in print_css


# --------------------------------------------------------------------------
# palette contrast
# --------------------------------------------------------------------------


def _relative_luminance(hex_colour: str) -> float:
    raw = hex_colour.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    a, b = _relative_luminance(foreground), _relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def test_light_palette_meets_wcag_aa():
    """The palette was deliberately lightened; this stops it going too far.

    Body text moved from roughly 15:1 to 9:1 against its ground, which reads much softer
    and is still about twice the WCAG AA floor. Four colours failed on the first pass at
    those lighter values — the label grey, and the three tag colours — and were solved back
    up rather than eyeballed. Pinning the ratios means a later "make it a bit lighter
    still" cannot quietly cross the accessibility line.

    Values are duplicated from the template on purpose: if someone edits the CSS without
    updating this list, the two drift and the mismatch test below fails.
    """
    pairs = [
        ("body text on page", "#3a4560", "#fafcfe", 4.5),
        ("body text on panel", "#3a4560", "#ffffff", 4.5),
        ("muted text", "#6b7691", "#ffffff", 4.5),
        ("faint labels", "#8890a2", "#ffffff", 3.0),
        ("links", "#2f74c8", "#ffffff", 4.5),
        ("masthead heading", "#2b3a5e", "#e9f1fb", 4.5),
        ("masthead tagline", "#5a6885", "#e9f1fb", 4.5),
        ("deterministic tag", "#287b5d", "#f0faf5", 4.5),
        ("narrative tag", "#506da2", "#f4f7fd", 4.5),
        ("warning tag", "#a1574c", "#fdf4f2", 4.5),
    ]
    failures = [
        f"{name}: {contrast_ratio(fg, bg):.2f}:1 < {minimum}"
        for name, fg, bg, minimum in pairs
        if contrast_ratio(fg, bg) < minimum
    ]
    assert not failures, "contrast below WCAG AA -- " + "; ".join(failures)


def test_palette_tokens_in_the_template_match_the_tested_values(client):
    """Guard against the CSS and this test drifting apart."""
    html = client.get("/").text
    for token, value in (
        ("--ink", "#3a4560"),
        ("--muted", "#6b7691"),
        ("--faint", "#8890a2"),
        ("--det", "#287b5d"),
        ("--ai", "#506da2"),
        ("--warn", "#a1574c"),
    ):
        assert f"{token}: {value};" in html, f"{token} changed without updating the test"
