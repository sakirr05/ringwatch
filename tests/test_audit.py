"""Tests for the human approval gate and the orchestrator's audit trail.

The property under test is mostly a *negative* one: approving a disposition must change
nothing except a database row. Several tests below exist only to fail if someone later adds
an execution path to this route, which is exactly the change that would quietly turn a
recommender into a decider.

The trail is also tested as append-only. An audit log that can be updated in place cannot
answer the question an audit asks — what did the reviewer actually see at the time?
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import store
from app.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "docs" / "results.json"

DRAFT = {
    "case_id": "CASE-001",
    "recommendation": "escalate",
    "confidence": "medium",
    "rationale": "Ambiguous shared infrastructure.",
    "key_factors": ["shared attributes"],
}
EVIDENCE = {"cluster_id": 1, "entity_count": 4, "core_number": 2}


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("RINGWATCH_DB", str(tmp_path / "audit.db"))
    store.init_db()
    return tmp_path


@pytest.fixture
def client(db) -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------
# the trail
# --------------------------------------------------------------------------


def test_a_decision_records_what_was_drafted_and_what_was_seen(db):
    """Outcome alone is not auditable. The inputs have to be in the row."""
    entry = store.record_decision("CASE-001", "approved", "analyst-a", DRAFT, EVIDENCE)
    assert entry.drafted == DRAFT
    assert entry.evidence_seen == EVIDENCE
    assert entry.decision == "approved"
    assert entry.reviewer == "analyst-a"


def test_the_trail_is_append_only(db):
    """A reviewer changing their mind adds a row; it must not overwrite the first."""
    store.record_decision("CASE-001", "approved", "analyst-a", DRAFT, EVIDENCE)
    store.record_decision("CASE-001", "rejected", "analyst-b", DRAFT, EVIDENCE)

    trail = store.audit_trail(case_id="CASE-001")
    assert len(trail) == 2
    assert [e.decision for e in trail] == ["rejected", "approved"]  # newest first
    assert store.latest_decisions()["CASE-001"].decision == "rejected"


def test_the_store_exposes_no_way_to_edit_or_delete_a_decision():
    """Structural, not conventional: no UPDATE or DELETE against the table exists."""
    source = (REPO_ROOT / "app" / "store.py").read_text().upper()
    assert "UPDATE DISPOSITIONS" not in source
    assert "DELETE FROM DISPOSITIONS" not in source


def test_an_unknown_decision_verb_is_refused(db):
    with pytest.raises(ValueError, match="must be one of"):
        store.record_decision("CASE-001", "auto_blocked", "analyst", DRAFT, EVIDENCE)


def test_an_empty_case_id_is_refused(db):
    with pytest.raises(ValueError, match="case_id"):
        store.record_decision("", "approved", "analyst", DRAFT, EVIDENCE)


def test_an_unnamed_reviewer_is_recorded_as_unattributed(db):
    """Anonymous is allowed; silently blank is not — the row must say which."""
    entry = store.record_decision("CASE-001", "approved", "", DRAFT, EVIDENCE)
    assert entry.reviewer == "unattributed"


def test_the_trail_can_be_narrowed_to_one_case(db):
    store.record_decision("CASE-001", "approved", "a", DRAFT, EVIDENCE)
    store.record_decision("CASE-002", "rejected", "a", DRAFT, EVIDENCE)
    assert len(store.audit_trail(case_id="CASE-001")) == 1
    assert len(store.audit_trail()) == 2


def test_an_empty_trail_is_empty_not_an_error(db):
    assert store.audit_trail() == []
    assert store.latest_decisions() == {}


# --------------------------------------------------------------------------
# the route: records, never acts
# --------------------------------------------------------------------------


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_approving_records_a_row_and_says_nothing_was_applied(client):
    from app.results import load_results

    clusters = load_results().get("clusters", [])
    case_id = next(
        (c["disposition"]["case_id"] for c in clusters if c.get("disposition")), None
    )
    if case_id is None:
        pytest.skip("results.json predates the orchestrator")

    reply = client.post(
        f"/api/dispositions/{case_id}/decision",
        json={"decision": "approved", "reviewer": "analyst-a", "note": "agreed"},
    )
    assert reply.status_code == 200
    body = reply.json()
    assert body["recorded"] is True
    assert body["applied"] is False
    assert "nothing was executed" in body["effect"]

    trail = client.get("/api/audit", params={"case_id": case_id}).json()
    assert trail["count"] == 1
    assert trail["entries"][0]["decision"] == "approved"
    # The recorded draft came from the artifact, not from the request body.
    assert trail["entries"][0]["drafted"]["case_id"] == case_id
    assert trail["entries"][0]["evidence_seen"]


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_a_client_cannot_rewrite_what_the_model_was_recorded_as_saying(client):
    """The draft is read from the committed artifact, so a forged body is ignored."""
    from app.results import load_results

    clusters = load_results().get("clusters", [])
    case = next((c for c in clusters if c.get("disposition")), None)
    if case is None:
        pytest.skip("results.json predates the orchestrator")
    case_id = case["disposition"]["case_id"]

    client.post(
        f"/api/dispositions/{case_id}/decision",
        json={
            "decision": "approved",
            "reviewer": "attacker",
            "drafted": {"recommendation": "confirm", "rationale": "forged"},
            "evidence_seen": {"cluster_id": 999},
        },
    )
    recorded = client.get("/api/audit", params={"case_id": case_id}).json()["entries"][0]
    assert recorded["drafted"] == case["disposition"]
    assert recorded["evidence_seen"] == case["evidence"]


def test_an_unknown_case_is_rejected(client):
    reply = client.post(
        "/api/dispositions/CASE-999/decision", json={"decision": "approved"}
    )
    assert reply.status_code in (404, 503)  # 503 only if the artifact is missing


def test_an_unknown_decision_verb_is_rejected_by_the_route(client):
    reply = client.post(
        "/api/dispositions/CASE-001/decision", json={"decision": "block_the_card"}
    )
    assert reply.status_code == 400
    assert "must be one of" in reply.json()["error"]


def test_a_non_json_body_is_a_client_error_not_a_crash(client):
    reply = client.post("/api/dispositions/CASE-001/decision", content=b"not json")
    assert reply.status_code == 400


def test_a_non_object_body_is_rejected(client):
    reply = client.post("/api/dispositions/CASE-001/decision", json=["approved"])
    assert reply.status_code == 400


# --------------------------------------------------------------------------
# the gate cannot become an action
# --------------------------------------------------------------------------


def test_the_decision_route_makes_no_outbound_call():
    """No HTTP client, no notification, no payment API is reachable from app/main.py.

    Pinned as a test because 'approve' is precisely the button someone would later wire
    to a real block. If that day comes, this fails first and the caveat gets updated
    rather than quietly becoming false.
    """
    tree = ast.parse((REPO_ROOT / "app" / "main.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    for outbound in ("requests", "httpx", "urllib", "http", "smtplib", "razorpay", "boto3"):
        assert outbound not in imported, f"app/main.py can now reach {outbound}"


# --------------------------------------------------------------------------
# the gate on the page
# --------------------------------------------------------------------------


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_every_drafted_disposition_is_rendered_behind_a_gate(client):
    import json

    clusters = json.loads(RESULTS_PATH.read_text()).get("clusters", [])
    drafted = [c for c in clusters if c.get("disposition")]
    if not drafted:
        pytest.skip("results.json predates the orchestrator")

    html = client.get("/").text
    assert html.count('class="gate"') == len(drafted)
    # A recommendation must never appear on the page without its gate.
    assert html.count("Approve recommendation") == len(drafted)


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_the_page_says_the_gate_executes_nothing(client):
    html = client.get("/").text
    if 'class="gate"' not in html:
        pytest.skip("results.json predates the orchestrator")
    assert "executes nothing" in html
    assert "no card is blocked" in html.lower()
    assert "detection-only" in html.lower()


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_the_recommendation_is_labelled_a_draft_not_a_decision(client):
    """A reviewer must not be able to mistake the draft for something already done."""
    html = client.get("/").text
    if 'class="gate"' not in html:
        pytest.skip("results.json predates the orchestrator")
    assert "Draft" in html and "not applied" in html
    assert "Recommends:" in html
    for decided in ("Blocked", "Action taken", "Automatically confirmed"):
        assert decided not in html


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="docs/results.json not generated")
def test_the_gate_shows_the_findings_that_argue_against_acting(client):
    """Showing only the incriminating half would make the gate a rubber stamp."""
    html = client.get("/").text
    if 'class="gate"' not in html:
        pytest.skip("results.json predates the orchestrator")
    assert "findings arguing against" in html.lower()
    assert "core/investigation.py" in html


def test_the_gate_script_sends_no_draft_or_evidence():
    """The client posts a verdict and a name. It cannot restate what the model said."""
    html = (REPO_ROOT / "app" / "templates" / "dashboard.html").read_text()
    start = html.index("HUMAN APPROVAL GATE", html.index("<script>"))
    script = html[start : html.index("</script>", start)]
    assert "JSON.stringify" in script
    body = script[script.index("JSON.stringify") : script.index("JSON.stringify") + 400]
    assert "decision" in body and "reviewer" in body
    for forbidden in ("drafted", "evidence_seen", "rationale", "key_factors"):
        assert forbidden not in body


def test_the_artifact_marks_every_disposition_as_requiring_approval():
    if not RESULTS_PATH.exists():
        pytest.skip("docs/results.json not generated")
    import json

    clusters = json.loads(RESULTS_PATH.read_text()).get("clusters", [])
    dispositions = [c["disposition"] for c in clusters if c.get("disposition")]
    if not dispositions:
        pytest.skip("results.json predates the orchestrator")
    assert all(d["requires_human_approval"] for d in dispositions)
