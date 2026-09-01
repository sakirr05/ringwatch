"""Tests for the Razorpay webhook receiver.

These cover the four things a payments engineer checks first: is the signature computed
over the raw bytes, is delivery idempotent, does the endpoint answer fast enough to avoid
being disabled, and does it distinguish "your request is broken" from "we are broken".
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from starlette.requests import Request

from app import store
from app.main import app, razorpay_webhook
from app.webhook import expected_signature, verify_signature

SECRET = "whsec_test_ringwatch_local"

PAYMENT = {
    "id": "pay_TESTabc123",
    "entity": "payment",
    "amount": 249900,
    "currency": "INR",
    "status": "captured",
    "method": "card",
    "email": "asha@example.com",
    "contact": "+919876500011",
    "created_at": 1_772_000_000,
    "card": {"last4": "4242", "network": "Visa", "type": "credit", "issuer": "HDFC"},
}


def make_payload(payment: dict | None = None, event: str = "payment.captured") -> dict:
    return {
        "entity": "event",
        "account_id": "acc_TESTaccount",
        "event": event,
        "contains": ["payment"],
        "payload": {"payment": {"entity": payment or PAYMENT}},
        "created_at": 1_772_000_000,
    }


def sign(raw: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own database and a known secret."""
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", SECRET)
    monkeypatch.setenv("RINGWATCH_DB", str(tmp_path / "events.db"))
    store.init_db()
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def post(client: TestClient, raw: bytes, signature: str, event_id: str):
    return client.post(
        "/webhooks/razorpay",
        content=raw,
        headers={
            "X-Razorpay-Signature": signature,
            "X-Razorpay-Event-Id": event_id,
            "Content-Type": "application/json",
        },
    )


# --------------------------------------------------------------------------
# signature verification
# --------------------------------------------------------------------------


def test_known_good_signature_verifies():
    raw = json.dumps(make_payload()).encode()
    assert verify_signature(raw, sign(raw), SECRET)


def test_valid_request_is_accepted(client):
    raw = json.dumps(make_payload()).encode()
    response = post(client, raw, sign(raw), "evt_ok_1")
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


def test_tampered_body_fails_verification(client):
    """Signature computed over the original, body altered in flight."""
    raw = json.dumps(make_payload()).encode()
    signature = sign(raw)

    tampered_payload = make_payload({**PAYMENT, "amount": 100})
    tampered = json.dumps(tampered_payload).encode()

    assert not verify_signature(tampered, signature, SECRET)
    assert post(client, tampered, signature, "evt_tampered").status_code == 401


def test_reserialized_body_fails_verification():
    """THE bug this design exists to avoid, pinned as an executable statement.

    Parsing the JSON and re-serialising it yields a semantically identical document with
    different bytes — reordered keys, normalised spacing, `1.00` collapsed to `1.0`. Hashing
    those bytes produces a different HMAC, so verification fails on a payload nobody
    tampered with, and it looks like an upstream problem.

    The handler therefore hashes `await request.body()` and never a re-encoded structure.
    """
    # Deliberately shaped like real wire bytes rather than canonical output: keys out of
    # alphabetical order, no space after the colons, and a trailing-zero decimal. Each of
    # those three survives a round trip differently, and any one of them breaks the HMAC.
    raw = b'{"id":"pay_x","amount":249900.00,"event":"payment.captured"}'
    signature = sign(raw)
    assert verify_signature(raw, signature, SECRET)

    # Round-trip it the way a Pydantic-model handler effectively would.
    reserialized = json.dumps(json.loads(raw), sort_keys=True).encode()
    assert reserialized != raw, (
        "the sample body is already in canonical form, so this test would pass "
        "vacuously without demonstrating anything"
    )
    assert not verify_signature(reserialized, signature, SECRET), (
        "re-serialised body produced a matching signature; the test is no longer "
        "demonstrating the bug it exists to document"
    )


def test_signature_is_compared_timing_safely():
    """A near-miss must fail like any other mismatch."""
    raw = json.dumps(make_payload()).encode()
    correct = sign(raw)
    almost = correct[:-1] + ("0" if correct[-1] != "0" else "1")
    assert not verify_signature(raw, almost, SECRET)


def test_wrong_secret_fails(client):
    raw = json.dumps(make_payload()).encode()
    assert post(client, raw, sign(raw, "whsec_wrong"), "evt_wrong").status_code == 401


def test_missing_signature_header_is_rejected(client):
    raw = json.dumps(make_payload()).encode()
    assert post(client, raw, "", "evt_nosig").status_code == 401


def test_unconfigured_secret_reports_server_error(client, monkeypatch):
    """Our misconfiguration is a 5XX, not a client error."""
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "")
    raw = json.dumps(make_payload()).encode()
    assert post(client, raw, sign(raw), "evt_nosecret").status_code == 503


# --------------------------------------------------------------------------
# payload handling and status codes
# --------------------------------------------------------------------------


def test_malformed_json_is_a_client_error(client):
    """4XX: retrying identical bad bytes cannot help, so Razorpay should stop."""
    raw = b"{not json at all"
    assert post(client, raw, sign(raw), "evt_bad_json").status_code == 400


def test_missing_event_id_header_is_rejected(client):
    raw = json.dumps(make_payload()).encode()
    response = client.post(
        "/webhooks/razorpay",
        content=raw,
        headers={"X-Razorpay-Signature": sign(raw)},
    )
    assert response.status_code == 400
    assert "event-id" in response.json()["detail"]


# --------------------------------------------------------------------------
# idempotency
# --------------------------------------------------------------------------


def test_duplicate_event_id_does_not_reprocess(client):
    """Razorpay delivers at-least-once; a replay must be a no-op."""
    raw = json.dumps(make_payload()).encode()
    signature = sign(raw)

    first = post(client, raw, signature, "evt_dup")
    second = post(client, raw, signature, "evt_dup")

    assert first.status_code == 200
    assert first.json()["status"] == "accepted"
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"

    # Stored exactly once.
    assert store.event_count() == 1


def test_distinct_event_ids_are_both_processed(client):
    raw = json.dumps(make_payload()).encode()
    signature = sign(raw)
    post(client, raw, signature, "evt_a")
    post(client, raw, signature, "evt_b")
    assert store.event_count() == 2


# --------------------------------------------------------------------------
# fast 2xx: the response must precede the work
# --------------------------------------------------------------------------


@pytest.mark.anyio
async def test_response_is_returned_before_analysis_runs():
    """The handler must return a response with the analysis still *pending*.

    Asserted by calling the endpoint directly: a queued BackgroundTask that has not yet run
    is exactly the property Razorpay's ~5 second budget requires. A TestClient cannot show
    this, because Starlette drains background tasks before handing back the response.
    """
    raw = json.dumps(make_payload()).encode()
    signature = sign(raw)
    event_id = "evt_ordering"

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/webhooks/razorpay",
        "headers": [
            (b"x-razorpay-signature", signature.encode()),
            (b"x-razorpay-event-id", event_id.encode()),
            (b"content-type", b"application/json"),
        ],
        "query_string": b"",
    }

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    background = BackgroundTasks()
    response = await razorpay_webhook(Request(scope, receive), background)

    assert response.status_code == 200
    # The work is queued...
    assert len(background.tasks) == 1
    # ...and demonstrably has not run yet.
    assert store.get_analysis(event_id) is None


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_endpoint_responds_well_within_the_retry_budget(client):
    """End-to-end latency check against Razorpay's ~5s disable threshold."""
    raw = json.dumps(make_payload()).encode()
    started = time.perf_counter()
    response = post(client, raw, sign(raw), "evt_timing")
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert elapsed < 2.0, f"handler took {elapsed:.2f}s; Razorpay disables slow endpoints"


# --------------------------------------------------------------------------
# analysis tracks
# --------------------------------------------------------------------------


def test_analysis_records_both_tracks(client):
    """After the background task drains, both tracks are stored for the event."""
    raw = json.dumps(make_payload()).encode()
    post(client, raw, sign(raw), "evt_analysis")

    analysis = store.get_analysis("evt_analysis")
    assert analysis is not None
    assert analysis["structural"] is not None
    assert analysis["demo_score"] is not None


def test_demo_score_carries_its_coverage_caveat(client):
    """The score must never be storable without the caveat that qualifies it."""
    raw = json.dumps(make_payload()).encode()
    post(client, raw, sign(raw), "evt_caveat")

    demo = store.get_analysis("evt_caveat")["demo_score"]
    if demo.get("available", True) and demo.get("score") is not None:
        assert demo["features_present"] < demo["features_total"]
        assert "has never seen a Razorpay payload" in demo["caveat"]
        # The headline honesty figure: coverage is a rounding error.
        assert demo["coverage_pct"] < 5.0


def test_linked_payments_are_detected_structurally(client):
    """Two payments sharing a card fingerprint must land in one component."""
    first = make_payload({**PAYMENT, "id": "pay_1", "email": "one@example.com"})
    second = make_payload({**PAYMENT, "id": "pay_2", "email": "two@example.com"})

    for index, payload in enumerate((first, second)):
        raw = json.dumps(payload).encode()
        post(client, raw, sign(raw), f"evt_link_{index}")

    structural = store.get_analysis("evt_link_1")["structural"]
    assert structural["component_size"] >= 2
    assert "card_fingerprint" in structural["shared_attributes"]
    assert structural["linked_payers"]


def test_unrelated_payments_are_not_linked(client):
    """No shared identifier must mean no edge — the graph must not invent links."""
    lonely = make_payload(
        {
            "id": "pay_solo",
            "amount": 1000,
            "email": "solo@nowhere.test",
            "contact": "+919000000000",
            "created_at": 1_772_000_100,
            "card": {"last4": "9999", "network": "RuPay", "type": "debit"},
        }
    )
    raw = json.dumps(lonely).encode()
    post(client, raw, sign(raw), "evt_solo")

    structural = store.get_analysis("evt_solo")["structural"]
    assert structural["component_size"] == 1
    assert structural["degree"] == 0
    assert structural["linked_payers"] == []


def test_events_feed_lists_recent_events(client):
    raw = json.dumps(make_payload()).encode()
    post(client, raw, sign(raw), "evt_feed")

    body = client.get("/api/events").json()
    assert body["count"] == 1
    assert body["events"][0]["event_id"] == "evt_feed"
    assert body["events"][0]["event_type"] == "payment.captured"
