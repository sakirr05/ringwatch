"""RingWatch web app: read-only dashboard plus the Razorpay webhook receiver.

    uvicorn app.main:app --reload

WHAT THIS LAYER IS ALLOWED TO DO
--------------------------------
Render `docs/results.json`, and receive webhooks. That is all. It does not train, score,
threshold, bootstrap, or recompute an ablation — every figure on the dashboard was computed
locally by `run.py` and frozen into a committed artifact by `scripts/export_results.py`.

This is the same boundary the project already enforces around its LLM layer, applied to its
web layer: one place computes numbers, another displays them, and the display side has no
code path to the computation.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import store
from app.results import ResultsUnavailable, load_results
from app.webhook import SignatureError, extract_payment, verify_or_raise, webhook_secret

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

app = FastAPI(
    title="RingWatch",
    description=(
        "Graph-aware fraud ring detection with a code-enforced AI/determinism boundary. "
        "This service renders precomputed results and receives Razorpay webhooks; it does "
        "not train or score models on request."
    ),
    version="1.0.0",
)

DOCS_DIR = REPO_ROOT / "docs"
if DOCS_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DOCS_DIR)), name="static")


def _fmt_inr(value: float) -> str:
    """Indian digit grouping, matching how the CLI reports rupee figures."""
    try:
        whole = int(round(float(value)))
    except (TypeError, ValueError):
        return "-"
    sign = "-" if whole < 0 else ""
    digits = str(abs(whole))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join([*groups, tail])
    return f"{sign}{digits}"


TEMPLATES.env.filters["inr"] = _fmt_inr


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Ensure the webhook tables exist before the first request."""
    store.init_db()
    yield


app.router.lifespan_context = lifespan


def analyse_event(event_id: str, payment: dict) -> None:
    """Run both analysis tracks. Called AFTER the 200 has been sent.

    Imports are local on purpose: this runs off the request path, and keeping LightGBM out
    of module import time means the web process starts fast and stays up even if the model
    artifact is missing.

    Track 1 (graph structure) is a real computation. Track 2 (model score) is a labelled
    demonstration whose caveat travels with it in the same record. Neither writes to
    `docs/results.json`, so neither can alter a reported metric.
    """
    from core.demo_score import score_payment
    from core.live_entities import analyse, extract_event

    structural = None
    demo = None

    try:
        target = extract_event(event_id, payment)
        if target is not None:
            history = [
                event
                for stored_id, stored_payment in store.all_payments()
                if (event := extract_event(stored_id, stored_payment)) is not None
                and stored_id != event_id
            ]
            structural = analyse(history, target).to_dict()
    except Exception as exc:  # noqa: BLE001 - analysis must never take the service down
        structural = {"error": str(exc)}

    try:
        demo = score_payment(payment).to_dict()
    except Exception as exc:  # noqa: BLE001
        demo = {"available": False, "reason": str(exc)}

    store.record_analysis(event_id, structural, demo)


@app.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request, background: BackgroundTasks) -> JSONResponse:
    """Receive a Razorpay webhook.

    Ordering is deliberate and is the thing to check first when reviewing this:

      1. read the RAW body bytes — never a parsed model, see app/webhook.py
      2. verify the HMAC signature
      3. record the event id (this is what makes replays no-ops)
      4. RETURN 200
      5. only then analyse, in a background task

    Razorpay retries with exponential backoff for 24 hours and disables endpoints that do
    not answer within roughly five seconds, so nothing slow may happen before step 4.
    """
    # 1. Raw bytes. Binding a Pydantic body model here would parse the JSON and any
    #    re-serialisation would change the bytes, breaking the signature.
    raw_body = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")
    event_id = request.headers.get("x-razorpay-event-id", "")

    # 2. Authenticity. 4XX throughout: a retry of the same bytes cannot succeed.
    try:
        verify_or_raise(raw_body, signature, webhook_secret())
    except SignatureError as exc:
        detail = str(exc)
        # A missing server-side secret is our misconfiguration, not a bad request.
        status = 503 if "not configured" in detail else 401
        return JSONResponse({"status": "rejected", "detail": detail}, status_code=status)

    try:
        payload = json.loads(raw_body)
        if not isinstance(payload, dict):
            raise ValueError("payload is not a JSON object")
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(
            {"status": "rejected", "detail": f"malformed payload: {exc}"}, status_code=400
        )

    if not event_id:
        return JSONResponse(
            {"status": "rejected", "detail": "missing x-razorpay-event-id header"},
            status_code=400,
        )

    event_type = str(payload.get("event", "unknown"))

    # 3. Idempotency. Insert-and-detect-collision rather than check-then-insert, which
    #    would race under concurrent redelivery.
    try:
        is_new = store.record_event(event_id, event_type, payload)
    except store.StorageError as exc:
        # 5XX: transient and our fault, so Razorpay SHOULD retry this one.
        return JSONResponse(
            {"status": "error", "detail": f"storage unavailable: {exc}"}, status_code=503
        )

    if not is_new:
        return JSONResponse(
            {"status": "duplicate", "event_id": event_id, "detail": "already processed"}
        )

    # 5. Queue the work. Runs after the response is sent.
    payment = extract_payment(payload)
    if payment is not None and event_type.startswith("payment."):
        background.add_task(analyse_event, event_id, payment)

    # 4. Fast 2xx.
    return JSONResponse(
        {"status": "accepted", "event_id": event_id, "event": event_type}
    )


@app.get("/api/events")
def api_events(limit: int = 25) -> JSONResponse:
    """Recent webhook events and their analysis, for the live feed."""
    events = store.recent_events(limit=limit)
    return JSONResponse(
        {
            "count": store.event_count(),
            "events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "received_at": event.received_at,
                    "analysis": event.analysis,
                }
                for event in events
            ],
        }
    )


@app.get("/health")
def health() -> JSONResponse:
    """Liveness probe for the hosting platform."""
    try:
        results = load_results()
        return JSONResponse(
            {
                "status": "ok",
                "results_commit": results["meta"]["git_commit"],
                "generated_at": results["meta"]["generated_at"],
            }
        )
    except ResultsUnavailable as exc:
        return JSONResponse({"status": "degraded", "detail": str(exc)}, status_code=503)


@app.get("/api/results")
def api_results() -> JSONResponse:
    """The raw artifact, so a reviewer can check the page against its source."""
    try:
        return JSONResponse(load_results())
    except ResultsUnavailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


# ---------------------------------------------------------------------------
# HUMAN APPROVAL GATE
#
# The orchestrator drafts; a person decides. These two routes are the entire decision
# surface, and neither of them acts on anything: approving a disposition writes an audit
# row and nothing else. No card is blocked, no customer is contacted, no external call is
# made. That is not a limitation of the demo -- it is the design. A recommender that can
# also execute is a decider, and this one is deliberately not.
# ---------------------------------------------------------------------------


@app.post("/api/dispositions/{case_id}/decision")
async def decide_disposition(case_id: str, request: Request) -> JSONResponse:
    """Record a human approve/reject on a drafted disposition."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - malformed body is a client error, not a crash
        return JSONResponse({"error": "body must be JSON"}, status_code=400)

    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

    decision = body.get("decision")
    if decision not in store.DECISIONS:
        return JSONResponse(
            {"error": f"decision must be one of {', '.join(store.DECISIONS)}"},
            status_code=400,
        )

    # The draft and the evidence are read from the committed artifact, never from the
    # request. A client cannot rewrite what the model was recorded as having said, or
    # what it was recorded as having seen.
    try:
        results = load_results()
    except ResultsUnavailable as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    cluster = next(
        (
            item
            for item in results.get("clusters", [])
            if (item.get("disposition") or {}).get("case_id") == case_id
        ),
        None,
    )
    if cluster is None:
        return JSONResponse({"error": f"unknown case {case_id}"}, status_code=404)

    note = body.get("note")
    if note is not None and not isinstance(note, str):
        return JSONResponse({"error": "note must be a string"}, status_code=400)

    try:
        entry = store.record_decision(
            case_id=case_id,
            decision=decision,
            reviewer=str(body.get("reviewer") or "unattributed")[:120],
            drafted=cluster["disposition"],
            evidence_seen=cluster["evidence"],
            note=note[:1000] if note else None,
        )
    except store.StorageError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    return JSONResponse(
        {
            "recorded": True,
            "id": entry.id,
            "case_id": entry.case_id,
            "decision": entry.decision,
            "decided_at": entry.decided_at,
            "reviewer": entry.reviewer,
            "note": entry.note,
            # Said explicitly in the response so no client can mistake this for an action.
            "applied": False,
            "effect": "audit record only; nothing was executed",
        }
    )


@app.get("/api/audit")
def api_audit(case_id: str | None = None, limit: int = 50) -> JSONResponse:
    """The orchestrator audit trail: what was drafted, what was seen, who decided."""
    try:
        entries = store.audit_trail(case_id=case_id, limit=max(1, min(limit, 200)))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=503)

    return JSONResponse(
        {
            "count": len(entries),
            "entries": [
                {
                    "id": entry.id,
                    "case_id": entry.case_id,
                    "decided_at": entry.decided_at,
                    "decision": entry.decision,
                    "reviewer": entry.reviewer,
                    "note": entry.note,
                    "drafted": entry.drafted,
                    "evidence_seen": entry.evidence_seen,
                }
                for entry in entries
            ],
        }
    )


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    try:
        results = load_results()
    except ResultsUnavailable as exc:
        return HTMLResponse(
            f"<pre style='font:14px ui-monospace,monospace;padding:2rem'>{exc}</pre>",
            status_code=503,
        )

    ablation = results["ablation"]
    baseline = next(row for row in ablation if row["key"] == "baseline")

    # The live feed is the only part of the page that is not precomputed. It reports
    # events this instance actually received; it never contributes to a reported metric.
    try:
        events = store.recent_events(limit=15)
        event_total = store.event_count()
    except Exception:  # noqa: BLE001 - the feed must never break the results page
        events, event_total = [], 0

    try:
        decisions = {
            case_id: {
                "decision": entry.decision,
                "reviewer": entry.reviewer,
                "decided_at": entry.decided_at,
                "note": entry.note,
            }
            for case_id, entry in store.latest_decisions().items()
        }
    except Exception:  # noqa: BLE001 - same rule: the gate must not break the page
        decisions = {}

    return TEMPLATES.TemplateResponse(
        request,
        "dashboard.html",
        {
            "r": results,
            "ablation": ablation,
            "baseline": baseline,
            "ring": results["ring_evidence"],
            "graph": results["graph"],
            "ops": results["operating_points"],
            "calibration": results["calibration"],
            "clusters": results["clusters"],
            "dataset": results["dataset"],
            "assumptions": results["assumptions"],
            "vw": results.get("value_weighted"),
            "events": events,
            "event_total": event_total,
            "decisions": decisions,
            "webhook_configured": bool(webhook_secret()),
        },
    )
