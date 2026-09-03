"""SQLite persistence for webhook events.

Its main job is **idempotency**. Razorpay delivers at-least-once: the same event genuinely
does arrive more than once, especially if a previous attempt timed out. `event_id` is a
PRIMARY KEY and insertion is the deduplication mechanism — a replay collides, the collision
is detected, and no work is repeated. Checking "does this row exist?" before inserting would
race under concurrent delivery; letting the database enforce uniqueness does not.

On free-tier hosting the disk is ephemeral, so this log does not survive a restart. That is
acceptable for a demonstration and is stated in the README rather than left to be
discovered. No reported metric depends on it.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "webhook_events.db"

_LOCK = threading.Lock()


class StorageError(RuntimeError):
    """Storage is unavailable. Callers should surface 5XX so Razorpay retries."""


@dataclass
class StoredEvent:
    event_id: str
    event_type: str
    received_at: str
    payload: dict
    analysis: dict | None


def db_path() -> Path:
    return Path(os.environ.get("RINGWATCH_DB", str(DEFAULT_DB)))


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _LOCK, _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id    TEXT PRIMARY KEY,
                event_type  TEXT NOT NULL,
                received_at TEXT NOT NULL,
                payload     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS analysis (
                event_id    TEXT PRIMARY KEY,
                analysed_at TEXT NOT NULL,
                structural  TEXT,
                demo_score  TEXT,
                FOREIGN KEY (event_id) REFERENCES events(event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_received
                ON events(received_at DESC);

            -- Audit trail for the investigation orchestrator. Append-only by
            -- convention AND by shape: there is no UPDATE or DELETE anywhere in this
            -- module, so a reviewer's later change to a case adds a row rather than
            -- rewriting one. An audit log you can edit is not an audit log.
            CREATE TABLE IF NOT EXISTS dispositions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id       TEXT NOT NULL,
                decided_at    TEXT NOT NULL,
                decision      TEXT NOT NULL,
                reviewer      TEXT NOT NULL,
                note          TEXT,
                drafted       TEXT NOT NULL,
                evidence_seen TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_dispositions_case
                ON dispositions(case_id, id DESC);
            """
        )


def record_event(event_id: str, event_type: str, payload: dict) -> bool:
    """Insert an event. Returns False if this event_id was already seen.

    The return value is the idempotency signal: False means a replay, and the caller must
    do no further work.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with _LOCK, _connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO events (event_id, event_type, received_at, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (event_id, event_type, now, json.dumps(payload)),
                )
            except sqlite3.IntegrityError:
                return False  # duplicate delivery, already handled
        return True
    except sqlite3.Error as exc:
        raise StorageError(str(exc)) from exc


def record_analysis(event_id: str, structural: dict | None, demo_score: dict | None) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with _LOCK, _connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO analysis "
                "(event_id, analysed_at, structural, demo_score) VALUES (?, ?, ?, ?)",
                (
                    event_id,
                    now,
                    json.dumps(structural) if structural is not None else None,
                    json.dumps(demo_score) if demo_score is not None else None,
                ),
            )
    except sqlite3.Error as exc:
        raise StorageError(str(exc)) from exc


def get_analysis(event_id: str) -> dict | None:
    with _LOCK, _connect() as connection:
        row = connection.execute(
            "SELECT * FROM analysis WHERE event_id = ?", (event_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "event_id": row["event_id"],
        "analysed_at": row["analysed_at"],
        "structural": json.loads(row["structural"]) if row["structural"] else None,
        "demo_score": json.loads(row["demo_score"]) if row["demo_score"] else None,
    }


def recent_events(limit: int = 25) -> list[StoredEvent]:
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            """
            SELECT e.*, a.structural, a.demo_score
            FROM events e LEFT JOIN analysis a ON a.event_id = e.event_id
            ORDER BY e.received_at DESC, e.rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    events = []
    for row in rows:
        analysis = None
        if row["structural"] or row["demo_score"]:
            analysis = {
                "structural": json.loads(row["structural"]) if row["structural"] else None,
                "demo_score": json.loads(row["demo_score"]) if row["demo_score"] else None,
            }
        events.append(
            StoredEvent(
                event_id=row["event_id"],
                event_type=row["event_type"],
                received_at=row["received_at"],
                payload=json.loads(row["payload"]),
                analysis=analysis,
            )
        )
    return events


def all_payments() -> list[tuple[str, dict]]:
    """Every stored payment entity, for rebuilding the live entity graph."""
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            "SELECT event_id, payload FROM events ORDER BY received_at ASC, rowid ASC"
        ).fetchall()

    payments = []
    for row in rows:
        payload = json.loads(row["payload"])
        entity = (
            payload.get("payload", {}).get("payment", {}).get("entity")
            if isinstance(payload.get("payload"), dict)
            else None
        )
        if isinstance(entity, dict):
            payments.append((row["event_id"], entity))
    return payments


def event_count() -> int:
    with _LOCK, _connect() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])


# ---------------------------------------------------------------------------
# ORCHESTRATOR AUDIT TRAIL
#
# What is recorded is deliberately more than the outcome. `drafted` is the full
# recommendation the model produced and `evidence_seen` is the exact case file it was
# handed, so a later reviewer can reconstruct *why* a recommendation was made and judge
# whether the human was right to accept it. Logging only "approved" would make the trail
# useless for exactly the question an audit asks.
#
# Nothing here executes anything. Recording an approval blocks no card, notifies no
# customer, and calls no external service -- the row is the entire effect.
# ---------------------------------------------------------------------------

DECISIONS: tuple[str, ...] = ("approved", "rejected")


@dataclass
class AuditEntry:
    id: int
    case_id: str
    decided_at: str
    decision: str
    reviewer: str
    note: str | None
    drafted: dict
    evidence_seen: dict


def record_decision(
    case_id: str,
    decision: str,
    reviewer: str,
    drafted: dict,
    evidence_seen: dict,
    note: str | None = None,
) -> AuditEntry:
    """Append one human decision on a drafted disposition. Never overwrites."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {', '.join(DECISIONS)}, got {decision!r}")
    if not case_id:
        raise ValueError("case_id is required")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with _LOCK, _connect() as connection:
            cursor = connection.execute(
                "INSERT INTO dispositions "
                "(case_id, decided_at, decision, reviewer, note, drafted, evidence_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    case_id,
                    now,
                    decision,
                    reviewer or "unattributed",
                    note,
                    json.dumps(drafted),
                    json.dumps(evidence_seen),
                ),
            )
            new_id = int(cursor.lastrowid)
    except sqlite3.Error as exc:
        raise StorageError(str(exc)) from exc

    return AuditEntry(
        id=new_id,
        case_id=case_id,
        decided_at=now,
        decision=decision,
        reviewer=reviewer or "unattributed",
        note=note,
        drafted=drafted,
        evidence_seen=evidence_seen,
    )


def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
    return AuditEntry(
        id=int(row["id"]),
        case_id=row["case_id"],
        decided_at=row["decided_at"],
        decision=row["decision"],
        reviewer=row["reviewer"],
        note=row["note"],
        drafted=json.loads(row["drafted"]),
        evidence_seen=json.loads(row["evidence_seen"]),
    )


def audit_trail(case_id: str | None = None, limit: int = 50) -> list[AuditEntry]:
    """The trail, newest first. Optionally narrowed to one case."""
    with _LOCK, _connect() as connection:
        if case_id:
            rows = connection.execute(
                "SELECT * FROM dispositions WHERE case_id = ? ORDER BY id DESC LIMIT ?",
                (case_id, limit),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM dispositions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_row_to_entry(row) for row in rows]


def latest_decisions() -> dict[str, AuditEntry]:
    """The most recent decision per case, for rendering gate state.

    Superseded rows stay in the table; this only picks which one is current.
    """
    with _LOCK, _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM dispositions WHERE id IN "
            "(SELECT MAX(id) FROM dispositions GROUP BY case_id)"
        ).fetchall()
    return {row["case_id"]: _row_to_entry(row) for row in rows}
