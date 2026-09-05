"""Send a correctly-signed test webhook to a running RingWatch instance.

Exists for the demo video, and for anyone who wants to exercise the receiver without a
Razorpay account. It builds a payment event, signs it with HMAC-SHA256 over the exact
bytes it is about to send, and POSTs it — which is the same thing Razorpay does.

    python scripts/demo_webhook.py                 # send one event
    python scripts/demo_webhook.py --replay        # send the SAME event twice, to show
                                                   # idempotency reject the second
    python scripts/demo_webhook.py --tamper        # corrupt the body after signing, to
                                                   # show signature verification reject it
    python scripts/demo_webhook.py --url https://ringwatch.onrender.com

The secret comes from RAZORPAY_WEBHOOK_SECRET and must match the one the server was
started with, or verification correctly fails.

TEST MODE ONLY. Never point this at a live-mode secret.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
import uuid
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Read .env the same way run.py does, so one setup works for both."""
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def build_event(event_id: str) -> dict:
    """A payment.captured envelope shaped like Razorpay's."""
    return {
        "entity": "event",
        "event": "payment.captured",
        "contains": ["payment"],
        "created_at": int(time.time()),
        "payload": {
            "payment": {
                "entity": {
                    "id": f"pay_{uuid.uuid4().hex[:14]}",
                    "entity": "payment",
                    "amount": 250_000,          # paise -> Rs 2,500.00
                    "currency": "INR",
                    "status": "captured",
                    "method": "card",
                    "international": False,
                    "email": "demo@example.com",
                    "contact": "+919999999999",
                    "created_at": int(time.time()),
                    "card": {"network": "Visa", "type": "credit", "last4": "1111"},
                    "notes": {"source": "scripts/demo_webhook.py"},
                }
            }
        },
    }


def post(url: str, body: bytes, signature: str, event_id: str) -> None:
    reply = requests.post(
        f"{url.rstrip('/')}/webhooks/razorpay",
        data=body,                                   # raw bytes, not json=
        headers={
            "Content-Type": "application/json",
            "X-Razorpay-Signature": signature,
            "X-Razorpay-Event-Id": event_id,
        },
        timeout=90,
    )
    print(f"  HTTP {reply.status_code}   {reply.text[:200]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--replay", action="store_true",
                        help="send the same event twice; the second must be a no-op")
    parser.add_argument("--tamper", action="store_true",
                        help="alter the body after signing; must be rejected")
    args = parser.parse_args()

    load_env()
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "").strip()
    if not secret:
        print("RAZORPAY_WEBHOOK_SECRET is not set.\n")
        print("Add it to .env, then restart the server so it picks it up:")
        print('    echo "RAZORPAY_WEBHOOK_SECRET=demo_secret_for_local_testing" >> .env')
        return 1

    event_id = f"evt_{uuid.uuid4().hex[:14]}"
    body = json.dumps(build_event(event_id)).encode()

    # Signed over exactly the bytes that go on the wire. Re-serialising after this point
    # changes those bytes and verification fails -- which is the whole point of the check.
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    if args.tamper:
        body = body.replace(b'"amount": 250000', b'"amount": 999999')
        print(f"Tampered body AFTER signing (amount 2,500 -> 9,999.99)")
        print(f"  event id : {event_id}")
        post(args.url, body, signature, event_id)
        print("\n  Expected 401: the signature covers bytes that no longer match.")
        return 0

    print(f"Signed event  {event_id}")
    print(f"  amount   : Rs 2,500.00 (250000 paise)")
    print(f"  signature: {signature[:32]}...")
    post(args.url, body, signature, event_id)

    if args.replay:
        print("\nReplaying the identical event (same id, same signature):")
        post(args.url, body, signature, event_id)
        print("\n  The second is acknowledged but does no work -- event_id is a PRIMARY")
        print("  KEY, so the insert collides and the duplicate is detected by the")
        print("  database rather than by a check-then-insert race.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
