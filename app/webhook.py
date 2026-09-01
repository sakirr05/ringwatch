"""Razorpay webhook signature verification.

THE ONE THING THAT MATTERS HERE
-------------------------------
The signature is an HMAC over the **exact bytes Razorpay sent**. It must be computed on the
raw request body, before anything parses it.

This is the most common integration bug in the category, and it is worth being precise
about why. Parsing JSON and re-serialising it produces different bytes for the same
document: key order changes, whitespace is normalised, `1.00` becomes `1.0`. The HMAC of
those bytes will not match, and the failure presents as *intermittent* verification errors
on payloads nobody tampered with — which looks like a problem at Razorpay's end and is not.
`tests/test_webhook.py::test_reserialized_body_fails_verification` reproduces exactly this
and asserts the failure, so the constraint is documented in a form that cannot rot.

Comparison uses `hmac.compare_digest`. A plain `==` on a signature leaks, through timing,
how many leading bytes were correct, which is enough to forge a signature given patience.
"""

from __future__ import annotations

import hashlib
import hmac
import os


class SignatureError(Exception):
    """The request is not authentic. Always a 4XX — retrying it unchanged cannot help."""


def webhook_secret() -> str:
    """Test-mode secret from the environment. Never hardcoded, never committed."""
    return os.environ.get("RAZORPAY_WEBHOOK_SECRET", "").strip()


def expected_signature(raw_body: bytes, secret: str) -> str:
    """HMAC-SHA256 over the raw body bytes, hex-encoded — Razorpay's scheme."""
    if not isinstance(raw_body, (bytes, bytearray)):
        raise TypeError(
            "raw_body must be bytes. Passing a str means it was decoded, and probably "
            "parsed, which defeats the purpose of signing the wire format."
        )
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """Timing-safe comparison of the provided signature against the expected one."""
    if not signature or not secret:
        return False
    return hmac.compare_digest(expected_signature(raw_body, secret), signature)


def verify_or_raise(raw_body: bytes, signature: str, secret: str) -> None:
    if not secret:
        raise SignatureError("RAZORPAY_WEBHOOK_SECRET is not configured")
    if not signature:
        raise SignatureError("missing X-Razorpay-Signature header")
    if not verify_signature(raw_body, signature, secret):
        raise SignatureError("signature does not match request body")


def extract_payment(payload: dict) -> dict | None:
    """Pull the payment entity out of a Razorpay event envelope.

    Shape: {"event": "payment.captured", "payload": {"payment": {"entity": {...}}}}
    """
    container = payload.get("payload")
    if not isinstance(container, dict):
        return None
    payment = container.get("payment")
    if not isinstance(payment, dict):
        return None
    entity = payment.get("entity")
    return entity if isinstance(entity, dict) else None
