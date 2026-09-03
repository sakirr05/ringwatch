"""Strict validation of the language model's JSON output.

Validation is deliberately unforgiving. Every check below rejects rather than repairs,
because a repaired narrative is a narrative partly written by the parser, and an analyst
cannot tell which sentences those were.

The number-provenance check is the important one. A fraud analyst reading "this cluster
moved Rs 4,20,000 across 14 cards" will act on those figures. If the model invented them,
the narrative is worse than useless — it is misleading in a way that looks authoritative.
So every numeric token in the output must already appear in the evidence the model was
handed; anything else means the model is doing arithmetic it is not permitted to do.
"""

from __future__ import annotations

import json

from ai.contract import (
    CONFIDENCE_LEVELS,
    DISPOSITIONS,
    PROBABLE_CAUSES,
    CaseFile,
    ClusterEvidence,
    ClusterNarrative,
    Disposition,
    extract_numbers,
)

MAX_SUMMARY_CHARS = 1200
MAX_ACTION_CHARS = 300

REQUIRED_FIELDS = ("probable_cause", "confidence", "human_summary", "suggested_action")


class ValidationError(Exception):
    """Raised when the model's response cannot be accepted as-is."""


def _strip_code_fence(raw: str) -> str:
    """Models wrap JSON in ```json fences often enough to be worth handling."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def validate(raw: str, evidence: ClusterEvidence) -> ClusterNarrative:
    """Parse and validate one model response. Raises ValidationError on any problem."""
    text = _strip_code_fence(raw)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValidationError(f"expected a JSON object, got {type(payload).__name__}")

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValidationError(f"missing required field(s): {', '.join(missing)}")

    extra = set(payload) - set(REQUIRED_FIELDS)
    if extra:
        raise ValidationError(f"unexpected field(s): {', '.join(sorted(extra))}")

    for field in REQUIRED_FIELDS:
        if not isinstance(payload[field], str):
            raise ValidationError(
                f"field '{field}' must be a string, got "
                f"{type(payload[field]).__name__}"
            )

    if payload["probable_cause"] not in PROBABLE_CAUSES:
        raise ValidationError(
            f"probable_cause '{payload['probable_cause']}' is not one of "
            f"{', '.join(PROBABLE_CAUSES)}"
        )

    if payload["confidence"] not in CONFIDENCE_LEVELS:
        raise ValidationError(
            f"confidence '{payload['confidence']}' is not one of "
            f"{', '.join(CONFIDENCE_LEVELS)}"
        )

    summary = payload["human_summary"].strip()
    action = payload["suggested_action"].strip()

    if not summary:
        raise ValidationError("human_summary is empty")
    if not action:
        raise ValidationError("suggested_action is empty")
    if len(summary) > MAX_SUMMARY_CHARS:
        raise ValidationError(
            f"human_summary is {len(summary)} chars, limit {MAX_SUMMARY_CHARS}"
        )
    if len(action) > MAX_ACTION_CHARS:
        raise ValidationError(
            f"suggested_action is {len(action)} chars, limit {MAX_ACTION_CHARS}"
        )

    _reject_invented_numbers(summary, action, evidence)

    return ClusterNarrative(
        cluster_id=evidence.cluster_id,
        probable_cause=payload["probable_cause"],
        confidence=payload["confidence"],
        human_summary=summary,
        suggested_action=action,
    )


def _reject_invented_numbers(
    summary: str, action: str, evidence: ClusterEvidence
) -> None:
    """The model may quote the evidence's numbers and invent none of its own."""
    allowed = evidence.allowed_numbers()
    used = extract_numbers(summary) | extract_numbers(action)
    invented = sorted(used - allowed)
    if invented:
        raise ValidationError(
            "response contains number(s) absent from the evidence: "
            f"{', '.join(invented)}"
        )


# ---------------------------------------------------------------------------
# DISPOSITION VALIDATION
#
# Same discipline as the narrative validator above, applied to the orchestrator's output.
# The number-provenance guard matters more here, not less: an analyst reading a
# recommendation is deciding whether to act, and an invented figure inside a confident
# rationale is exactly the input that gets rubber-stamped.
# ---------------------------------------------------------------------------

DISPOSITION_FIELDS = ("recommendation", "confidence", "rationale", "key_factors")

MAX_RATIONALE_CHARS = 1500
MAX_FACTOR_CHARS = 160
MAX_FACTORS = 6


def validate_disposition(raw: str, case: CaseFile) -> Disposition:
    """Parse and validate one drafted disposition. Raises ValidationError on any problem."""
    text = _strip_code_fence(raw)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"response is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValidationError(f"expected a JSON object, got {type(payload).__name__}")

    missing = [f for f in DISPOSITION_FIELDS if f not in payload]
    if missing:
        raise ValidationError(f"missing required field(s): {', '.join(missing)}")

    extra = set(payload) - set(DISPOSITION_FIELDS)
    if extra:
        raise ValidationError(f"unexpected field(s): {', '.join(sorted(extra))}")

    if payload["recommendation"] not in DISPOSITIONS:
        raise ValidationError(
            f"recommendation '{payload['recommendation']}' is not one of "
            f"{', '.join(DISPOSITIONS)}"
        )

    if payload["confidence"] not in CONFIDENCE_LEVELS:
        raise ValidationError(
            f"confidence '{payload['confidence']}' is not one of "
            f"{', '.join(CONFIDENCE_LEVELS)}"
        )

    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValidationError("rationale must be a non-empty string")
    rationale = rationale.strip()
    if len(rationale) > MAX_RATIONALE_CHARS:
        raise ValidationError(
            f"rationale is {len(rationale)} chars, limit {MAX_RATIONALE_CHARS}"
        )

    factors = payload["key_factors"]
    if not isinstance(factors, list) or not factors:
        raise ValidationError("key_factors must be a non-empty list")
    if len(factors) > MAX_FACTORS:
        raise ValidationError(f"key_factors has {len(factors)} items, limit {MAX_FACTORS}")
    for factor in factors:
        if not isinstance(factor, str) or not factor.strip():
            raise ValidationError("every key_factor must be a non-empty string")
        if len(factor) > MAX_FACTOR_CHARS:
            raise ValidationError(f"a key_factor exceeds {MAX_FACTOR_CHARS} chars")

    # The provenance guard, over the rationale AND every factor.
    allowed = case.allowed_numbers()
    used = extract_numbers(rationale)
    for factor in factors:
        used |= extract_numbers(factor)
    invented = sorted(used - allowed)
    if invented:
        raise ValidationError(
            "response contains number(s) absent from the case file: "
            f"{', '.join(invented)}"
        )

    return Disposition(
        case_id=case.case_id,
        recommendation=payload["recommendation"],
        confidence=payload["confidence"],
        rationale=rationale,
        key_factors=tuple(f.strip() for f in factors),
    )
