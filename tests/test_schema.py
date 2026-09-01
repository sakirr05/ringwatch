"""Validation tests for the LLM output schema and the number-provenance guard.

The number guard is the load-bearing one. Everything else here rejects malformed output;
the number guard rejects *well-formed, plausible, confident* output that happens to
contain a figure the model made up — which is the failure mode that would actually reach
an analyst and be believed.
"""

from __future__ import annotations

import json

import pytest

from ai.contract import ClusterEvidence, extract_numbers
from ai.schema import ValidationError, validate

from tests.test_ai_boundary import make_evidence


def response(**overrides) -> str:
    payload = {
        "probable_cause": "CARD_TESTING",
        "confidence": "medium",
        "human_summary": "Four entities share a card fingerprint across 21 days.",
        "suggested_action": "Review the cluster manually.",
    }
    payload.update(overrides)
    return json.dumps(payload)


# --------------------------------------------------------------------------
# well-formed output
# --------------------------------------------------------------------------


def test_accepts_valid_response():
    narrative = validate(response(), make_evidence())
    assert narrative.probable_cause == "CARD_TESTING"
    assert narrative.confidence == "medium"
    assert narrative.cluster_id == 1
    assert narrative.status == "OK"


def test_accepts_response_wrapped_in_a_code_fence():
    """Models emit ```json fences often enough that rejecting them wastes a retry."""
    fenced = f"```json\n{response()}\n```"
    assert validate(fenced, make_evidence()).probable_cause == "CARD_TESTING"


# --------------------------------------------------------------------------
# malformed output
# --------------------------------------------------------------------------


def test_rejects_non_json():
    with pytest.raises(ValidationError, match="not valid JSON"):
        validate("I think this cluster looks like card testing.", make_evidence())


def test_rejects_json_array():
    with pytest.raises(ValidationError, match="expected a JSON object"):
        validate("[1, 2, 3]", make_evidence())


def test_rejects_missing_field():
    payload = json.loads(response())
    del payload["suggested_action"]
    with pytest.raises(ValidationError, match="missing required field"):
        validate(json.dumps(payload), make_evidence())


def test_rejects_extra_field():
    """An extra field usually means the model invented a decision field of its own."""
    payload = json.loads(response())
    payload["risk_score"] = "0.97"
    with pytest.raises(ValidationError, match="unexpected field"):
        validate(json.dumps(payload), make_evidence())


def test_rejects_non_string_field():
    with pytest.raises(ValidationError, match="must be a string"):
        validate(response(confidence=0.9), make_evidence())


def test_rejects_unknown_probable_cause():
    with pytest.raises(ValidationError, match="is not one of"):
        validate(response(probable_cause="DEFINITELY_FRAUD"), make_evidence())


def test_rejects_unknown_confidence():
    with pytest.raises(ValidationError, match="is not one of"):
        validate(response(confidence="very high"), make_evidence())


def test_rejects_empty_summary():
    with pytest.raises(ValidationError, match="empty"):
        validate(response(human_summary="   "), make_evidence())


def test_rejects_overlong_summary():
    with pytest.raises(ValidationError, match="limit"):
        validate(response(human_summary="x" * 5000), make_evidence())


# --------------------------------------------------------------------------
# the number-provenance guard
# --------------------------------------------------------------------------


def test_accepts_numbers_present_in_the_evidence():
    evidence = make_evidence()
    text = response(
        human_summary=(
            "This cluster contains 4 entities and 37 transactions, of which 12 were "
            "flagged, over a span of 21 days."
        )
    )
    assert validate(text, evidence).status == "OK"


def test_rejects_an_invented_rupee_total():
    """The classic hallucination: a confident, plausible, fabricated figure."""
    evidence = make_evidence()
    text = response(
        human_summary="The cluster moved Rs 4,20,000 across several cards."
    )
    with pytest.raises(ValidationError, match="absent from the evidence"):
        validate(text, evidence)


def test_rejects_an_invented_transaction_count():
    evidence = make_evidence()
    text = response(human_summary="We observed 4821 transactions in this cluster.")
    with pytest.raises(ValidationError, match="absent from the evidence"):
        validate(text, evidence)


def test_rejects_arithmetic_the_model_performed_itself():
    """37 transactions minus 12 flagged is 25 -- correct, and still not permitted.

    The model is not allowed to compute, even accurately: an analyst cannot tell a
    correct derived figure from an incorrect one, and permitting arithmetic means
    permitting arithmetic errors.
    """
    evidence = make_evidence()
    text = response(human_summary="That leaves 25 transactions unflagged.")
    with pytest.raises(ValidationError, match="absent from the evidence"):
        validate(text, evidence)


def test_rejects_invented_number_in_the_action_field_too():
    evidence = make_evidence()
    text = response(suggested_action="Freeze all 96 associated accounts.")
    with pytest.raises(ValidationError, match="absent from the evidence"):
        validate(text, evidence)


def test_small_integers_are_allowed_as_prose():
    """Banning 'two of the three' would reject valid prose without stopping fabrication."""
    evidence = make_evidence()
    text = response(human_summary="Two of the entities share a single address.")
    assert validate(text, evidence).status == "OK"


def test_amount_is_accepted_in_several_renderings():
    evidence = make_evidence(total_amount_inr=184320.50)
    for rendering in ("184320.50", "184320.5", "184,320.50", "184320"):
        text = response(human_summary=f"The cluster total is Rs {rendering}.")
        assert validate(text, evidence).status == "OK"


def test_extract_numbers_normalises_separators_and_decimals():
    assert extract_numbers("Rs 1,200.00 and 1200 and 1,200") == {"1200"}


def test_allowed_numbers_includes_every_evidence_figure():
    evidence = make_evidence()
    allowed = evidence.allowed_numbers()
    for value in (
        evidence.entity_count,
        evidence.transaction_count,
        evidence.flagged_transaction_count,
        evidence.span_days,
    ):
        assert str(value) in allowed
