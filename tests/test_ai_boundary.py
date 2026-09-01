"""The AI/determinism boundary, enforced by code rather than asserted in a README.

RingWatch's central claim is that the language model cannot reach any number. A comment
saying so is worth nothing — someone adds one import six months later and the claim
silently becomes false. So the boundary is a test: it parses every module under `ai/` and
fails if any of them can reach the scoring engine at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai.contract import ClusterEvidence, ClusterNarrative, unavailable

AI_PACKAGE = Path(__file__).resolve().parent.parent / "ai"


def imported_modules(path: Path) -> set[str]:
    """Every module name imported by a source file."""
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def ai_source_files() -> list[Path]:
    return sorted(p for p in AI_PACKAGE.glob("*.py") if p.name != "__init__.py")


def test_there_are_ai_modules_to_check():
    """Guard against this whole test file silently passing on an empty glob."""
    assert ai_source_files()


@pytest.mark.parametrize("path", ai_source_files(), ids=lambda p: p.name)
def test_ai_never_imports_the_deterministic_engine(path: Path):
    """No module under ai/ may import core/ — that is the boundary.

    If this fails, the language model has a code path to the model, the scores, or the
    graph, and RingWatch's determinism claim is no longer true.
    """
    offending = {m for m in imported_modules(path) if m == "core" or m.startswith("core.")}
    assert not offending, (
        f"{path.name} imports the deterministic engine: {sorted(offending)}. "
        "The AI layer must only receive evidence through ai.contract."
    )


@pytest.mark.parametrize("path", ai_source_files(), ids=lambda p: p.name)
def test_ai_never_imports_a_modelling_library(path: Path):
    """The narrative layer has no business holding a model or a fitted transformer."""
    banned = {"lightgbm", "sklearn", "xgboost", "torch", "networkx"}
    offending = {m.split(".")[0] for m in imported_modules(path)} & banned
    assert not offending, f"{path.name} imports modelling library: {sorted(offending)}"


def test_evidence_is_immutable():
    """The narrative layer cannot alter the evidence it was handed."""
    evidence = make_evidence()
    with pytest.raises(Exception):
        evidence.max_risk_score = 0.99  # type: ignore[misc]


def test_narrative_carries_no_score_field():
    """A narrative must not be able to express a score, flag, or decision."""
    fields = set(ClusterNarrative.__dataclass_fields__)
    for forbidden in ("score", "risk", "flag", "is_fraud", "decision", "threshold"):
        assert not any(forbidden in field for field in fields), (
            f"ClusterNarrative exposes a '{forbidden}'-like field; the narrative layer "
            "must not carry decisions."
        )


def test_unavailable_fallback_is_honest():
    """The fallback must be unmistakably empty, never a plausible-looking guess."""
    narrative = unavailable(7, "provider down")
    assert narrative.status == "NARRATIVE_UNAVAILABLE"
    assert narrative.human_summary == "NARRATIVE_UNAVAILABLE"
    assert narrative.suggested_action == "NARRATIVE_UNAVAILABLE"
    assert narrative.confidence == "low"
    assert narrative.rejection_reason == "provider down"


def make_evidence(**overrides) -> ClusterEvidence:
    defaults = dict(
        cluster_id=1,
        entity_count=4,
        transaction_count=37,
        flagged_transaction_count=12,
        component_size=4,
        core_number=2,
        max_degree=3,
        shared_attributes=("card1", "addr1_pemail"),
        distinct_cards=4,
        distinct_addresses=1,
        distinct_email_domains=1,
        span_days=21,
        total_amount_inr=184320.50,
        max_risk_score=0.8123,
        mean_risk_score=0.4471,
    )
    defaults.update(overrides)
    return ClusterEvidence(**defaults)  # type: ignore[arg-type]
