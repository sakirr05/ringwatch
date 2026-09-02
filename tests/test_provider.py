"""Tests for the LLM provider layer, including the fallback that had never run.

WHY THIS FILE EXISTS
--------------------
`ai/provider.py` was functionally untested. It appeared in the suite only via the AST
import-boundary check, which reads it as text and never executes a line of it. More
pointedly: **`_call_groq` had never been called even once**, in any context — no Groq key
was ever configured, so the fallback was plausible-looking code with an unexercised request
shape and an unverified response parse.

A fallback that has never run is not a fallback. Its first execution would otherwise have
been in production, during the exact incident it exists to survive.

Every test here stubs `requests.post`, so nothing touches the network and no credential is
needed. What is being verified is the code's own behaviour: does it build the right
request, read the right field out of each provider's differently-shaped response, fall
through in the right order, retry the right number of times, and fail honestly when
everything is down.
"""

from __future__ import annotations

import json

import pytest

from ai import provider
from ai.provider import ProviderError, complete

PROMPT = "describe cluster 8"
GEMINI_TEXT = '{"probable_cause":"CARD_TESTING"}'
GROQ_TEXT = '{"probable_cause":"COORDINATED_RING"}'


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def gemini_payload(text: str = GEMINI_TEXT) -> dict:
    """The shape Gemini actually returns."""
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def groq_payload(text: str = GROQ_TEXT) -> dict:
    """The shape Groq actually returns -- OpenAI-compatible, and NOT Gemini's."""
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Own cache directory, no real sleeping, no real network."""
    monkeypatch.setattr(provider, "CACHE_DIR", tmp_path / "llm")
    monkeypatch.setattr(provider.time, "sleep", lambda _seconds: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yield


@pytest.fixture
def calls(monkeypatch):
    """Record every outbound request and serve scripted responses per host."""
    recorded: list[dict] = []
    script: dict[str, list] = {"gemini": [], "groq": []}

    def fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        which = "groq" if "groq.com" in url else "gemini"
        recorded.append(
            {"which": which, "url": url, "headers": headers or {}, "body": json or {}}
        )
        queue = script[which]
        if not queue:
            raise AssertionError(f"unexpected extra call to {which}")
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(provider.requests, "post", fake_post)
    return {"recorded": recorded, "script": script}


# --------------------------------------------------------------------------
# the Groq path, exercised for the first time
# --------------------------------------------------------------------------


def test_groq_response_is_parsed_from_the_openai_shape(calls, monkeypatch):
    """Groq nests its text under choices[0].message.content, not Gemini's candidates path.

    Reading the wrong field would raise KeyError on first use -- which, before this test,
    would have happened live during a Gemini outage.
    """
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    calls["script"]["groq"] = [FakeResponse(groq_payload())]

    result = complete(PROMPT, use_cache=False)
    assert result.text == GROQ_TEXT
    assert result.provider == "groq"


def test_groq_request_is_shaped_correctly(calls, monkeypatch):
    """Auth header, model and JSON-mode flag must all be present and correct."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    calls["script"]["groq"] = [FakeResponse(groq_payload())]

    complete(PROMPT, use_cache=False)
    sent = calls["recorded"][0]

    assert sent["headers"]["Authorization"] == "Bearer gsk_test"
    assert sent["body"]["model"] == provider.GROQ_MODEL
    assert sent["body"]["messages"] == [{"role": "user", "content": PROMPT}]
    # Temperature 0 and JSON mode: the narrative must be reproducible and parseable.
    assert sent["body"]["temperature"] == 0.0
    assert sent["body"]["response_format"] == {"type": "json_object"}


def test_groq_http_error_is_raised(calls, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    calls["script"]["groq"] = [
        FakeResponse({}, status=500),
        FakeResponse({}, status=500),
    ]
    with pytest.raises(ProviderError):
        complete(PROMPT, use_cache=False)


# --------------------------------------------------------------------------
# the Gemini path, and the order between them
# --------------------------------------------------------------------------


def test_gemini_response_is_parsed_from_its_own_shape(calls, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    calls["script"]["gemini"] = [FakeResponse(gemini_payload())]

    result = complete(PROMPT, use_cache=False)
    assert result.text == GEMINI_TEXT
    assert result.provider == "gemini"


def test_gemini_is_preferred_and_groq_is_not_called(calls, monkeypatch):
    """With both configured and Gemini healthy, the fallback must stay untouched."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    calls["script"]["gemini"] = [FakeResponse(gemini_payload())]

    result = complete(PROMPT, use_cache=False)
    assert result.provider == "gemini"
    assert [c["which"] for c in calls["recorded"]] == ["gemini"]


def test_groq_takes_over_when_gemini_fails(calls, monkeypatch):
    """THE test this file exists for: the fallback actually falls back.

    Gemini fails both its attempts, Groq answers, and the caller gets a usable narrative
    rather than an exception.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    calls["script"]["gemini"] = [RuntimeError("read timeout"), RuntimeError("read timeout")]
    calls["script"]["groq"] = [FakeResponse(groq_payload())]

    result = complete(PROMPT, use_cache=False)

    assert result.provider == "groq"
    assert result.text == GROQ_TEXT
    assert [c["which"] for c in calls["recorded"]] == ["gemini", "gemini", "groq"]


def test_each_provider_is_retried_once(calls, monkeypatch):
    """One retry, not zero and not an unbounded loop."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    calls["script"]["gemini"] = [RuntimeError("transient"), FakeResponse(gemini_payload())]

    result = complete(PROMPT, use_cache=False)
    assert result.provider == "gemini"
    assert len(calls["recorded"]) == 2
    assert result.attempts == 2


def test_every_provider_failing_raises_with_all_reasons(calls, monkeypatch):
    """The caller records NARRATIVE_UNAVAILABLE, so the reason must survive."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    calls["script"]["gemini"] = [RuntimeError("gemini down"), RuntimeError("gemini down")]
    calls["script"]["groq"] = [RuntimeError("groq down"), RuntimeError("groq down")]

    with pytest.raises(ProviderError) as excinfo:
        complete(PROMPT, use_cache=False)

    message = str(excinfo.value)
    assert "gemini down" in message
    assert "groq down" in message
    assert len(calls["recorded"]) == 4  # two providers, two attempts each


def test_no_configured_provider_explains_itself():
    """Absence of credentials is a normal state, and must say so usefully."""
    with pytest.raises(ProviderError) as excinfo:
        complete(PROMPT, use_cache=False)

    message = str(excinfo.value)
    assert "GEMINI_API_KEY" in message and "GROQ_API_KEY" in message
    # And it must make clear nothing else is affected.
    assert "deterministic pipeline does not need them" in message


# --------------------------------------------------------------------------
# the response cache
# --------------------------------------------------------------------------


def test_a_cached_prompt_is_not_re_requested(calls, monkeypatch):
    """Re-runs must be free and byte-identical, which is what makes demos reproducible."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    calls["script"]["gemini"] = [FakeResponse(gemini_payload())]

    first = complete(PROMPT, use_cache=True)
    second = complete(PROMPT, use_cache=True)

    assert first.text == second.text
    assert first.cached is False and second.cached is True
    assert len(calls["recorded"]) == 1  # the second call never hit the network


def test_a_different_prompt_misses_the_cache(calls, monkeypatch):
    """Keyed on the prompt, so changed evidence cannot serve a stale narrative."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    calls["script"]["gemini"] = [
        FakeResponse(gemini_payload("first")),
        FakeResponse(gemini_payload("second")),
    ]

    assert complete("prompt A", use_cache=True).text == "first"
    assert complete("prompt B", use_cache=True).text == "second"
    assert len(calls["recorded"]) == 2


def test_cache_is_written_as_readable_json(calls, monkeypatch, tmp_path):
    """The cache is an artifact a human may need to inspect or delete selectively."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    calls["script"]["gemini"] = [FakeResponse(gemini_payload())]

    complete(PROMPT, use_cache=True)
    files = list((tmp_path / "llm").glob("*.json"))
    assert len(files) == 1

    stored = json.loads(files[0].read_text())
    assert stored["text"] == GEMINI_TEXT
    assert stored["provider"] == "gemini"
    # Named by SHA-256 of the prompt: 64 hex characters.
    assert len(files[0].stem) == 64


def test_use_cache_false_bypasses_a_populated_cache(calls, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza_test")
    calls["script"]["gemini"] = [
        FakeResponse(gemini_payload("one")),
        FakeResponse(gemini_payload("two")),
    ]

    complete(PROMPT, use_cache=True)
    fresh = complete(PROMPT, use_cache=False)
    assert fresh.text == "two"
    assert len(calls["recorded"]) == 2
