"""LLM providers: Gemini primary, Groq fallback, retry once.

WHY THERE IS NO CIRCUIT BREAKER HERE
------------------------------------
An earlier design for this project carried a full circuit breaker — closed/open/half-open
states, exponential backoff with jitter, the works. It is deliberately absent here, and
that absence is an engineering judgment worth defending rather than an omission.

A circuit breaker earns its place when calls are synchronous, latency-critical, and in
the path of a user request, so that a failing dependency must be shed fast to protect the
caller. None of that describes this workload. The narrative layer runs **offline, in a
batch, over a few dozen already-flagged clusters**, after every score and every flag has
already been computed. If a provider is down, the correct behaviour is to try the other
one, then write NARRATIVE_UNAVAILABLE and move on — which costs nothing and blocks
nobody. Adding state machines and jitter would be infrastructure theatre: more code, more
tests, more failure modes, protecting against a harm that does not exist here.

What IS needed: a fallback provider, one retry, a hard timeout, and an honest failure
value. That is what this module implements.

The response cache is keyed on a SHA-256 of the exact prompt, so re-running the pipeline
reproduces byte-identical narratives without re-billing the API — which also means the
demo video and the metrics are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "cache" / "llm"

REQUEST_TIMEOUT_SECONDS = 45
MAX_ATTEMPTS_PER_PROVIDER = 2  # the initial call plus one retry

GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class ProviderError(Exception):
    """Every provider failed, or none was configured."""


@dataclass
class ProviderResponse:
    text: str
    provider: str
    cached: bool
    attempts: int


def _cache_path(prompt: str) -> Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _read_cache(prompt: str) -> ProviderResponse | None:
    path = _cache_path(prompt)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return ProviderResponse(
        text=payload["text"], provider=payload["provider"], cached=True, attempts=0
    )


def _write_cache(prompt: str, response: ProviderResponse) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(prompt).write_text(
        json.dumps({"text": response.text, "provider": response.provider})
    )


def _call_gemini(prompt: str, api_key: str) -> str:
    response = requests.post(
        GEMINI_URL.format(model=GEMINI_MODEL),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            # Temperature 0: the narrative layer should be as reproducible as the rest of
            # the pipeline. Creative variation is not a feature in a fraud report.
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(prompt: str, api_key: str) -> str:
    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def complete(prompt: str, use_cache: bool = True) -> ProviderResponse:
    """Get a completion, trying Gemini then Groq, one retry each.

    Raises ProviderError if every configured provider fails, so the caller can record
    NARRATIVE_UNAVAILABLE rather than receiving a fabricated string.
    """
    if use_cache:
        cached = _read_cache(prompt)
        if cached is not None:
            return cached

    providers = []
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if gemini_key:
        providers.append(("gemini", lambda p: _call_gemini(p, gemini_key)))
    if groq_key:
        providers.append(("groq", lambda p: _call_groq(p, groq_key)))

    if not providers:
        raise ProviderError(
            "no LLM provider configured; set GEMINI_API_KEY or GROQ_API_KEY "
            "(see .env.example). The deterministic pipeline does not need them."
        )

    failures: list[str] = []
    attempts = 0
    for name, call in providers:
        for attempt in range(MAX_ATTEMPTS_PER_PROVIDER):
            attempts += 1
            try:
                text = call(prompt)
                response = ProviderResponse(
                    text=text, provider=name, cached=False, attempts=attempts
                )
                if use_cache:
                    _write_cache(prompt, response)
                return response
            except Exception as exc:  # noqa: BLE001 - any failure means try the fallback
                failures.append(f"{name} attempt {attempt + 1}: {exc}")
                if attempt + 1 < MAX_ATTEMPTS_PER_PROVIDER:
                    time.sleep(1.0)

    raise ProviderError("; ".join(failures))
