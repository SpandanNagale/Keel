"""The single entry point for every LLM call Keel makes.

Every call in the codebase must go through complete() / complete_json() so that:
  - the eval harness can stub this module instead of hitting the network, and
  - degradation on API failure (or on no provider being configured) is handled
    in exactly one place.

Provider selection is a fixed priority chain, picked by whichever API key is
present in the environment -- Anthropic, then Groq, then Ollama Cloud:
  ANTHROPIC_API_KEY -> claude-opus-5
  GROQ_API_KEY      -> openai/gpt-oss-120b
  OLLAMA_API_KEY    -> gpt-oss:120b (via Ollama Cloud, https://ollama.com)

Both functions return None on any failure (no key configured, network error,
API error, bad JSON) -- callers are required to fall back to static template
text rather than crash mid-session.
"""
from __future__ import annotations

import json
import os
from typing import Optional

ANTHROPIC_MODEL = "claude-opus-5"
GROQ_MODEL = "openai/gpt-oss-120b"
OLLAMA_MODEL = "gpt-oss:120b"


def _active_provider() -> Optional[str]:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OLLAMA_API_KEY"):
        return "ollama"
    return None


def _call_anthropic(system: str, user: str, max_tokens: int, json_mode: bool) -> Optional[str]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": user}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return None


def _call_groq(system: str, user: str, max_tokens: int, json_mode: bool) -> Optional[str]:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    return response.choices[0].message.content


def _call_ollama(system: str, user: str, max_tokens: int, json_mode: bool) -> Optional[str]:
    from ollama import Client

    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    kwargs = {"format": "json"} if json_mode else {}
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"num_predict": max_tokens},
        **kwargs,
    )
    return response["message"]["content"]


_CALLERS = {
    "anthropic": _call_anthropic,
    "groq": _call_groq,
    "ollama": _call_ollama,
}


def _call(system: str, user: str, max_tokens: int, json_mode: bool) -> Optional[str]:
    provider = _active_provider()
    if provider is None:
        return None
    try:
        return _CALLERS[provider](system, user, max_tokens, json_mode)
    except Exception:
        return None


def complete(system: str, user: str, *, max_tokens: int = 1024, effort: str = "low") -> Optional[str]:
    return _call(system, user, max_tokens, json_mode=False)


def complete_json(system: str, user: str, *, max_tokens: int = 1024, effort: str = "low") -> Optional[dict]:
    text = _call(system, user, max_tokens, json_mode=True)
    if text is None:
        return None

    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    return parsed if isinstance(parsed, dict) else None
