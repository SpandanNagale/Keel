"""The one and only place Keel talks to an LLM.

``complete_json`` is the single entry point. It returns a ``(result, error)``
tuple — never a bare ``None``, never a raised exception. Exactly one of the two
is set:

  * success  -> ``(dict, None)``
  * failure  -> ``(None, "<human-readable reason>")``

The caller is required to surface ``error`` to the user. Silent fallback to
template text — the bug that made the previous build never call the model — is
structurally impossible here: there is no code path that discards the reason.

Providers:

  * ``groq``, ``ollama-cloud``, ``anthropic`` — hosted, chosen by which API key
    is configured (``PROVIDER_ORDER``), or forced with ``KEEL_PROVIDER``.
  * ``ollama`` — a **local** daemon at ``http://localhost:11434``, for
    development only. Selected exclusively via ``KEEL_PROVIDER=ollama``; needs no
    key. The deployed app never reaches this path unless a deployer explicitly
    sets that env var, and if it does and nothing answers on localhost the error
    says so — it never silently degrades to "as if the call succeeded".

Each provider enforces JSON: Groq via a native response mode, Ollama via
``format="json"``, Anthropic via an assistant-turn prefill of ``{``. A model that
answers in prose still fails ``json.loads`` with a *distinguishable* parse error.

No provider branching exists outside this module.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping, Optional

# Default model per provider. Override the active one with the KEEL_MODEL secret
# (or KEEL_OLLAMA_MODEL for the local daemon).
DEFAULT_MODELS: dict[str, str] = {
    "groq": "openai/gpt-oss-120b",
    "ollama-cloud": "gpt-oss:120b",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.2",
}

# When several hosted keys are present and KEEL_PROVIDER is unset, the first of
# these with a key wins.
PROVIDER_ORDER: tuple[str, ...] = ("groq", "ollama-cloud", "anthropic")

# Env / secret name carrying each hosted provider's key. The local "ollama"
# provider is deliberately absent — it needs no key.
SECRET_KEYS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "ollama-cloud": "OLLAMA_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_NEEDS_KEY = frozenset(SECRET_KEYS)

OLLAMA_CLOUD_HOST = "https://ollama.com"
LOCAL_OLLAMA_HOST = "http://localhost:11434"

# Vision-capable model per provider, for reference Mode C (image intake).
# Groq's default model is text-only; a vision model must be named explicitly via
# KEEL_VISION_MODEL. Override any of these with KEEL_VISION_MODEL.
VISION_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.2-vision",
    "ollama-cloud": "llama3.2-vision",
}
VISION_MIME_TYPES = ("image/png", "image/jpeg", "image/webp")

# One question — plus its recommended answer, a one-line rationale, and a
# one-line "revisit this if" condition — is never worth more than this.
MAX_OUTPUT_TOKENS = 900

# Back-compat alias: some callers/tests still reference a single default.
DEFAULT_MODEL = DEFAULT_MODELS["anthropic"]


@dataclass(frozen=True)
class Provider:
    name: str
    api_key: str
    model: str
    host: str = ""  # only meaningful for the ollama providers


def resolve_provider(
    available: Mapping[str, str],
    *,
    model_override: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Optional[Provider], Optional[str]]:
    """Pick a provider.

    ``KEEL_PROVIDER`` (from ``env``, default ``os.environ``) forces one:
      * ``ollama``       -> the local daemon, no key.
      * ``groq`` / ``anthropic`` / ``ollama-cloud`` -> that hosted provider,
        erroring if its key is absent from ``available``.
    Unset -> the first hosted key present, in ``PROVIDER_ORDER``.
    """
    env = env if env is not None else os.environ
    forced = str(env.get("KEEL_PROVIDER", "")).strip().lower()

    if forced == "ollama":
        model = (
            (model_override or "").strip()
            or str(env.get("KEEL_OLLAMA_MODEL", "")).strip()
            or DEFAULT_MODELS["ollama"]
        )
        return Provider("ollama", api_key="", model=model, host=LOCAL_OLLAMA_HOST), None

    if forced in _NEEDS_KEY:
        key = str(available.get(SECRET_KEYS[forced]) or available.get(forced) or "").strip()
        if not key:
            return None, f"KEEL_PROVIDER={forced} but {SECRET_KEYS[forced]} is not configured"
        model = (model_override or "").strip() or DEFAULT_MODELS[forced]
        return Provider(forced, key, model), None

    if forced:
        return None, f"KEEL_PROVIDER={forced!r} is not a known provider"

    for name in PROVIDER_ORDER:
        key = str(available.get(SECRET_KEYS[name]) or available.get(name) or "").strip()
        if key:
            model = (model_override or "").strip() or DEFAULT_MODELS[name]
            return Provider(name, key, model), None
    return None, (
        "no API key configured (set one of "
        + ", ".join(SECRET_KEYS.values())
        + ", or KEEL_PROVIDER=ollama for a local daemon)"
    )


def resolve_vision_provider(
    available: Mapping[str, str],
    *,
    model_override: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Optional[Provider], Optional[str]]:
    """Pick a provider that can read an image for reference Mode C.

    ``KEEL_PROVIDER=ollama`` -> the local daemon with a vision model. Otherwise
    Anthropic if its key is present (Claude Haiku is multimodal); or Groq if
    ``KEEL_VISION_MODEL`` names one and only the Groq key is configured. No vision
    setup -> ``(None, reason)`` so the caller can degrade with a clear message.
    """
    env = env if env is not None else os.environ
    forced = str(env.get("KEEL_PROVIDER", "")).strip().lower()
    vm = (model_override or "").strip() or str(env.get("KEEL_VISION_MODEL", "")).strip()

    if forced == "ollama":
        model = vm or str(env.get("KEEL_OLLAMA_VISION_MODEL", "")).strip() or VISION_MODELS["ollama"]
        return Provider("ollama", api_key="", model=model, host=LOCAL_OLLAMA_HOST), None

    anth = str(available.get("ANTHROPIC_API_KEY") or available.get("anthropic") or "").strip()
    if anth:
        return Provider("anthropic", anth, vm or VISION_MODELS["anthropic"]), None

    groq_key = str(available.get("GROQ_API_KEY") or available.get("groq") or "").strip()
    if groq_key and vm:
        return Provider("groq", groq_key, vm), None

    return None, (
        "no vision-capable model is configured — set ANTHROPIC_API_KEY, or run a "
        "local Ollama vision model (KEEL_PROVIDER=ollama), or set KEEL_VISION_MODEL "
        "to a vision model on your Groq key"
    )


def complete_json(
    system: str,
    user: str,
    *,
    provider: Optional[Provider],
    max_tokens: int = MAX_OUTPUT_TOKENS,
    image: Optional[tuple[bytes, str]] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Make one call to ``provider`` and parse a JSON object out of the reply.

    Returns ``(parsed_dict, None)`` on success or ``(None, reason)`` on any
    failure: no provider, missing key, import failure, network/API error, empty
    reply, non-JSON reply, or JSON that is not an object.

    ``image`` is an optional ``(bytes, mime_type)`` pair for a multimodal call
    (reference Mode C). Only ``anthropic``, ``ollama``, ``ollama-cloud``, and a
    Groq provider whose ``model`` is explicitly a vision model accept it.
    """
    if provider is None:
        return None, "no LLM provider configured for this session"
    if provider.name in _NEEDS_KEY and not str(provider.api_key).strip():
        return None, f"no {provider.name} API key configured for this session"
    if image is not None:
        img_bytes, mime = image
        if mime not in VISION_MIME_TYPES:
            return None, f"unsupported image type {mime!r} (use PNG, JPEG, or WebP)"
        if not img_bytes:
            return None, "the image is empty"

    dispatch = {
        "groq": _groq_raw,
        "ollama-cloud": _ollama_raw,
        "ollama": _ollama_raw,
        "anthropic": _anthropic_raw,
    }
    fn = dispatch.get(provider.name)
    if fn is None:
        return None, f"unknown provider {provider.name!r}"

    try:
        raw, error = fn(system, user, provider, max_tokens, image)
    except Exception as exc:  # noqa: BLE001 - every failure must become a reason string
        return None, f"{provider.name}: {type(exc).__name__}: {exc}"

    if error is not None:
        return None, error
    if not raw or not raw.strip():
        return None, f"{provider.name} returned no text content"
    return _parse_json_object(raw)


# --------------------------------------------------------------------------- #
# Per-provider calls: return (raw_text, error). Exceptions bubble to caller.
# The trailing ``image`` arg is an optional (bytes, mime) pair for a vision call.
# --------------------------------------------------------------------------- #
def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


def _anthropic_raw(system, user, provider, max_tokens, image=None):
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        return None, f"anthropic package not importable: {exc}"

    if image is not None:
        img_bytes, mime = image
        user_content = [
            {"type": "image", "source": {"type": "base64", "media_type": mime,
                                         "data": _b64(img_bytes)}},
            {"type": "text", "text": user},
        ]
    else:
        user_content = user

    client = anthropic.Anthropic(api_key=provider.api_key)
    response = client.messages.create(
        model=provider.model,
        max_tokens=max_tokens,
        system=system,
        messages=[
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "{"},
        ],
    )
    parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if not parts:
        return None, "anthropic returned no text content"
    return "{" + "".join(parts), None


def _groq_raw(system, user, provider, max_tokens, image=None):
    try:
        import groq
    except ImportError as exc:  # pragma: no cover
        return None, f"groq package not importable: {exc}"

    if image is not None:
        img_bytes, mime = image
        user_msg = {"role": "user", "content": [
            {"type": "text", "text": user},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{_b64(img_bytes)}"}},
        ]}
    else:
        user_msg = {"role": "user", "content": user}

    client = groq.Groq(api_key=provider.api_key)
    response = client.chat.completions.create(
        model=provider.model,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, user_msg],
    )
    if not response.choices:
        return None, "groq returned no choices"
    return response.choices[0].message.content, None


def _ollama_raw(system, user, provider, max_tokens, image=None):
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover
        return None, f"ollama package not importable: {exc}"

    host = provider.host or OLLAMA_CLOUD_HOST
    headers = {"Authorization": f"Bearer {provider.api_key}"} if provider.api_key else None
    client = ollama.Client(host=host, headers=headers)
    user_msg = {"role": "user", "content": user}
    if image is not None:
        user_msg["images"] = [_b64(image[0])]
    try:
        response = client.chat(
            model=provider.model,
            messages=[{"role": "system", "content": system}, user_msg],
            format="json",
            options={"num_predict": max_tokens},
        )
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__.lower()
        text = str(exc).lower()
        if "connect" in name or "connection" in text or "refused" in text or "max retries" in text:
            return None, (
                f"Ollama not reachable at {host} — is the local daemon running "
                f"(`ollama serve`) and the model pulled (`ollama pull {provider.model}`)?"
            )
        raise

    content = response.get("message", {}).get("content") if hasattr(response, "get") else None
    if content is None:
        content = getattr(getattr(response, "message", None), "content", None)
    return content, None


# --------------------------------------------------------------------------- #
def _parse_json_object(raw: str) -> tuple[Optional[dict], Optional[str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # One recovery attempt: trim anything trailing the final closing brace
        # (reasoning models sometimes append a note after the object).
        end = text.rfind("}")
        if end != -1:
            try:
                parsed = json.loads(text[: end + 1])
            except (json.JSONDecodeError, ValueError):
                return None, f"model did not return valid JSON; got: {raw[:200]!r}"
        else:
            return None, f"model did not return valid JSON; got: {raw[:200]!r}"

    if not isinstance(parsed, dict):
        return None, f"model returned JSON but not an object; got: {raw[:200]!r}"
    return parsed, None
