"""llm contract:

  * resolve_provider picks by key presence, in PROVIDER_ORDER.
  * complete_json always returns a (result, error) tuple, never raises, with
    exactly one side populated — for every provider.

The three vendor SDKs are faked.
"""
from __future__ import annotations

import sys
import types

import pytest

from keel import llm

P = lambda name: llm.Provider(name, "test-key", llm.DEFAULT_MODELS[name])


# --------------------------------------------------------------------------- #
# resolve_provider  (env={} so a stray KEEL_PROVIDER can't leak in)
# --------------------------------------------------------------------------- #
def test_resolve_prefers_groq_then_ollama_cloud_then_anthropic():
    both = {"GROQ_API_KEY": "g", "OLLAMA_API_KEY": "o", "ANTHROPIC_API_KEY": "a"}
    assert llm.resolve_provider(both, env={})[0].name == "groq"
    assert llm.resolve_provider({"OLLAMA_API_KEY": "o", "ANTHROPIC_API_KEY": "a"}, env={})[0].name == "ollama-cloud"
    assert llm.resolve_provider({"ANTHROPIC_API_KEY": "a"}, env={})[0].name == "anthropic"


def test_resolve_uses_default_model_and_honours_override():
    prov, err = llm.resolve_provider({"GROQ_API_KEY": "g"}, env={})
    assert err is None and prov.model == llm.DEFAULT_MODELS["groq"]
    prov2, _ = llm.resolve_provider({"GROQ_API_KEY": "g"}, model_override="custom-model", env={})
    assert prov2.model == "custom-model"


def test_resolve_with_no_keys_is_an_error():
    prov, err = llm.resolve_provider({}, env={})
    assert prov is None
    assert "no API key configured" in err


def test_keel_provider_ollama_is_local_and_needs_no_key():
    prov, err = llm.resolve_provider({}, env={"KEEL_PROVIDER": "ollama"})
    assert err is None
    assert prov.name == "ollama" and prov.api_key == ""
    assert prov.host == llm.LOCAL_OLLAMA_HOST
    assert prov.model == llm.DEFAULT_MODELS["ollama"]


def test_keel_provider_ollama_model_from_env():
    prov, _ = llm.resolve_provider(
        {}, env={"KEEL_PROVIDER": "ollama", "KEEL_OLLAMA_MODEL": "phi3:mini"}
    )
    assert prov.model == "phi3:mini"


def test_keel_provider_forces_hosted_and_errors_without_its_key():
    prov, err = llm.resolve_provider({"GROQ_API_KEY": "g"}, env={"KEEL_PROVIDER": "anthropic"})
    assert prov is None and "ANTHROPIC_API_KEY is not configured" in err
    prov2, err2 = llm.resolve_provider({"ANTHROPIC_API_KEY": "a"}, env={"KEEL_PROVIDER": "anthropic"})
    assert err2 is None and prov2.name == "anthropic"


def test_keel_provider_unknown_value_is_an_error():
    prov, err = llm.resolve_provider({"GROQ_API_KEY": "g"}, env={"KEEL_PROVIDER": "banana"})
    assert prov is None and "not a known provider" in err


# --------------------------------------------------------------------------- #
# resolve_vision_provider  (reference Mode C)
# --------------------------------------------------------------------------- #
def test_vision_prefers_anthropic_when_its_key_is_present():
    prov, err = llm.resolve_vision_provider(
        {"GROQ_API_KEY": "g", "ANTHROPIC_API_KEY": "a"}, env={}
    )
    assert err is None and prov.name == "anthropic"
    assert prov.model == llm.VISION_MODELS["anthropic"]


def test_vision_uses_local_ollama_when_forced():
    prov, err = llm.resolve_vision_provider({}, env={"KEEL_PROVIDER": "ollama"})
    assert err is None and prov.name == "ollama" and prov.host == llm.LOCAL_OLLAMA_HOST
    assert prov.model == llm.VISION_MODELS["ollama"]


def test_vision_allows_groq_only_with_an_explicit_vision_model():
    none_prov, err = llm.resolve_vision_provider({"GROQ_API_KEY": "g"}, env={})
    assert none_prov is None and "no vision-capable model" in err
    prov, err2 = llm.resolve_vision_provider(
        {"GROQ_API_KEY": "g"}, env={"KEEL_VISION_MODEL": "some-vlm"}
    )
    assert err2 is None and prov.name == "groq" and prov.model == "some-vlm"


def test_vision_with_nothing_configured_is_a_clear_error():
    prov, err = llm.resolve_vision_provider({}, env={})
    assert prov is None and "ANTHROPIC_API_KEY" in err


def test_complete_json_rejects_a_bad_image_mime():
    result, error = llm.complete_json(
        "s", "u", provider=P("anthropic"), image=(b"\x89PNG", "image/gif")
    )
    assert result is None and "unsupported image type" in error


def test_anthropic_vision_message_carries_the_image_block(fake_anthropic):
    seen = {}

    class R:
        content = [type("B", (), {"type": "text", "text": '"ok": true}'})()]

    fake_anthropic(lambda **kw: seen.update(kw) or R())
    result, error = llm.complete_json(
        "s", "describe", provider=P("anthropic"), image=(b"\x89PNGdata", "image/png")
    )
    assert error is None and result == {"ok": True}
    content = seen["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/png"
    assert content[1] == {"type": "text", "text": "describe"}


# --------------------------------------------------------------------------- #
# complete_json — guard rails
# --------------------------------------------------------------------------- #
def test_missing_provider_is_an_error_not_a_call():
    assert llm.complete_json("s", "u", provider=None) == (None, "no LLM provider configured for this session")


def test_blank_key_is_an_error():
    result, error = llm.complete_json("s", "u", provider=llm.Provider("groq", "  ", "m"))
    assert result is None and "no groq API key" in error


# --------------------------------------------------------------------------- #
# Anthropic path
# --------------------------------------------------------------------------- #
class _ABlock:
    def __init__(self, text):
        self.type, self.text = "text", text


def _fake_anthropic(handler):
    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kw):
            return handler(**kw)

    class _Anthropic:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    mod.Anthropic = _Anthropic
    return mod


@pytest.fixture
def fake_anthropic(monkeypatch):
    return lambda h: monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(h))


def test_anthropic_prefill_brace_reattached_and_parsed(fake_anthropic):
    class R:
        content = [_ABlock('"a": 1, "b": [2, 3]}')]

    fake_anthropic(lambda **kw: R())
    result, error = llm.complete_json("s", "u", provider=P("anthropic"))
    assert error is None and result == {"a": 1, "b": [2, 3]}


def test_anthropic_prefill_is_actually_sent(fake_anthropic):
    seen = {}

    class R:
        content = [_ABlock('"ok": true}')]

    fake_anthropic(lambda **kw: seen.update(kw) or R())
    llm.complete_json("s", "u", provider=P("anthropic"))
    assert seen["messages"][-1] == {"role": "assistant", "content": "{"}
    assert seen["max_tokens"] == llm.MAX_OUTPUT_TOKENS


def test_anthropic_prose_reply_is_a_distinguishable_parse_error(fake_anthropic):
    class R:
        content = [_ABlock("I think you want a CLI tool, here's why...")]

    fake_anthropic(lambda **kw: R())
    result, error = llm.complete_json("s", "u", provider=P("anthropic"))
    assert result is None and "did not return valid JSON" in error


def test_anthropic_api_exception_becomes_a_reason_string(fake_anthropic):
    def boom(**kw):
        raise RuntimeError("rate limited")

    fake_anthropic(boom)
    result, error = llm.complete_json("s", "u", provider=P("anthropic"))
    assert result is None and "anthropic: RuntimeError: rate limited" in error


# --------------------------------------------------------------------------- #
# Groq path
# --------------------------------------------------------------------------- #
def _fake_groq(handler):
    mod = types.ModuleType("groq")

    class _Completions:
        def create(self, **kw):
            return handler(**kw)

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Groq:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    mod.Groq = _Groq
    return mod


@pytest.fixture
def fake_groq(monkeypatch):
    return lambda h: monkeypatch.setitem(sys.modules, "groq", _fake_groq(h))


def test_groq_reads_choice_message_content(fake_groq):
    msg = types.SimpleNamespace(message=types.SimpleNamespace(content='{"q": "ok"}'))
    seen = {}
    fake_groq(lambda **kw: seen.update(kw) or types.SimpleNamespace(choices=[msg]))
    result, error = llm.complete_json("s", "u", provider=P("groq"))
    assert error is None and result == {"q": "ok"}
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["model"] == llm.DEFAULT_MODELS["groq"]


def test_groq_no_choices_is_an_error(fake_groq):
    fake_groq(lambda **kw: types.SimpleNamespace(choices=[]))
    result, error = llm.complete_json("s", "u", provider=P("groq"))
    assert result is None and "no choices" in error


def test_groq_exception_becomes_a_reason_string(fake_groq):
    def boom(**kw):
        raise ValueError("bad request")

    fake_groq(boom)
    result, error = llm.complete_json("s", "u", provider=P("groq"))
    assert result is None and "groq: ValueError: bad request" in error


# --------------------------------------------------------------------------- #
# Ollama path (shared by "ollama-cloud" and local "ollama")
# --------------------------------------------------------------------------- #
def _fake_ollama(handler):
    mod = types.ModuleType("ollama")
    mod.last_init = {}

    class _Client:
        def __init__(self, host=None, headers=None):
            mod.last_init.update(host=host, headers=headers)

        def chat(self, **kw):
            return handler(**kw)

    mod.Client = _Client
    return mod


@pytest.fixture
def fake_ollama(monkeypatch):
    return lambda h: monkeypatch.setitem(sys.modules, "ollama", _fake_ollama(h))


def test_ollama_reads_message_content_and_sets_json_format(fake_ollama):
    seen = {}
    fake_ollama(lambda **kw: seen.update(kw) or {"message": {"content": '{"ok": 1}'}})
    result, error = llm.complete_json("s", "u", provider=P("ollama-cloud"))
    assert error is None and result == {"ok": 1}
    assert seen["format"] == "json"
    assert seen["options"] == {"num_predict": llm.MAX_OUTPUT_TOKENS}


def test_ollama_cloud_targets_cloud_host_with_bearer_auth(fake_ollama):
    fake_ollama(lambda **kw: {"message": {"content": "{}"}})
    llm.complete_json("s", "u", provider=llm.Provider("ollama-cloud", "sk-secret", "gpt-oss:120b"))
    init = sys.modules["ollama"].last_init
    assert init["host"] == llm.OLLAMA_CLOUD_HOST
    assert init["headers"] == {"Authorization": "Bearer sk-secret"}


def test_local_ollama_targets_localhost_with_no_auth(fake_ollama):
    fake_ollama(lambda **kw: {"message": {"content": "{}"}})
    prov = llm.Provider("ollama", api_key="", model="llama3.2", host=llm.LOCAL_OLLAMA_HOST)
    result, error = llm.complete_json("s", "u", provider=prov)
    assert error is None
    init = sys.modules["ollama"].last_init
    assert init["host"] == llm.LOCAL_OLLAMA_HOST
    assert init["headers"] is None


def test_ollama_connection_refused_is_a_clear_message(fake_ollama):
    def boom(**kw):
        raise ConnectionError("[Errno 111] Connection refused")

    fake_ollama(boom)
    prov = llm.Provider("ollama", api_key="", model="llama3.2", host=llm.LOCAL_OLLAMA_HOST)
    result, error = llm.complete_json("s", "u", provider=prov)
    assert result is None
    assert "Ollama not reachable at http://localhost:11434" in error


def test_ollama_non_connection_exception_becomes_a_reason_string(fake_ollama):
    def boom(**kw):
        raise RuntimeError("model not found")

    fake_ollama(boom)
    result, error = llm.complete_json("s", "u", provider=P("ollama-cloud"))
    assert result is None and "ollama-cloud: RuntimeError: model not found" in error
