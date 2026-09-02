from __future__ import annotations

import pytest

from keel import engine
from keel.models import SessionState, SlotValue


@pytest.fixture
def make_session():
    def _make(prompt: str = "build a small tool", template: str = "default") -> SessionState:
        return engine.start_session(prompt, template, created_date="2026-09-01")

    return _make


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace keel.llm.complete_json with a scripted fake.

    Pass a callable(system, user, **kw) -> (result, error), or a fixed
    (result, error) tuple.
    """

    def _install(responder):
        if callable(responder):
            fn = responder
        else:
            def fn(system, user, **kw):
                return responder

        calls: list[tuple[str, str]] = []

        def wrapped(system, user, **kw):
            calls.append((system, user))
            return fn(system, user, **kw)

        monkeypatch.setattr("keel.llm.complete_json", wrapped)
        return calls

    return _install


def filled(**names_to_values) -> dict[str, SlotValue]:
    return {
        name: SlotValue(value=val, source="llm_default")
        for name, val in names_to_values.items()
    }
