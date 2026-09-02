"""Phase 1: contradiction detection as a separate LLM call, made before
synthesis, whose result is written into "Open questions" by Python — not by the
synthesis model — plus the deterministic checks that ride alongside it
(fabricated figures, criteria that restate constraints, sensitive-domain gap).

The LLM is always stubbed; these tests are about wiring and rules, not model
quality.
"""
from __future__ import annotations

import pytest

from keel import engine, llm, render
from keel.models import SlotValue

PROV = llm.Provider("groq", "k", "m")
_SENTINEL = "- None — every dimension was addressed."

_SECTIONS = {
    "context": "A personal tool for one person and their own data.",
    "objective": "Build the tool described in the idea. It removes a manual step. "
                 "The person runs it on demand. That is the whole scope.",
    "io_contract": "- Input: a local file\n- Output: a structured file written beside it",
    "constraints": "- Local Python script\n- Standard library only\n- Runs offline",
    "acceptance_criteria": "- One run produces the output\n- Every item is represented\n"
                           "- The output opens cleanly",
    "non_goals": "- No web UI\n- No database\n- No accounts",
    "open_questions": _SENTINEL,
}


def _finished(template: str = "default", prompt: str = "build a small tool"):
    s = engine.start_session(prompt, template, created_date="2026-09-01")
    t = engine.load_template(template)
    engine.freeze_pending(s, t)
    for name in list(s.pending_slots):
        slot = t.slot(name)
        engine.accept_answer(s, name, slot.default_text, recommended=slot.default_text)
    engine.fill_unasked_slots(s, t, provider=None)
    return s


def _route(monkeypatch, *, conflicts=None, conflict_tuple=None, sections=None):
    """Answer the conflict call and the synthesis call distinctly."""
    payload = dict(_SECTIONS) if sections is None else sections

    def fake(system, user, **kw):
        if "contradiction checker" in system:
            if conflict_tuple is not None:
                return conflict_tuple
            return {"conflicts": conflicts or []}, None
        return payload if isinstance(payload, tuple) else (payload, None)

    monkeypatch.setattr("keel.llm.complete_json", fake)


def _oq(md: str) -> str:
    return md.split("## Open questions", 1)[1].split("\n---", 1)[0]


# --------------------------------------------------------------------------- #
# check_conflicts itself
# --------------------------------------------------------------------------- #
def test_check_conflicts_happy_path_returns_a_normalised_list(monkeypatch):
    _route(monkeypatch, conflicts=[
        {"slots": ["constraints", "objective"],
         "conflict": "Offline stdlib-only cannot do AI.",
         "suggested_resolution": "Permit a local model or drop the AI claim."},
        {"conflict": "missing slots key is tolerated"},
        "garbage-not-a-dict",
        {"slots": [], "conflict": "   "},
    ])
    conflicts, err = render.check_conflicts(_finished(), provider=PROV)
    assert err is None
    assert [c["conflict"] for c in conflicts] == [
        "Offline stdlib-only cannot do AI.",
        "missing slots key is tolerated",
    ]
    assert conflicts[0]["slots"] == ["constraints", "objective"]
    assert conflicts[1]["slots"] == []


def test_check_conflicts_failure_is_surfaced_not_swallowed(monkeypatch):
    _route(monkeypatch, conflict_tuple=(None, "RateLimitError: slow down"))
    s = _finished()
    conflicts, err = render.check_conflicts(s, provider=PROV)
    assert conflicts == [] and err == "RateLimitError: slow down"
    assert s.degraded is True


def test_check_conflicts_counts_against_the_session_call_cap(monkeypatch):
    _route(monkeypatch, conflicts=[])
    s = _finished()
    render.check_conflicts(s, provider=PROV)
    assert s.call_count == 1


def test_check_conflicts_is_refused_once_the_cap_is_hit(monkeypatch):
    _route(monkeypatch, conflicts=[])
    s = _finished()
    s.call_count = engine.MAX_LLM_CALLS_PER_SESSION
    conflicts, err = render.check_conflicts(s, provider=PROV)
    assert conflicts == [] and "limit reached" in err


def test_conflict_system_prompt_is_check_only_and_covers_premise_drift():
    p = render._CONFLICT_SYSTEM
    low = p.lower()
    assert "contradiction checker" in low
    assert "premise drift" in low
    assert "you do not resolve them" in low
    assert "original idea" in low
    assert '{"conflicts": []}' in p


# --------------------------------------------------------------------------- #
# conflicts -> Open questions, mechanically
# --------------------------------------------------------------------------- #
def test_detected_conflicts_are_written_into_open_questions_by_python(monkeypatch):
    conflicts = [{
        "slots": ["constraints", "objective"],
        "conflict": "An offline standard-library-only build cannot produce "
                    "AI-generated interpretations.",
        "suggested_resolution": "Allow a local model dependency, or drop the "
                                "AI interpretation.",
    }]
    _route(monkeypatch, conflicts=conflicts)
    s = _finished()
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    oq = _oq(md)
    assert "AI-generated interpretations" in oq
    assert "Suggested resolution" in oq
    assert _SENTINEL not in oq


def test_open_questions_is_never_only_none_when_a_slot_was_skipped(monkeypatch):
    _route(monkeypatch, conflicts=[])
    s = _finished()
    first = engine.load_template("default").required_slots()[0].name
    s.slots[first] = SlotValue(value="", source="skipped")
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    oq = _oq(md)
    assert _SENTINEL not in oq
    assert "left unanswered" in oq


def test_conflict_check_failure_is_noted_in_the_document(monkeypatch):
    def fake(system, user, **kw):
        if "contradiction checker" in system:
            return None, "TimeoutError: checker timed out"
        return dict(_SECTIONS), None

    monkeypatch.setattr("keel.llm.complete_json", fake)
    s = _finished()
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    oq = _oq(md)
    assert "Conflict check unavailable" in oq
    assert "TimeoutError: checker timed out" in oq


# --------------------------------------------------------------------------- #
# deterministic checks that ride alongside
# --------------------------------------------------------------------------- #
def test_fabricated_numeric_threshold_in_acceptance_is_flagged(monkeypatch):
    sections = dict(
        _SECTIONS,
        acceptance_criteria="- Classifier confidence is >= 0.9 for every reading\n"
                            "- One run produces the output file",
    )
    _route(monkeypatch, conflicts=[], sections=sections)
    s = _finished(prompt="classify my sensor readings")
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    oq = _oq(md)
    assert "Unverified figure" in oq and "0.9" in oq


def test_acceptance_criterion_restating_a_non_goal_is_flagged(monkeypatch):
    sections = dict(
        _SECTIONS,
        non_goals="- No authentication is implemented\n- No database\n- No web UI",
        acceptance_criteria="- No authentication is implemented\n"
                            "- One run produces the output file",
    )
    _route(monkeypatch, conflicts=[], sections=sections)
    s = _finished()
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    assert "Non-testable criteria" in _oq(md)


def test_sensitive_domain_without_a_disclaimer_is_flagged(monkeypatch):
    _route(monkeypatch, conflicts=[])
    s = _finished(prompt="app to track my blood pressure and flag concerning trends")
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert "Sensitive domain" in _oq(md)


def test_sensitive_domain_with_a_disclaimer_is_not_flagged(monkeypatch):
    sections = dict(
        _SECTIONS,
        constraints=_SECTIONS["constraints"]
        + "\n- Output states it is for informational purposes only, not medical advice",
    )
    _route(monkeypatch, conflicts=[], sections=sections)
    s = _finished(prompt="app to track my blood pressure")
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert "Sensitive domain" not in _oq(md)


# --------------------------------------------------------------------------- #
# the frozen regression fixture Phase 1 ships against
# --------------------------------------------------------------------------- #
def test_health_monitoring_fixture_surfaces_offline_ai_and_dropped_chat(monkeypatch):
    conflicts = [
        {"slots": ["constraints", "objective"],
         "conflict": "An offline, Python-standard-library-only implementation cannot "
                     "produce AI-generated interpretations of health data.",
         "suggested_resolution": "Either permit a local model dependency, or reduce "
                                 "the output to rule-based interpretation."},
        {"slots": ["original idea", "io_contract"],
         "conflict": "The idea describes a chat bot, but no answer defines a "
                     "conversation interface, session, or message history.",
         "suggested_resolution": "Decide whether a conversational interface is in "
                                 "scope, or the tool is a single stateless call."},
    ]
    _route(monkeypatch, conflicts=conflicts)
    s = _finished(prompt="AI chat bot for health monitoring")
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    oq = _oq(md)
    assert _SENTINEL not in oq
    assert "AI-generated interpretations" in oq
    assert "conversation interface" in oq
    assert "Sensitive domain" in oq
