"""Phase 0: provenance split (Bug 1) and conflict re-validation (Bug 2).

Bug 1 — ``defaulted`` conflated an accepted LLM suggestion with a static template
fallback, so the sidebar and the degraded banner lied about a document full of
contextual detail. The source is now split, and ``SessionState.degraded`` is
derived: it fires only for a genuine fallback (a ``template_default`` slot, or a
failed synthesis / conflict call), never for a transient earlier error.

Bug 2 — the conflict list was computed against the answers and never rechecked,
so a stale "there is no User entity" survived next to a document containing one.
Synthesis now reports what it resolved, and each pre-synthesis conflict is
re-validated against the rendered document before it can reach Open questions.
"""
from __future__ import annotations

import pytest

from keel import engine, llm, render
from keel.models import SessionState, SlotValue

PROV = llm.Provider("groq", "k", "m")
_SENTINEL = "- None — every required dimension was addressed."

_SECTIONS = {
    "context": "A personal tool for one person and their own data.",
    "objective": "Build the tool described in the idea. It removes a manual step. "
                 "The person runs it on demand. That is the whole scope.",
    "io_contract": "- Input: a local file\n- Output: a structured file beside it",
    "constraints": "- Local Python script\n- Standard library only\n- Runs offline",
    "acceptance_criteria": "- One run produces the output\n- Every item is represented\n"
                           "- The output opens cleanly",
    "non_goals": "- No web UI\n- No database\n- No accounts",
    "open_questions": _SENTINEL,
}


def _oq(md: str) -> str:
    return md.split("## Open questions", 1)[1].split("\n---", 1)[0]


# --------------------------------------------------------------------------- #
# Bug 1: the source split and the derived degraded flag
# --------------------------------------------------------------------------- #
def test_accepting_a_suggestion_is_llm_default_not_a_fallback(make_session):
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)

    engine.accept_answer(session, session.pending_slots[0], "the model's line",
                         recommended="the model's line")
    engine.accept_answer(session, session.pending_slots[1], "",
                         recommended="the static hint",
                         recommended_source="template_default")

    assert session.slots[session.pending_slots[0]].source == "llm_default"
    assert session.slots[session.pending_slots[1]].source == "template_default"


def test_degraded_is_derived_and_read_only():
    session = SessionState(original_prompt="x", template_name="default",
                           created_date="2026-09-01")
    assert session.degraded is False
    with pytest.raises(AttributeError):
        session.degraded = True  # derived from state, never set directly


def test_session_with_no_failed_calls_never_degrades(stub_llm, make_session):
    """A full run where every LLM call succeeds must not render the degraded
    banner — the motivating bug was a sticky flag over a good document."""
    def ok(system, user, **kw):
        if "compiles vague software project ideas" in system:
            return {}, None
        if "contradiction checker" in system:
            return {"conflicts": []}, None
        if "were not asked about" in system:
            return {"data_model": "one Record: id, name", "interfaces": "one command",
                    "error_handling": "bad input exits non-zero"}, None
        if "write a software specification" in system:
            return dict(_SECTIONS, resolved_conflicts=[]), None
        return {"question": "How many?", "recommended": "About a thousand, run weekly."}, None

    stub_llm(ok)
    session = make_session("dedupe my contacts export", "default")
    template = engine.load_template("default")
    engine.extract_prefilled(session, template, provider=PROV)
    engine.freeze_pending(session, template)
    while not session.finished:
        slot = engine.current_slot(session, template)
        p = engine.next_question(session, template, provider=PROV)
        q, rec, err = p.question, p.recommended, p.error
        engine.accept_answer(session, slot.name, rec, recommended=rec,
                             recommended_source="template_default" if err else "llm_default")
    engine.fill_unasked_slots(session, template, provider=PROV)

    assert not any(v.source == "template_default" for v in session.slots.values())
    assert session.degraded is False
    conflicts, cerr = render.check_conflicts(session, provider=PROV)
    md, serr = render.synthesize_spec(session, provider=PROV,
                                      conflicts=conflicts, conflict_error=cerr)
    assert serr is None
    assert session.degraded is False
    assert "without LLM assistance" not in md


def test_one_template_default_slot_is_enough_to_degrade(make_session):
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    for name in session.pending_slots:
        engine.accept_answer(session, name, "an answer", recommended="an answer")
    assert session.degraded is False
    engine.fill_unasked_slots(session, template, provider=None)  # static-fills the rest
    assert any(v.source == "template_default" for v in session.slots.values())
    assert session.degraded is True


# --------------------------------------------------------------------------- #
# Bug 2: conflicts re-validated against the document synthesis produced
# --------------------------------------------------------------------------- #
def _session_with_conflicts(conflicts: list[dict], prompt="build a small tool") -> SessionState:
    s = engine.start_session(prompt, "default", created_date="2026-09-01")
    t = engine.load_template("default")
    engine.freeze_pending(s, t)
    for name in list(s.pending_slots):
        slot = t.slot(name)
        engine.accept_answer(s, name, slot.default_text, recommended=slot.default_text)
    engine.fill_unasked_slots(s, t, provider=None)
    s.conflicts = list(conflicts)
    return s


def _route(monkeypatch, *, sections):
    def fake(system, user, **kw):
        if "contradiction checker" in system:
            return {"conflicts": []}, None
        return sections, None
    monkeypatch.setattr("keel.llm.complete_json", fake)


def test_conflict_the_model_reports_resolving_leaves_open_questions(monkeypatch):
    conflict = {
        "slots": ["constraints", "objective"],
        "conflict": "Offline stdlib-only cannot produce AI interpretations.",
        "suggested_resolution": "Permit a local model, or drop the AI claim.",
    }
    sections = dict(
        _SECTIONS,
        resolved_conflicts=[{
            "conflict": "offline stdlib cannot do AI interpretations",
            "how": "kept it offline; changed the output to rule-based interpretation",
        }],
    )
    _route(monkeypatch, sections=sections)
    s = _session_with_conflicts([conflict])
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=[conflict],
                                     conflict_error=None)
    assert err is None
    assert s.conflicts == []                                   # no survivors
    assert len(s.resolved_conflicts) == 1
    assert "rule-based" in s.resolved_conflicts[0]["resolution"]
    oq = _oq(md)
    assert "Offline stdlib-only" not in oq                     # not raised any more
    assert _SENTINEL in oq


def test_conflict_the_model_does_not_resolve_still_surfaces(monkeypatch):
    conflict = {
        "slots": ["constraints", "objective"],
        "conflict": "Offline stdlib-only cannot produce AI interpretations.",
        "suggested_resolution": "Permit a local model, or drop the AI claim.",
    }
    _route(monkeypatch, sections=dict(_SECTIONS, resolved_conflicts=[]))
    s = _session_with_conflicts([conflict])
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=[conflict],
                                     conflict_error=None)
    assert err is None
    assert len(s.conflicts) == 1 and s.resolved_conflicts == []
    assert "Offline stdlib-only cannot produce AI interpretations." in _oq(md)
    assert _SENTINEL not in _oq(md)


def test_premise_drift_conflict_is_dropped_once_the_document_covers_it(monkeypatch):
    """The Phase 0 'ships when': a conflict about a missing entity must not keep
    being reported once synthesis has put that entity in the document."""
    conflict = {
        "slots": ["original idea", "data_model"],
        "conflict": "No user account entity is defined, though the pages assume a "
                    "signed-in customer profile with an order history.",
        "suggested_resolution": "Decide whether accounts are in scope.",
    }
    sections = dict(
        _SECTIONS,
        io_contract=(
            "Data model: User(id, name, email); Customer profile; Order history. "
            "Pages: a signed-in customer sees their account; the entity is defined "
            "per user; each page shows the order history."
        ),
        resolved_conflicts=[],  # the model did NOT report it — the doc check must catch it
    )
    _route(monkeypatch, sections=sections)
    s = _session_with_conflicts([conflict], prompt="a small store for one seller")
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=[conflict],
                                     conflict_error=None)
    assert err is None
    assert s.conflicts == []
    assert len(s.resolved_conflicts) == 1
    assert "user account entity" not in _oq(md).lower()


def test_a_direct_value_contradiction_is_not_dropped_just_because_words_recur(monkeypatch):
    # "offline" and "model" both reappear in the prose, but the contradiction is
    # a value clash, not a missing capability — it must survive.
    conflict = {
        "slots": ["constraints", "io_contract"],
        "conflict": "The constraints say offline with no model weights, but the "
                    "output is an AI-model interpretation.",
        "suggested_resolution": "Pick one.",
    }
    sections = dict(
        _SECTIONS,
        constraints="- Runs offline\n- No model weights on disk",
        io_contract="- Output: an AI-model interpretation of the input",
        resolved_conflicts=[],
    )
    _route(monkeypatch, sections=sections)
    s = _session_with_conflicts([conflict])
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=[conflict],
                                     conflict_error=None)
    assert err is None
    assert len(s.conflicts) == 1
    assert "value clash" not in _oq(md)  # sanity
    assert "offline with no model weights" in _oq(md)
