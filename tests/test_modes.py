"""Part A — A5: audience mode. Guided asks only the six core slots, ignores the
depth selector, and leans on "Decide for me" for everything else; Technical keeps
today's depth-driven free-text flow.
"""
from __future__ import annotations

from keel import engine, llm, render
from keel.models import SlotValue

PROV = llm.Provider("groq", "k", "m")

_SECTIONS = {
    "context": "A personal tool for one person and their own data.",
    "objective": "Build the tool described in the idea. It removes a manual step. "
                 "The person runs it on demand. That is the whole scope.",
    "io_contract": "- Input: a local file\n- Output: a structured file beside it",
    "constraints": "- Local Python script\n- Standard library only\n- Runs offline",
    "acceptance_criteria": "- One run produces the output\n- Every item is represented\n"
                           "- The output opens cleanly",
    "non_goals": "- No web UI\n- No database\n- No accounts",
    "open_questions": "- None — every dimension was addressed.",
}


# --------------------------------------------------------------------------- #
# slot selection
# --------------------------------------------------------------------------- #
def test_guided_asks_only_the_six_core_slots_and_ignores_depth():
    for depth in ("quick", "standard", "thorough"):
        s = engine.start_session("x", "default", created_date="2026-09-01",
                                 depth=depth, mode="guided")
        t = engine.load_template("default")
        names = [slot.name for slot in engine.askable_slots(s, t)]
        assert set(names) == set(engine.GUIDED_SLOTS)
        assert len(names) == 6


def test_technical_keeps_the_depth_driven_set():
    t = engine.load_template("default")
    counts = {}
    for depth in ("quick", "standard", "thorough"):
        s = engine.start_session("x", "default", created_date="2026-09-01",
                                 depth=depth, mode="technical")
        counts[depth] = len(engine.askable_slots(s, t))
    assert counts["quick"] == 6
    assert counts["standard"] == 8
    assert counts["thorough"] == 9


def test_engine_start_session_is_mode_neutral_and_validates_mode():
    # the engine defaults Technical (back-compat); app.py passes Guided for new
    # sessions. A bad value falls back to technical, never crashes.
    assert engine.start_session("x", "default", created_date="2026-09-01").mode == "technical"
    assert engine.start_session("x", "default", created_date="2026-09-01",
                                mode="guided").mode == "guided"
    assert engine.start_session("x", "default", created_date="2026-09-01",
                                mode="nonsense").mode == "technical"


# --------------------------------------------------------------------------- #
# Criterion 1: a Guided run needs no typing beyond the idea
# --------------------------------------------------------------------------- #
def test_guided_clickthrough_reaches_a_complete_spec_with_no_typing(monkeypatch):
    def fake(system, user, **kw):
        if "compiles vague software project ideas" in system:
            return {}, None
        if "contradiction checker" in system:
            return {"conflicts": []}, None
        if "were not asked about" in system:
            return {
                "data_model": {"value": "one Room record and one Booking record",
                               "rationale": "smallest model the idea needs",
                               "revisit_if": "you need per-guest accounts"},
                "interfaces": {"value": "a search page, a booking form, a confirmation page",
                               "rationale": "the minimum to complete a booking",
                               "revisit_if": "staff need an admin view"},
                "error_handling": {"value": "invalid dates re-render the form with an inline error",
                                   "rationale": "standard web-form behaviour",
                                   "revisit_if": "you add payment steps"},
            }, None
        if "write a software specification" in system:
            return dict(_SECTIONS, resolved_conflicts=[]), None
        return {
            "question": "How is it served?",
            "recommended": "one small server process, server-rendered pages",
            "rationale": "a small hotel does not need a single-page app",
            "revisit_if": "the calendar must update without a page reload",
        }, None

    monkeypatch.setattr("keel.llm.complete_json", fake)

    s = engine.start_session("a website where people can book rooms at my hotel",
                             "web-app", created_date="2026-09-01", mode="guided")
    t = engine.load_template("web-app")
    engine.extract_prefilled(s, t, provider=PROV)
    engine.freeze_pending(s, t)
    assert len(s.pending_slots) == 6

    # every step: press "Decide for me" — no free text typed
    while not s.finished:
        slot = engine.current_slot(s, t)
        p = engine.next_question(s, t, provider=PROV)
        engine.decide_for_me(s, slot.name, p.recommended,
                             rationale=p.rationale, revisit_if=p.revisit_if)
    engine.fill_unasked_slots(s, t, provider=PROV)

    # complete: every template slot is filled
    for slot in t.slots:
        assert s.slots.get(slot.name) and s.slots[slot.name].value

    conflicts, cerr = render.check_conflicts(s, provider=PROV)
    md, serr = render.synthesize_spec(s, provider=PROV, conflicts=conflicts,
                                      conflict_error=cerr)
    assert serr is None
    assert "## Decisions Keel made for you" in md
    dec = md.split("## Decisions Keel made for you", 1)[1].split("\n## ", 1)[0]
    # each of the six delegated slots shows a reason and a revisit clause
    assert dec.count("Revisit this if") >= 6


def test_decide_for_me_and_skip_are_recorded_differently():
    s = engine.start_session("x", "default", created_date="2026-09-01", mode="guided")
    t = engine.load_template("default")
    engine.freeze_pending(s, t)
    a, b = s.pending_slots[0], s.pending_slots[1]
    engine.decide_for_me(s, a, "a chosen value", rationale="why", revisit_if="when")
    engine.skip_slot(s, b)
    assert s.slots[a].source == "keel_decided" and s.slots[a].rationale == "why"
    assert s.slots[b].source == "skipped"
    assert a in s.decided_slots() and a not in s.skipped_slots()
    assert b in s.skipped_slots() and b not in s.decided_slots()
