"""Part A — A3 + A6: the ``keel_decided`` source, the "Decisions Keel made for
you" section, and the honest progress breakdown.

"Decide for me" is a distinct intention from "skip": the user delegated the
choice, so the value carries a one-line reason and a "revisit this if …" clause,
and it lands in its own section — never in "Open questions", never counted as a
dimension the user resolved.
"""
from __future__ import annotations

import app
from keel import engine, llm, render, session_io
from keel.models import SessionState, SlotValue

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
    "open_questions": "- None — every required dimension was addressed.",
}


def _finished(prompt="build a small tool", template="default"):
    s = engine.start_session(prompt, template, created_date="2026-09-01")
    t = engine.load_template(template)
    engine.freeze_pending(s, t)
    for name in list(s.pending_slots):
        slot = t.slot(name)
        engine.accept_answer(s, name, slot.default_text, recommended=slot.default_text)
    engine.fill_unasked_slots(s, t, provider=None)
    return s


def _stub(monkeypatch, payload):
    def fake(system, user, **kw):
        if "contradiction checker" in system:
            return {"conflicts": []}, None
        return (payload, None)
    monkeypatch.setattr("keel.llm.complete_json", fake)


# --------------------------------------------------------------------------- #
# decide_for_me and the source
# --------------------------------------------------------------------------- #
def test_decide_for_me_records_value_reason_and_revisit_condition():
    s = engine.start_session("x", "default", created_date="2026-09-01")
    t = engine.load_template("default")
    engine.freeze_pending(s, t)
    first = s.pending_slots[0]

    engine.decide_for_me(
        s, first, "server-rendered pages, one process",
        rationale="a single small audience does not need a single-page app",
        revisit_if="the UI needs to update without full page reloads",
    )
    v = s.slots[first]
    assert v.source == "keel_decided"
    assert v.value == "server-rendered pages, one process"
    assert v.rationale and v.revisit_if
    assert first in s.decided_slots()
    assert first not in s.skipped_slots()
    assert s.current_index == 1  # it advanced


def test_next_question_carries_rationale_and_revisit_from_one_call(make_session, stub_llm):
    stub_llm((
        {"question": "How is it served?", "recommended": "one server process",
         "rationale": "smallest thing that works for a private tool",
         "revisit_if": "you expect public traffic"},
        None,
    ))
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)

    p = engine.next_question(session, template, provider=PROV)
    assert p.rationale == "smallest thing that works for a private tool"
    assert p.revisit_if == "you expect public traffic"
    assert session.call_count == 1  # no extra call for the rationale


# --------------------------------------------------------------------------- #
# the Decisions section
# --------------------------------------------------------------------------- #
def test_decisions_section_omitted_when_nothing_was_delegated(monkeypatch):
    _stub(monkeypatch, dict(_SECTIONS))
    md, err = render.synthesize_spec(_finished(), provider=PROV)
    assert err is None
    assert "## Decisions Keel made for you" not in md


def test_decisions_section_present_with_a_reason_and_a_revisit_clause(monkeypatch):
    _stub(monkeypatch, dict(_SECTIONS))
    s = _finished()
    s.slots["runtime"] = SlotValue(
        value="one server process, server-rendered pages",
        source="keel_decided",
        rationale="a private tool does not need a separate frontend build",
        revisit_if="the app must serve public traffic",
    )
    md, err = render.synthesize_spec(s, provider=PROV)
    assert err is None

    assert "## Decisions Keel made for you" in md
    dec = md.split("## Decisions Keel made for you", 1)[1].split("\n## ", 1)[0]
    assert "one server process, server-rendered pages" in dec
    assert "a private tool does not need a separate frontend build" in dec
    assert "Revisit this if" in dec and "public traffic" in dec

    # position: after Acceptance criteria, before Non-goals
    order = [ln for ln in md.splitlines() if ln.startswith("## ")]
    assert order.index("## Decisions Keel made for you") == order.index("## Acceptance criteria") + 1
    assert order.index("## Non-goals") == order.index("## Decisions Keel made for you") + 1


def test_delegated_slot_does_not_appear_in_open_questions(monkeypatch):
    _stub(monkeypatch, dict(_SECTIONS))
    s = _finished()
    s.slots["runtime"] = SlotValue(
        value="one server process", source="keel_decided",
        rationale="smallest viable choice", revisit_if="public traffic is expected",
    )
    md, err = render.synthesize_spec(s, provider=PROV)
    assert err is None
    oq = md.split("## Open questions", 1)[1]
    assert "smallest viable choice" not in oq
    assert "one server process" not in oq


def test_fallback_renderer_also_emits_the_decisions_section():
    s = _finished()
    s.slots["runtime"] = SlotValue(
        value="one server process", source="keel_decided",
        rationale="smallest viable choice", revisit_if="public traffic is expected",
    )
    md = render.render_markdown(s)
    assert "## Decisions Keel made for you" in md
    assert "smallest viable choice" in md


def test_ceiling_note_is_in_the_footer_once(monkeypatch):
    _stub(monkeypatch, dict(_SECTIONS))
    md, err = render.synthesize_spec(_finished(), provider=PROV)
    assert err is None
    assert md.count("benefits from review by someone with software experience") == 1


# --------------------------------------------------------------------------- #
# A6: the progress breakdown is never a bare n / n
# --------------------------------------------------------------------------- #
def test_progress_breakdown_splits_answered_decided_skipped():
    s = engine.start_session("x", "default", created_date="2026-09-01")
    t = engine.load_template("default")
    slots = sorted(t.slots, key=lambda x: (x.priority, x.name))
    s.slots[slots[0].name] = SlotValue(value="I typed this", source="asked")
    s.slots[slots[1].name] = SlotValue(value="from the idea", source="extracted")
    s.slots[slots[2].name] = SlotValue(value="Keel chose", source="keel_decided",
                                       rationale="r", revisit_if="c")
    s.slots[slots[3].name] = SlotValue(value="Keel suggested", source="llm_default")
    s.slots[slots[4].name] = SlotValue(value="", source="skipped")

    line = app._progress_breakdown(s, slots)
    assert line == "2 answered · 2 Keel decided · 1 skipped · 4 pending"
    assert "/" not in line  # never a bare n / n


def test_answered_minority_flags_a_mostly_defaulted_spec():
    s = engine.start_session("x", "default", created_date="2026-09-01")
    t = engine.load_template("default")
    slots = sorted(t.slots, key=lambda x: (x.priority, x.name))
    for slot in slots:
        s.slots[slot.name] = SlotValue(value="v", source="keel_decided",
                                       rationale="r", revisit_if="c")
    s.slots[slots[0].name] = SlotValue(value="mine", source="asked")
    assert app._answered_minority(s, slots) is True

    for slot in slots:
        s.slots[slot.name] = SlotValue(value="mine", source="asked")
    assert app._answered_minority(s, slots) is False
