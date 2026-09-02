"""Part B — B5 + B6: the restructured "Open questions" (three labelled
subsections) and tightened "Non-goals" (no hedges, capped at eight).
"""
from __future__ import annotations

import pathlib

from keel import engine, llm, render, session_io
from keel.models import SlotValue

PROV = llm.Provider("groq", "k", "m")
_FIX = pathlib.Path("evals/fixtures")

_HEDGES = (
    "not a focus", "beyond basic", "beyond simple", "kept minimal",
    "for the purpose of the demo", "for the demo", "to some extent",
    "light-touch", "nothing fancy",
)

_CORE = {
    "context": "A small personal tool.",
    "objective": "Do the described job. It removes a manual step. The user runs it "
                 "when needed. That is the scope.",
    "io_contract": "- Input: a file\n- Output: a file",
    "constraints": "- Local Python script\n- Standard library only",
    "acceptance_criteria": "- One run produces the output\n- Every item is represented\n"
                           "- The output opens cleanly",
    "non_goals": "- No web UI\n- No database\n- No accounts",
    "open_questions": "- None — every dimension was addressed.",
}


def _finished(prompt="build a small tool", template="default"):
    s = engine.start_session(prompt, template, created_date="2026-09-01")
    t = engine.load_template(template)
    engine.freeze_pending(s, t)
    for name in list(s.pending_slots):
        engine.accept_answer(s, name, t.slot(name).default_text,
                             recommended=t.slot(name).default_text)
    engine.fill_unasked_slots(s, t, provider=None)
    return s


def _route(monkeypatch, *, conflicts, sections):
    def fake(system, user, **kw):
        if "contradiction checker" in system:
            return {"conflicts": conflicts}, None
        return sections, None
    monkeypatch.setattr("keel.llm.complete_json", fake)


def _oq(md):
    return md.split("## Open questions", 1)[1].split("\n---", 1)[0]


# --------------------------------------------------------------------------- #
# B5: three labelled subsections
# --------------------------------------------------------------------------- #
def test_all_three_subsections_render_when_all_have_content(monkeypatch):
    conflict = {"slots": ["a", "b"], "kind": "logical",
                "conflict": "two answers cannot both hold", "suggested_resolution": "pick one"}
    sections = dict(
        _CORE, resolved_conflicts=[],
        acceptance_criteria="- One run produces the output\n- It handles up to 5000 items",
    )
    _route(monkeypatch, conflicts=[conflict], sections=sections)
    s = _finished()
    s.slots["done"] = SlotValue(value="", source="skipped")
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    oq = _oq(md)
    assert "**Unresolved conflicts**" in oq
    assert "**Not specified**" in oq
    assert "**Worth deciding before starting**" in oq
    # order
    assert oq.index("**Unresolved conflicts**") < oq.index("**Not specified**") \
        < oq.index("**Worth deciding before starting**")


def test_empty_subsections_are_omitted(monkeypatch):
    _route(monkeypatch, conflicts=[], sections=dict(_CORE, resolved_conflicts=[]))
    s = _finished()  # no conflicts, no skips, no gaps
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    oq = _oq(md)
    assert "**Unresolved conflicts**" not in oq
    assert "**Not specified**" not in oq
    assert "None — every dimension was addressed." in oq


def test_never_claims_none_next_to_a_conflict(monkeypatch):
    """Criterion 10."""
    conflict = {"slots": ["x"], "kind": "logical", "conflict": "a real clash",
                "suggested_resolution": "decide"}
    _route(monkeypatch, conflicts=[conflict], sections=dict(_CORE, resolved_conflicts=[]))
    s = _finished()
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    oq = _oq(md)
    assert "a real clash" in oq
    assert "None — every dimension" not in oq
    assert "every dimension was addressed" not in oq


def test_decided_slots_never_appear_in_open_questions(monkeypatch):
    _route(monkeypatch, conflicts=[], sections=dict(_CORE, resolved_conflicts=[]))
    s = _finished()
    s.slots["runtime"] = SlotValue(value="one process", source="keel_decided",
                                   rationale="smallest", revisit_if="public traffic")
    c, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=c, conflict_error=cerr)
    assert err is None
    assert "smallest" not in _oq(md)


# --------------------------------------------------------------------------- #
# B6: tightened non-goals
# --------------------------------------------------------------------------- #
def test_hedged_non_goals_are_dropped(monkeypatch):
    sections = dict(
        _CORE,
        non_goals="- No payments\n"
                  "- Authentication is not a focus beyond simple login/signup for the demo\n"
                  "- No design system or extensive UI styling beyond basic templates\n"
                  "- No channel-manager sync",
        resolved_conflicts=[],
    )
    _route(monkeypatch, conflicts=[], sections=sections)
    md, err = render.synthesize_spec(_finished(), provider=PROV, conflicts=[],
                                     conflict_error=None)
    assert err is None
    ng = md.split("## Non-goals", 1)[1].split("\n## ", 1)[0].lower()
    for hedge in _HEDGES:
        assert hedge not in ng
    assert "no payments" in ng and "no channel-manager sync" in ng


def test_non_goals_capped_at_eight(monkeypatch):
    many = "\n".join(f"- No feature number {chr(65 + i)}" for i in range(12))
    _route(monkeypatch, conflicts=[], sections=dict(_CORE, non_goals=many, resolved_conflicts=[]))
    md, err = render.synthesize_spec(_finished(), provider=PROV, conflicts=[],
                                     conflict_error=None)
    assert err is None
    ng = md.split("## Non-goals", 1)[1].split("\n## ", 1)[0]
    assert len([ln for ln in ng.splitlines() if ln.strip().startswith("-")]) <= 8


def test_no_fixture_output_contains_a_hedged_non_goal():
    """Criterion 11: over the frozen fixtures, the tightened non-goals body has
    no hedging phrase."""
    for fx in sorted(_FIX.glob("*.json")):
        session, err = session_io.loads(fx.read_text("utf-8"))
        assert err is None
        template = engine.load_template(session.template_name)
        ng = session.slots.get("non_goals")
        raw = ng.value if ng else ""
        tightened = render._tighten_non_goals(raw).lower()
        for hedge in _HEDGES:
            assert hedge not in tightened, f"{fx.name}: {hedge!r}"


def test_synthesis_prompt_forbids_hedged_non_goals():
    p = render._SYNTHESIS_SYSTEM.lower()
    assert "reject hedges" in p
    assert "at most 8" in p or "at most eight" in p
