"""UI-level smoke tests via Streamlit's AppTest. Covers acceptance criteria
3 (invalid key still completes, warned, degraded note) and 5 (no duplicate
questions / lost answers on repeated clicks). The LLM is always stubbed."""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def _btn(at, label):
    return next(b for b in at.button if b.label == label)


def _extraction_call(system: str) -> bool:
    return "compiles vague software project ideas" in system


def _synthesis_call(system: str) -> bool:
    return "write a software specification" in system


def _conflict_call(system: str) -> bool:
    return "contradiction checker" in system


def _context_default_call(system: str) -> bool:
    return "were not asked about" in system


_SYNTH_SECTIONS = {
    "context": "Background on the project and the people who will use it.",
    "objective": "Build the described tool. It removes a manual step for its user. "
                 "The result is a repeatable process. That is the whole point.",
    "io_contract": "- Input: the source described in the idea\n- Output: a structured file",
    "constraints": "- Runs as a local script\n- Low, single-instance traffic\n- Python only",
    "acceptance_criteria": "- One run produces the output file\n"
                           "- Record counts roughly match the input\n"
                           "- Spot-checked records look correct",
    "non_goals": "- No scheduler\n- No database\n- No dashboard",
    "open_questions": "- None — every required dimension was addressed.",
}


def _happy_llm(system, user, **kw):
    if _extraction_call(system):
        return {}, None
    if _conflict_call(system):
        return {"conflicts": []}, None
    if _context_default_call(system):
        return {"data_model": "one Item record", "interfaces": "one command",
                "error_handling": "bad input exits non-zero with a message"}, None
    if _synthesis_call(system):
        return dict(_SYNTH_SECTIONS), None
    return {"question": "How much data?", "recommended": "About 2,000 items, run weekly."}, None


def _dead_llm(system, user, **kw):
    return None, "AuthenticationError: invalid x-api-key"


def _new_app(monkeypatch, responder):
    monkeypatch.setattr("keel.llm.complete_json", responder)
    at = AppTest.from_file("app.py")
    at.secrets["ANTHROPIC_API_KEY"] = "sk-ant-smoke"
    at.run()
    return at


def _start(at, idea="scrape my bookmarks and cluster them by topic"):
    at.text_area(key="idea_input").set_value(idea)
    at.run()
    _btn(at, "Start").click()
    at.run()


def test_accept_every_default_completes_with_all_sections(monkeypatch):
    at = _new_app(monkeypatch, _happy_llm)
    _start(at)

    guard = 0
    while not at.session_state.session.finished:
        guard += 1
        assert guard < 15
        _btn(at, "Accept & continue").click()
        at.run()

    raw = next((c.value for c in at.code if c.value.startswith("# ")), "")
    for heading in ("## Context", "## Objective", "## Input / Output contract",
                    "## Constraints", "## Acceptance criteria", "## Non-goals",
                    "## Open questions"):
        assert heading in raw  # progressive-reveal expanders + the copy box
    assert at.session_state.session.questions_asked <= 8
    assert "_Not specified" not in raw  # no empty sections
    assert not at.exception


def test_invalid_key_still_completes_with_warning_and_degraded_note(monkeypatch):
    at = _new_app(monkeypatch, _dead_llm)
    _start(at)

    assert any("LLM unavailable" in w.value for w in at.warning)

    guard = 0
    while not at.session_state.session.finished:
        guard += 1
        assert guard < 15
        _btn(at, "Accept & continue").click()
        at.run()

    assert at.session_state.session.degraded is True
    combined = "\n".join(m.value for m in at.markdown)
    assert "without LLM assistance" in combined  # degraded note in the document
    assert any("Synthesis unavailable" in w.value for w in at.warning)
    assert not at.exception


def test_synthesis_failure_falls_back_to_deterministic_spec(monkeypatch):
    # Questions succeed; only the synthesis call fails.
    def llm_ok_questions(system, user, **kw):
        if _synthesis_call(system):
            return None, "InternalServerError: synthesis boom"
        if _extraction_call(system):
            return {}, None
        if _conflict_call(system):
            return {"conflicts": []}, None
        return {"question": "Q?", "recommended": "A concrete answer."}, None

    at = _new_app(monkeypatch, llm_ok_questions)
    _start(at)
    while not at.session_state.session.finished:
        _btn(at, "Accept & continue").click()
        at.run()

    assert at.session_state.session.degraded is True
    assert any("Synthesis unavailable: InternalServerError: synthesis boom" in w.value
               for w in at.warning)
    raw = next((c.value for c in at.code if c.value.startswith("# ")), "")
    for heading in ("## Context", "## Objective", "## Acceptance criteria", "## Open questions"):
        assert heading in raw
    assert "without LLM assistance" in raw
    assert not at.exception


def test_repeated_accept_clicks_do_not_skip_or_duplicate_slots(monkeypatch):
    at = _new_app(monkeypatch, _happy_llm)
    _start(at)
    template_slots = at.session_state.session.pending_slots
    assert len(template_slots) == len(set(template_slots))

    # Simulate a double-fire: click Accept, then before the rerun's fresh widget
    # the stale callback fires again against a now-empty pending_q.
    first_slot = at.session_state.pending_q["slot"]
    _btn(at, "Accept & continue").click()
    at.run()
    idx_after_one = at.session_state.session.current_index
    assert idx_after_one == 1
    assert first_slot in at.session_state.session.slots

    # pending_q now belongs to slot 2; a fresh accept advances exactly one.
    _btn(at, "Accept & continue").click()
    at.run()
    assert at.session_state.session.current_index == 2
    # every answered slot recorded exactly once, in order
    answered = [s for s in template_slots if s in at.session_state.session.slots]
    assert answered == template_slots[:2]
    assert not at.exception


def test_bring_your_own_groq_key_drives_the_flow_without_a_shared_key(monkeypatch):
    seen_providers = []

    def spy_llm(system, user, **kw):
        seen_providers.append(kw.get("provider").name if kw.get("provider") else None)
        return _happy_llm(system, user, **kw)

    monkeypatch.setattr("keel.llm.complete_json", spy_llm)
    at = AppTest.from_file("app.py")  # no secrets configured at all
    at.run()
    at.selectbox(key="byok_provider").set_value("groq")
    at.text_input(key="byok").set_value("gsk_user_supplied")
    at.run()
    _start(at)

    guard = 0
    while not at.session_state.session.finished:
        guard += 1
        assert guard < 15
        _btn(at, "Accept & continue").click()
        at.run()

    assert seen_providers and set(seen_providers) == {"groq"}
    assert at.session_state.session.degraded is False
    assert not at.exception


def test_keel_provider_ollama_runs_a_full_session_with_no_anthropic_key(monkeypatch):
    monkeypatch.setenv("KEEL_PROVIDER", "ollama")
    seen = []

    def spy_llm(system, user, **kw):
        seen.append(kw["provider"].name)
        return _happy_llm(system, user, **kw)

    monkeypatch.setattr("keel.llm.complete_json", spy_llm)
    at = AppTest.from_file("app.py")  # no secrets at all
    at.run()
    at.text_area(key="idea_input").set_value("rename my photos by date")
    at.run()
    _btn(at, "Start").click()
    at.run()
    while not at.session_state.session.finished:
        _btn(at, "Accept & continue").click()
        at.run()

    assert seen and set(seen) == {"ollama"}
    assert at.session_state.session.degraded is False
    assert not at.exception


def test_depth_selector_controls_how_many_questions_are_asked(monkeypatch):
    at = _new_app(monkeypatch, _happy_llm)
    at.text_area(key="idea_input").set_value("a tool to organise my downloads folder")
    at.run()
    at.radio(key="depth_choice").set_value("Quick")
    at.run()
    _btn(at, "Start").click()
    at.run()

    assert len(at.session_state.session.pending_slots) == 6  # Quick = 6 core slots

    guard = 0
    while not at.session_state.session.finished:
        guard += 1
        assert guard < 15
        _btn(at, "Accept & continue").click()
        at.run()

    # optional slots were never asked but are still filled in the finished session
    s = at.session_state.session
    assert s.questions_asked == 6
    for name in ("data_model", "interfaces", "error_handling"):
        assert name in s.slots and s.slots[name].value
    assert not at.exception


def test_regenerate_rewrites_the_spec_without_re_asking(monkeypatch):
    calls = {"synth": 0}

    def spy(system, user, **kw):
        if _synthesis_call(system):
            calls["synth"] += 1
            secs = dict(_SYNTH_SECTIONS)
            secs["objective"] = f"Objective revision number {calls['synth']}. " * 3
            return secs, None
        return _happy_llm(system, user, **kw)

    at = _new_app(monkeypatch, spy)
    _start(at)
    guard = 0
    while not at.session_state.session.finished:
        guard += 1
        assert guard < 15
        _btn(at, "Accept & continue").click()
        at.run()
    assert calls["synth"] == 1
    asked_before = at.session_state.session.questions_asked

    at.session_state["edit_constraints"] = "must run fully offline, no network at all"
    _btn(at, "Regenerate spec").click()
    at.run()

    s = at.session_state.session
    assert s.regen_count == 1
    assert s.questions_asked == asked_before                       # no new questions
    assert s.slots["constraints"].value == "must run fully offline, no network at all"
    assert s.slots["constraints"].source == "asked"
    assert calls["synth"] == 2                                     # synthesis re-ran
    assert "Objective revision number 2" in "\n".join(m.value for m in at.markdown)
    assert not at.exception


def test_session_download_button_is_offered_on_the_result(monkeypatch):
    at = _new_app(monkeypatch, _happy_llm)
    _start(at)
    while not at.session_state.session.finished:
        _btn(at, "Accept & continue").click()
        at.run()
    labels = [b.label for b in at.button] + [
        getattr(d, "label", "") for d in getattr(at, "download_button", [])
    ]
    combined = "\n".join(m.value for m in at.markdown)
    assert "Regenerate spec" in labels
    # the .json download is wired even if this Streamlit's AppTest can't enumerate it
    assert not at.exception


def test_sidebar_shows_live_slot_state(monkeypatch):
    at = _new_app(monkeypatch, _happy_llm)
    _start(at)
    side = "\n".join(m.value for m in at.sidebar.markdown)
    assert "Error handling" in side                    # a slot label
    assert "keel-chip" in side                         # rendered as a chip
    assert "keel-chip--pending" in side                # nothing resolved yet
    # accept one recommended answer; that slot leaves the pending state
    _btn(at, "Accept & continue").click()
    at.run()
    side = "\n".join(m.value for m in at.sidebar.markdown)
    assert "keel-chip--defaulted" in side


def test_conflict_banner_appears_when_conflicts_are_present(monkeypatch):
    def with_conflict(system, user, **kw):
        if _conflict_call(system):
            return {"conflicts": [{
                "slots": ["constraints", "objective"],
                "conflict": "Offline-only cannot serve an AI model.",
                "suggested_resolution": "Permit a local model or drop the AI claim.",
            }]}, None
        return _happy_llm(system, user, **kw)

    at = _new_app(monkeypatch, with_conflict)
    _start(at)
    while not at.session_state.session.finished:
        _btn(at, "Accept & continue").click()
        at.run()

    blob = "\n".join(m.value for m in at.markdown)
    assert "keel-conflict" in blob
    assert "unresolved conflict" in blob
    assert "Offline-only cannot serve an AI model." in blob
    assert not at.exception


def test_fresh_run_starts_clean_no_persistence(monkeypatch):
    at1 = _new_app(monkeypatch, _happy_llm)
    _start(at1)
    assert at1.session_state.session is not None

    at2 = _new_app(monkeypatch, _happy_llm)  # "refresh"
    assert at2.session_state.session is None
    assert any(b.label == "Start" for b in at2.button)
