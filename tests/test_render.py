from __future__ import annotations

from keel import engine
from keel.models import SessionState, SlotValue
from keel.render import render_markdown

HEADINGS = [
    "## Context",
    "## Objective",
    "## Input / Output contract",
    "## Constraints",
    "## Acceptance criteria",
    "## Non-goals",
    "## Open questions",
]


def _session_all_defaulted(template_name: str, prompt: str = "build a widget") -> SessionState:
    template = engine.load_template(template_name)
    session = engine.start_session(prompt, template_name, created_date="2026-09-01")
    session.slots = {
        s.name: SlotValue(value=s.default_text, source="defaulted")
        for s in template.slots
    }
    session.finished = True
    return session


def test_all_seven_sections_present_and_in_order():
    md = render_markdown(_session_all_defaulted("default"))
    positions = [md.index(h) for h in HEADINGS]
    assert positions == sorted(positions)
    assert md.startswith("# Build a widget")


def test_no_section_is_empty_when_every_default_is_accepted():
    for name in engine.list_templates():
        md = render_markdown(_session_all_defaulted(name))
        blocks = md.split("## ")
        for block in blocks[1:]:
            heading, _, body = block.partition("\n")
            assert body.strip(), f"{name}: section {heading!r} rendered empty"


def test_default_strategy_never_leaks_into_any_rendered_spec():
    for name in engine.list_templates():
        template = engine.load_template(name)
        md = render_markdown(_session_all_defaulted(name))
        for slot in template.slots:
            assert slot.default_strategy not in md, f"{name}.{slot.name} leaked default_strategy"


def test_skipped_slot_shows_placeholder_and_appears_in_open_questions():
    template = engine.load_template("default")
    session = _session_all_defaulted("default")
    # skip every slot mapped to the io_contract section so the section has nothing
    for s in template.slots:
        if s.section == "io_contract":
            session.slots[s.name] = SlotValue(value="", source="skipped")
    md = render_markdown(session)

    io_block = md.split("## Input / Output contract\n", 1)[1].split("## ", 1)[0]
    assert "see Open questions" in io_block

    oq_block = md.split("## Open questions\n", 1)[1]
    assert template.slot("io_contract").label in oq_block


def test_degraded_session_carries_the_no_llm_note_in_open_questions():
    session = _session_all_defaulted("default")
    session.degraded = True
    md = render_markdown(session)
    oq_block = md.split("## Open questions\n", 1)[1]
    assert "without LLM assistance" in oq_block


def test_open_questions_present_even_when_nothing_skipped():
    md = render_markdown(_session_all_defaulted("default"))
    oq_block = md.split("## Open questions\n", 1)[1]
    assert "None — every required dimension" in oq_block


def test_title_derives_from_prompt_and_is_trimmed():
    session = _session_all_defaulted("default", prompt="  a" + " very" * 40 + " long idea.  ")
    md = render_markdown(session)
    title_line = md.splitlines()[0]
    assert title_line.startswith("# A")
    assert len(title_line) <= 75
