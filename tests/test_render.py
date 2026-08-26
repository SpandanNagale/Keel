from datetime import datetime, timezone

from keel.engine import load_template
from keel.models import DetectionResult, SessionState, SlotSource, SlotState
from keel.render import render_markdown

REQUIRED_SECTIONS = [
    "## Context",
    "## Objective",
    "## Input / Output contract",
    "## Constraints",
    "## Acceptance criteria",
    "## Non-goals",
    "## Open questions",
]


def _make_session(template_name: str = "default") -> SessionState:
    template = load_template(template_name)
    slots = {}
    for slot in template.slots:
        slots[slot.name] = SlotState(value=f"answer for {slot.name}", source=SlotSource.ASKED)
    return SessionState(
        created_at=datetime.now(timezone.utc).isoformat(),
        original_prompt="build a widget",
        template_name=template_name,
        title="Widget Builder",
        detection=DetectionResult(summary="a python project"),
        slots=slots,
    )


def test_all_sections_present_in_fixed_order():
    session = _make_session()
    md = render_markdown(session)
    positions = [md.index(h) for h in REQUIRED_SECTIONS]
    assert positions == sorted(positions)


def test_no_section_is_left_empty_even_with_no_slots_filled():
    template = load_template("default")
    session = SessionState(
        created_at=datetime.now(timezone.utc).isoformat(),
        original_prompt="build a widget",
        template_name="default",
        title="Widget Builder",
        detection=DetectionResult(),
        slots={s.name: SlotState() for s in template.slots},
    )
    md = render_markdown(session)
    for i, heading in enumerate(REQUIRED_SECTIONS):
        start = md.index(heading) + len(heading)
        end = md.index(REQUIRED_SECTIONS[i + 1]) if i + 1 < len(REQUIRED_SECTIONS) else len(md)
        body = md[start:end].strip()
        assert body, f"section {heading!r} was empty"


def test_skipped_slot_appears_in_open_questions_not_its_section():
    session = _make_session()
    session.slots["non_goals"] = SlotState(value=None, source=SlotSource.SKIPPED)
    md = render_markdown(session)
    open_section = md[md.index("## Open questions"):]
    assert "Non-goals" in open_section
    non_goals_section = md[md.index("## Non-goals"):md.index("## Open questions")]
    assert "answer for non_goals" not in non_goals_section


def test_title_and_prompt_in_context():
    session = _make_session()
    md = render_markdown(session)
    assert md.startswith("# Widget Builder")
    assert "build a widget" in md
