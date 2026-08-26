"""Renders a SessionState into the fixed output markdown schema.

Section order never varies:
  Context, Objective, Input / Output contract, Constraints,
  Acceptance criteria, Non-goals, Open questions.
"""
from __future__ import annotations

from keel.engine import load_template
from keel.models import SessionState, SlotSource

SECTION_TITLES = {
    "io_contract": "Input / Output contract",
    "constraints": "Constraints",
    "acceptance": "Acceptance criteria",
    "non_goals": "Non-goals",
}

SECTION_EMPTY_FALLBACK = {
    "io_contract": "_Not specified._",
    "constraints": "_No hard constraints captured._",
    "acceptance": "_Not specified._",
    "non_goals": "_None specified — the agent should use judgement and avoid unnecessary scope._",
}

SECTION_ORDER = ("io_contract", "constraints", "acceptance", "non_goals")

_IMPERATIVE_STARTS = (
    "build", "create", "make", "write", "scrape", "convert", "extract",
    "generate", "develop", "implement", "design", "add", "set up", "sync",
    "monitor", "automate", "parse", "fetch", "clean", "dedupe", "rename",
)


def _objective_sentence(prompt: str) -> str:
    text = prompt.strip().rstrip(".")
    if not text:
        return "Build the tool described by the developer."
    first_word = text.split(" ", 1)[0].lower()
    sentence = text if first_word in _IMPERATIVE_STARTS else f"Build a tool that {text}"
    return sentence[0].upper() + sentence[1:] + "."


def render_markdown(session: SessionState) -> str:
    template = load_template(session.template_name)
    slot_defs = {s.name: s for s in template.slots}

    lines: list[str] = [f"# {session.title}", ""]

    lines.append("## Context")
    lines.append("")
    lines.append(f'Original request: "{session.original_prompt}"')
    if session.detection.summary:
        lines.append("")
        lines.append(f"Detected environment: {session.detection.summary}.")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append(_objective_sentence(session.original_prompt))
    lines.append("")

    for section_key in SECTION_ORDER:
        lines.append(f"## {SECTION_TITLES[section_key]}")
        lines.append("")
        bullets = []
        for slot in template.slots:
            if slot.section != section_key:
                continue
            state = session.slots.get(slot.name)
            if state and state.value and state.source != SlotSource.SKIPPED:
                bullets.append(f"- **{slot.label}:** {state.value}")
        lines.extend(bullets if bullets else [SECTION_EMPTY_FALLBACK[section_key]])
        lines.append("")

    lines.append("## Open questions")
    lines.append("")
    open_items = [
        slot_defs[name].label
        for name, state in session.slots.items()
        if state.source == SlotSource.SKIPPED and name in slot_defs
    ]
    if open_items:
        for label in open_items:
            lines.append(f"- {label} — left unanswered; confirm with the developer before proceeding.")
    else:
        lines.append("_None — every required slot was filled or explicitly defaulted._")
    lines.append("")

    return "\n".join(lines)
