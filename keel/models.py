"""Pydantic models for Keel.

Deliberately free of any Streamlit or Anthropic import: the engine and renderer
must be usable (and testable) as plain Python. Streamlit state in ``app.py`` is a
thin mirror of :class:`SessionState`, not a replacement for it.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# How a slot came to hold its value. Rendered specs and the "Open questions"
# section key off these.
SlotSource = Literal["extracted", "asked", "defaulted", "skipped"]

# Fixed output sections, in render order. Every slot's ``section`` must be one of
# these; ``context`` also absorbs the opening prompt and the degraded note.
Section = Literal[
    "context",
    "io_contract",
    "constraints",
    "acceptance",
    "non_goals",
]


class SlotDef(BaseModel):
    """One dimension a prompt must pin down, loaded from a template YAML file."""

    name: str
    section: Section
    label: str
    question_hint: str
    """Static fallback question, shown verbatim when the LLM is unavailable."""
    default_text: str
    """Short, concrete, human-readable fallback answer. May appear in output."""
    default_strategy: str
    """Instruction for the model when generating a context-aware default.

    NEVER rendered into the output document. Only ``default_text`` or an
    LLM-generated string may become a slot's value.
    """
    required: bool = True
    priority: int = 100


class Template(BaseModel):
    name: str
    title: str
    description: str = ""
    slots: list[SlotDef]

    def required_slots(self) -> list[SlotDef]:
        return sorted(
            (s for s in self.slots if s.required),
            key=lambda s: (s.priority, s.name),
        )

    def slot(self, name: str) -> Optional[SlotDef]:
        return next((s for s in self.slots if s.name == name), None)


class SlotValue(BaseModel):
    value: str = ""
    source: SlotSource


class SessionState(BaseModel):
    """The whole of a Keel run. ``app.py`` stores exactly one of these in
    ``st.session_state`` and never mutates it during render."""

    original_prompt: str
    template_name: str
    created_date: str
    # Ordered list of slot names still to ask about, frozen at Start so the
    # "Question X of N" progress counter never moves under the user.
    pending_slots: list[str] = Field(default_factory=list)
    slots: dict[str, SlotValue] = Field(default_factory=dict)
    current_index: int = 0
    questions_asked: int = 0
    call_count: int = 0
    degraded: bool = False
    finished: bool = False
    # Contradictions found by the pre-synthesis check_conflicts pass. Each entry:
    # {"slots": [...], "conflict": "...", "suggested_resolution": "..."}. Written
    # into "Open questions" mechanically by render.py, never by the synthesis model.
    conflicts: list[dict] = Field(default_factory=list)
    # Reason the conflict check could not run, if it failed. Surfaced in the
    # output as an explicit "checking was unavailable" note — never silently dropped.
    conflict_check_error: Optional[str] = None

    def title(self) -> str:
        text = " ".join(self.original_prompt.split()).strip().rstrip(".")
        if not text:
            return "Untitled project"
        text = text[:70].strip()
        return text[0].upper() + text[1:]

    def skipped_slots(self) -> list[str]:
        return [name for name, v in self.slots.items() if v.source == "skipped"]
