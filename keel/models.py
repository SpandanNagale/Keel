"""Pydantic models for Keel.

Deliberately free of any Streamlit or Anthropic import: the engine and renderer
must be usable (and testable) as plain Python. Streamlit state in ``app.py`` is a
thin mirror of :class:`SessionState`, not a replacement for it.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# How a slot came to hold its value. Rendered specs, the sidebar, and the
# "Open questions" section key off these.
#
#   extracted        - pulled from the user's original prompt
#   asked            - the user typed this answer (or edited it in review)
#   reference        - confirmed from a scraped reference (a URL / product / image)
#   llm_default      - an LLM generated a contextual default; the user accepted it
#   template_default - static YAML fallback text; the LLM was unavailable
#   skipped          - the user declined to answer
#
# The split between llm_default and template_default matters: only the latter is
# a genuine degradation. A uniform "Keel default" label was Bug 1 — it made the
# sidebar and the degraded banner lie about a document full of contextual detail.
SlotSource = Literal[
    "extracted", "asked", "reference", "llm_default", "template_default", "skipped"
]

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


# --------------------------------------------------------------------------- #
# Reference intake (Phase 1): external evidence -> candidate slot values
# --------------------------------------------------------------------------- #
class Evidence(BaseModel):
    """Structural facts pulled from a scraped reference. Deliberately carries no
    marketing copy, branding, colours, or pricing — structure only."""

    product: str = ""
    core_entities: list[str] = Field(default_factory=list)
    primary_flows: list[str] = Field(default_factory=list)
    surfaces: list[str] = Field(default_factory=list)
    notable_features: list[str] = Field(default_factory=list)
    features_likely_out_of_scope: list[str] = Field(default_factory=list)


class SlotCandidate(BaseModel):
    """One proposed slot value derived from a reference. It becomes a real answer
    only after the developer keeps it (optionally edited) in the confirm step."""

    slot: str
    value: str
    evidence: str = ""          # the reference phrases this was built from
    decision: str = "keep"      # "keep" | "drop"


class ReferenceState(BaseModel):
    mode: str                                   # "url" (Phase 1); "name" / "image" later
    query: str = ""                             # the raw URL or product name given
    source_urls: list[str] = Field(default_factory=list)
    fetch_count: int = 0
    evidence: Optional[Evidence] = None
    candidates: list[SlotCandidate] = Field(default_factory=list)
    applied: int = 0
    confirmed: bool = False
    error: Optional[str] = None


class SessionState(BaseModel):
    """The whole of a Keel run. ``app.py`` stores exactly one of these in
    ``st.session_state`` and never mutates it during render."""

    original_prompt: str
    template_name: str
    created_date: str
    # How many slots to ASK about: "quick" (6), "standard" (8), "thorough" (9,
    # trimmed to the asked cap). Optional slots not asked are defaulted, not omitted.
    depth: str = "standard"
    # Ordered list of slot names still to ask about, frozen at Start so the
    # "Question X of N" progress counter never moves under the user.
    pending_slots: list[str] = Field(default_factory=list)
    slots: dict[str, SlotValue] = Field(default_factory=dict)
    current_index: int = 0
    questions_asked: int = 0
    call_count: int = 0
    # Set only when the synthesis LLM call itself failed and the document fell
    # back to the deterministic renderer. Contributes to ``degraded`` below.
    synthesis_failed: bool = False
    finished: bool = False
    # How many times the spec has been regenerated from edited answers (review
    # step). Capped; see engine.MAX_REGENERATIONS.
    regen_count: int = 0
    # Contradictions found by the pre-synthesis check_conflicts pass. Each entry:
    # {"slots": [...], "conflict": "...", "suggested_resolution": "..."}. Written
    # into "Open questions" mechanically by render.py, never by the synthesis model.
    conflicts: list[dict] = Field(default_factory=list)
    # Reason the conflict check could not run, if it failed. Surfaced in the
    # output as an explicit "checking was unavailable" note — never silently dropped.
    conflict_check_error: Optional[str] = None
    # Contradictions that the synthesis pass resolved (verified against the
    # rendered document afterwards). Each entry keeps the original conflict fields
    # plus "resolution". Shown in the UI as "Resolved during synthesis"; never
    # written into "Open questions" — only surviving conflicts go there.
    resolved_conflicts: list[dict] = Field(default_factory=list)
    # Reference intake, if one was gathered before the questions. Slots it filled
    # carry source "reference"; the source URL(s) are noted in "Open questions".
    reference: Optional[ReferenceState] = None

    @property
    def degraded(self) -> bool:
        """Whether the finished document genuinely fell back.

        True only when a slot is on static template text, or the conflict-check
        or synthesis call actually failed. An earlier failure that did not change
        the final document (a missed extraction, one re-tried question) must NOT
        trip this — Bug 1 was a sticky flag that cried wolf over a document full
        of contextual detail.
        """
        return (
            self.synthesis_failed
            or self.conflict_check_error is not None
            or any(v.source == "template_default" for v in self.slots.values())
        )

    def title(self) -> str:
        text = " ".join(self.original_prompt.split()).strip().rstrip(".")
        if not text:
            return "Untitled project"
        text = text[:70].strip()
        return text[0].upper() + text[1:]

    def skipped_slots(self) -> list[str]:
        return [name for name, v in self.slots.items() if v.source == "skipped"]
