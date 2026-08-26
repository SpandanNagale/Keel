"""The slot-filling state machine. Deterministic control flow driving stateless LLM calls.

Termination condition: every required slot has been addressed (filled, defaulted,
detected, extracted, marked not-applicable, or explicitly skipped). This is a plain
loop over template.slots — never "the LLM decides it has asked enough."
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from keel import llm
from keel.models import DetectionResult, SessionState, SlotDef, SlotSource, SlotState, Template

TEMPLATES_DIR = Path(__file__).parent / "templates"


def load_template(name: str) -> Template:
    path = TEMPLATES_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown template: {name}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Template.model_validate(data)


def list_templates() -> list[str]:
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.yaml"))


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "project"


def derive_title(prompt: str) -> str:
    text = prompt.strip().rstrip(".")
    if not text:
        return "Untitled Project"
    if len(text) > 80:
        text = text[:80].rsplit(" ", 1)[0] + "..."
    return text[0].upper() + text[1:]


def evaluate_skip_if(expr: str, detection: DetectionResult, slots: dict[str, SlotState]) -> bool:
    if not expr:
        return False
    context = {
        "detection": detection.model_dump(),
        "slots": {name: state.value for name, state in slots.items()},
    }
    try:
        return bool(eval(expr, {"__builtins__": {}}, context))  # noqa: S307 - restricted namespace
    except Exception:
        return False


EXTRACTION_SYSTEM = """You extract information from a short, vague software project prompt.
Given the prompt and a list of "slots" (dimensions a coding agent needs pinned down),
identify ONLY the slots the prompt already answers explicitly or very strongly implies.
Do not guess or invent information the prompt does not support — when in doubt, omit the slot.
Respond with ONLY a JSON object mapping slot name to a short extracted phrase.
Omit any slot the prompt does not address. No explanations, no markdown fences."""


def extract_slots(prompt: str, template: Template) -> dict[str, str]:
    slot_list = "\n".join(f"- {s.name}: {s.question_hint}" for s in template.slots)
    user = f'Prompt: "{prompt}"\n\nSlots:\n{slot_list}\n\nJSON:'
    result = llm.complete_json(EXTRACTION_SYSTEM, user, max_tokens=512, effort="low")
    if not result:
        return {}
    valid_names = {s.name for s in template.slots}
    return {
        name: str(value).strip()
        for name, value in result.items()
        if name in valid_names and str(value).strip()
    }


QUESTION_SYSTEM = """You write ONE clarifying question for a developer building a software
project, to pin down a specific missing detail (a "slot") before handing the project off
to a coding agent. You are given the slot's static hint and a strategy for choosing a
sensible default answer.

Rules:
- Ask about exactly this one slot. Do not ask about anything else.
- Propose a concrete, specific recommended default the developer can accept by pressing Enter.
- Use the project context (original prompt, detected repo info, slots already filled) to make
  the question and default as specific as possible instead of generic.
- Keep the question to one sentence.
Respond with ONLY a JSON object: {"question": "...", "default": "..."}. No markdown fences."""


def generate_question(slot: SlotDef, session: SessionState) -> tuple[str, str]:
    filled = "\n".join(
        f"- {name}: {state.value}" for name, state in session.slots.items() if state.value
    )
    context = (
        f'Original prompt: "{session.original_prompt}"\n'
        f"Detected project: {session.detection.summary or '(nothing detected)'}\n"
        f"Slots already filled:\n{filled or '(none yet)'}\n\n"
        f"Slot to ask about: {slot.name}\n"
        f"Static hint: {slot.question_hint}\n"
        f"Default strategy: {slot.default_strategy}"
    )
    result = llm.complete_json(QUESTION_SYSTEM, context, max_tokens=400, effort="low")
    if result and result.get("question") and result.get("default"):
        return str(result["question"]).strip(), str(result["default"]).strip()
    return slot.question_hint, slot.default_strategy


class Engine:
    def __init__(self, session: SessionState, template: Template):
        self.session = session
        self.template = template

    @classmethod
    def start(
        cls,
        prompt: str,
        template_name: str,
        detection: DetectionResult,
        *,
        extract: bool = True,
    ) -> "Engine":
        template = load_template(template_name)
        session = SessionState(
            created_at=datetime.now(timezone.utc).isoformat(),
            original_prompt=prompt,
            template_name=template_name,
            title=derive_title(prompt),
            detection=detection,
            slots={s.name: SlotState() for s in template.slots},
        )
        engine = cls(session, template)
        engine._mark_not_applicable()
        if extract:
            engine._apply_extraction()
        return engine

    def _mark_not_applicable(self) -> None:
        for slot in self.template.slots:
            if slot.skip_if and evaluate_skip_if(slot.skip_if, self.session.detection, self.session.slots):
                self.session.slots[slot.name] = SlotState(source=SlotSource.NOT_APPLICABLE)

    def _apply_extraction(self) -> None:
        extracted = extract_slots(self.session.original_prompt, self.template)
        for name, value in extracted.items():
            state = self.session.slots.get(name)
            if state and state.source is None:
                self.session.slots[name] = SlotState(value=value, source=SlotSource.EXTRACTED)

    def apply_detection_prefill(self, confirmed: bool) -> None:
        if not confirmed:
            return
        summary = self.session.detection.summary
        if not summary:
            return
        value = summary[0].upper() + summary[1:]
        for name in ("runtime", "constraints"):
            state = self.session.slots.get(name)
            if state and state.source is None:
                self.session.slots[name] = SlotState(value=value, source=SlotSource.DETECTED)

    def _is_answered(self, name: str) -> bool:
        state = self.session.slots.get(name)
        if state is None:
            return False
        if state.value:
            return True
        return state.source in (SlotSource.NOT_APPLICABLE, SlotSource.SKIPPED)

    def is_filled(self, name: str) -> bool:
        return self._is_answered(name)

    def remaining_required_slots(self) -> list[SlotDef]:
        return [s for s in self.template.slots if s.required and not self._is_answered(s.name)]

    def quick_priority_slots(self, limit: int = 3) -> list[SlotDef]:
        remaining = sorted(self.remaining_required_slots(), key=lambda s: s.priority)
        return remaining[:limit]

    def is_complete(self) -> bool:
        return len(self.remaining_required_slots()) == 0

    def generate_question_for(self, slot: SlotDef) -> tuple[str, str]:
        return generate_question(slot, self.session)

    def apply_answer(self, slot_name: str, raw: str, recommended_default: str) -> None:
        raw = raw.strip()
        self.session.questions_asked += 1
        if raw == "":
            self.session.slots[slot_name] = SlotState(value=recommended_default, source=SlotSource.DEFAULTED)
        elif raw.lower() == "skip":
            self.session.slots[slot_name] = SlotState(value=None, source=SlotSource.SKIPPED)
        else:
            self.session.slots[slot_name] = SlotState(value=raw, source=SlotSource.ASKED)

    def default_remaining_silently(self) -> None:
        for slot in self.remaining_required_slots():
            _, default = self.generate_question_for(slot)
            self.session.slots[slot.name] = SlotState(value=default, source=SlotSource.DEFAULTED)
