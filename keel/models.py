"""Pydantic models for Keel's slot state machine."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Output sections a slot can feed into. "context" and "objective" are always
# composed by the engine directly (they need the raw prompt/detection, not a
# single slot value), so they are not valid section targets for a slot.
SLOT_SECTIONS = ("io_contract", "constraints", "acceptance", "non_goals")


class SlotSource(str, Enum):
    EXTRACTED = "extracted"
    DETECTED = "detected"
    ASKED = "asked"
    DEFAULTED = "defaulted"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"


class SlotDef(BaseModel):
    name: str
    section: str
    label: str
    question_hint: str
    default_strategy: str
    required: bool = True
    skip_if: Optional[str] = None
    priority: int = 100


class Template(BaseModel):
    name: str
    description: str = ""
    slots: list[SlotDef]


class SlotState(BaseModel):
    value: Optional[str] = None
    source: Optional[SlotSource] = None


class DetectionResult(BaseModel):
    project_type: Optional[str] = None
    language: Optional[str] = None
    dependencies: list[str] = Field(default_factory=list)
    scripts: dict[str, str] = Field(default_factory=dict)
    repo_name: Optional[str] = None
    has_tests: bool = False
    has_dockerfile: bool = False
    has_env_example: bool = False
    summary: str = ""


class SessionState(BaseModel):
    version: int = 1
    created_at: str
    original_prompt: str
    template_name: str
    title: str
    detection: DetectionResult
    slots: dict[str, SlotState] = Field(default_factory=dict)
    questions_asked: int = 0
