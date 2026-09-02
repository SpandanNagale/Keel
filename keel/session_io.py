"""Serialise a :class:`~keel.models.SessionState` to JSON and back.

Download / upload only — Keel never persists a session server-side. The wrapper
carries a ``schema_version`` so a file written by a newer Keel is rejected with a
clear message instead of loading into a half-understood shape.

No Streamlit import: ``app.py`` wires the buttons, this module does the bytes.
"""
from __future__ import annotations

import json

from keel.models import SessionState

# Bump when the serialised shape changes in a way older Keel cannot read.
#   1 -> 2 (Phase 0/1): SlotSource "defaulted" split into llm_default /
#          template_default; "degraded" is now derived, not stored; added
#          synthesis_failed, resolved_conflicts, reference.
SCHEMA_VERSION = 2

_WRAPPER_KEY = "keel_session"


def _migrate_1_to_2(body: dict) -> dict:
    """A v1 file predates the provenance split. Map the old single "defaulted"
    source to "llm_default" (the common case — an accepted suggestion), and drop
    the stored "degraded" flag, which is now derived from state."""
    body.pop("degraded", None)
    for slot in (body.get("slots") or {}).values():
        if isinstance(slot, dict) and slot.get("source") == "defaulted":
            slot["source"] = "llm_default"
    return body


# version N -> N+1
_MIGRATIONS = {1: _migrate_1_to_2}


def dumps(session: SessionState) -> str:
    """The session as pretty JSON, wrapped with its schema version."""
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, _WRAPPER_KEY: session.model_dump()},
        indent=2,
        ensure_ascii=False,
    )


def loads(text: str) -> tuple[SessionState | None, str | None]:
    """Parse a session file. Returns ``(session, None)`` or ``(None, reason)``.

    Rejects, with a specific reason: non-JSON, a missing or non-integer
    ``schema_version``, a version newer than this Keel understands, a missing
    session body, and a body that does not validate against the model."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"not a valid session file (invalid JSON: {exc})"

    if not isinstance(data, dict):
        return None, "not a valid session file (expected a JSON object at the top level)"

    version = data.get("schema_version")
    if not isinstance(version, int):
        return None, "not a Keel session file (no schema_version)"
    if version > SCHEMA_VERSION:
        return None, (
            f"this file was written by a newer version of Keel "
            f"(schema {version}; this Keel understands {SCHEMA_VERSION}). Update Keel or "
            "re-export the session."
        )
    if version < 1:
        return None, "not a valid Keel session file (bad schema_version)"
    if any(v not in _MIGRATIONS for v in range(version, SCHEMA_VERSION)):
        return None, (
            f"this session file uses schema {version}, which this Keel cannot upgrade. "
            "Start a fresh session."
        )

    body = data.get(_WRAPPER_KEY)
    if not isinstance(body, dict):
        return None, "session file is missing its body"

    while version < SCHEMA_VERSION:  # step it up one schema at a time
        body = _MIGRATIONS[version](body)
        version += 1

    try:
        session = SessionState.model_validate(body)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError and friends
        return None, f"session file did not match the expected shape: {exc}"
    return session, None
