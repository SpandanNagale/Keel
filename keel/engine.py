"""Slot state machine, template loading, and the two — and only two — LLM call
sites in Keel.

No Streamlit import here. ``app.py`` owns the widgets and the session-state
mirror; this module owns the logic and is unit-tested as plain Python.

The two call sites:
  1. :func:`extract_prefilled` — pull slots the opening idea already answers.
  2. :func:`next_question`     — generate one question + recommended default.

Both go through :func:`capped_complete_json`, which enforces the per-session call
cap and returns a ``(result, error)`` tuple. Failure is never silent: the error
string is handed back to the caller to surface, and ``session.degraded`` is set.
The one-shot synthesis pass in :mod:`keel.render` uses the same capped helper.
"""
from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

from keel import llm
from keel.models import SessionState, SlotDef, SlotValue, Template

# Hard cap: a single session may make at most this many LLM calls. The next one
# is refused with an error rather than made.
MAX_LLM_CALLS_PER_SESSION = 10

# Opening ideas longer than this are rejected before any call is made.
MAX_PROMPT_CHARS = 500

_TEMPLATE_NAMES = ["default", "cli", "data-pipeline", "web-api", "web-app"]

# Keyword -> template. First-pass heuristic only; the user can override in the UI.
#
# Order matters for tie-breaking: a prompt that scores equally for several
# templates is resolved by _TIE_ORDER below, and anything genuinely ambiguous
# (no clear winner) falls to "default" rather than to a specific template whose
# non-goals might contradict the project premise ("hotel website" -> web-api,
# whose non-goals forbid a frontend, was the motivating bug).
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "web-app": (
        "website", "web site", "web app", "webapp", "web application",
        "landing page", "dashboard", "frontend", "front end", "front-end",
        "site", "portal", "web page", "webpage", "web ui", "web interface",
        "marketing page", "single page app", "spa",
    ),
    "web-api": (
        "api", "apis", "endpoint", "endpoints", "rest", "restful", "graphql",
        "grpc", "webhook", "webhooks", "web service", "microservice",
        "openapi", "swagger", "json api",
    ),
    "data-pipeline": (
        "pipeline", "etl", "scrape", "scraper", "scraping", "crawl", "crawler",
        "ingest", "extract", "transform", "dataset", "data set", "warehouse",
        "cluster", "clustering", "dedupe", "deduplicate", "aggregate",
        "batch", "records", "migrate", "migration", "csv", "parquet",
        "embeddings", "index them", "sync",
    ),
    "cli": (
        "cli", "command line", "command-line", "commandline", "terminal",
        "script", "rename", "convert", "flag", "flags", "stdin", "stdout",
        "argument", "arguments", "subcommand", "one-off tool", "shell",
    ),
}

# When scores tie, earlier in this list wins. web-app beats web-api because a
# "booking website with an API" is still a website; a UI-forbidding template is
# the worse wrong answer.
_TIE_ORDER = ("web-app", "data-pipeline", "cli", "web-api")


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
def list_templates() -> list[str]:
    return list(_TEMPLATE_NAMES)


@lru_cache(maxsize=None)
def load_template(name: str) -> Template:
    if name not in _TEMPLATE_NAMES:
        raise ValueError(f"unknown template: {name!r}")
    try:
        text = (
            resources.files("keel.templates").joinpath(f"{name}.yaml").read_text("utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError):
        text = (Path(__file__).parent / "templates" / f"{name}.yaml").read_text("utf-8")
    data = yaml.safe_load(text)
    return Template.model_validate(data)


def select_template(prompt: str) -> str:
    """Pick a template by keyword frequency.

    No match -> "default". A tie between specific templates is broken by
    ``_TIE_ORDER`` (web-app before web-api, so a "website" is never routed to a
    UI-forbidding API template).
    """
    low = f" {prompt.lower()} "
    scores = {name: 0 for name in _KEYWORDS}
    for name, kws in _KEYWORDS.items():
        for kw in kws:
            # word-boundary hits are worth more than a bare substring hit
            scores[name] += 2 * low.count(f" {kw} ")
            scores[name] += 1 if kw in low else 0

    best_score = max(scores.values())
    if best_score == 0:
        return "default"
    winners = [name for name, s in scores.items() if s == best_score]
    if len(winners) == 1:
        return winners[0]
    for name in _TIE_ORDER:
        if name in winners:
            return name
    return "default"


def slugify(text: str, *, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = slug[:max_len].strip("-")
    return slug or "project"


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #
def start_session(prompt: str, template_name: str, *, created_date: str) -> SessionState:
    """Build an empty session. Makes no LLM call."""
    return SessionState(
        original_prompt=prompt.strip(),
        template_name=template_name,
        created_date=created_date,
    )


def freeze_pending(session: SessionState, template: Template) -> None:
    """Fix the ordered list of slots still to ask about. Call once, after
    extraction, so the ``Question X of N`` counter never shifts."""
    session.pending_slots = [
        s.name
        for s in template.required_slots()
        if session.slots.get(s.name) is None
    ]
    session.current_index = 0
    if not session.pending_slots:
        session.finished = True


def current_slot(session: SessionState, template: Template) -> SlotDef | None:
    if session.current_index >= len(session.pending_slots):
        return None
    return template.slot(session.pending_slots[session.current_index])


def _advance(session: SessionState) -> None:
    session.current_index += 1
    if session.current_index >= len(session.pending_slots):
        session.finished = True


def accept_answer(session: SessionState, slot_name: str, text: str, *, recommended: str) -> None:
    """Record an answer. If the text is unchanged from the recommended default,
    the source is ``defaulted``; otherwise ``asked``."""
    text = text.strip()
    if not text:
        text = recommended.strip()
        source = "defaulted"
    else:
        source = "defaulted" if text == recommended.strip() else "asked"
    session.slots[slot_name] = SlotValue(value=text, source=source)
    session.questions_asked += 1
    _advance(session)


def skip_slot(session: SessionState, slot_name: str) -> None:
    session.slots[slot_name] = SlotValue(value="", source="skipped")
    _advance(session)


def fill_remaining_defaults(session: SessionState, template: Template) -> None:
    """Backfill every required slot that was never resolved (caps hit, or a
    non-asked required slot) with its static ``default_text``."""
    for s in template.required_slots():
        if session.slots.get(s.name) is None:
            session.slots[s.name] = SlotValue(value=s.default_text, source="defaulted")
    session.finished = True


# --------------------------------------------------------------------------- #
# LLM call sites
# --------------------------------------------------------------------------- #
def capped_complete_json(
    session: SessionState,
    system: str,
    user: str,
    *,
    provider: "llm.Provider | None",
    max_tokens: int = llm.MAX_OUTPUT_TOKENS,
) -> tuple[dict | None, str | None]:
    """The only way any part of Keel makes an LLM call. Enforces the per-session
    call cap, increments the counter, and forwards to :func:`keel.llm.complete_json`."""
    if session.call_count >= MAX_LLM_CALLS_PER_SESSION:
        return None, f"session LLM call limit reached ({MAX_LLM_CALLS_PER_SESSION} calls)"
    session.call_count += 1
    return llm.complete_json(system, user, provider=provider, max_tokens=max_tokens)


_EXTRACTION_SYSTEM = """You are Keel, which compiles vague software project ideas into precise build specs.

Given a one-line idea and a list of specification slots, decide which slots the idea ALREADY \
answers — explicitly, or by clear and direct implication. For each, extract a short factual \
phrase (at most 15 words) stating what the idea says about that slot.

Omit any slot the idea leaves unstated. Do not guess, do not fill from convention, do not \
infer beyond what the words support.

Respond with a JSON object mapping slot name to the extracted phrase. Use {} if the idea \
answers none of the slots."""


def extract_prefilled(
    session: SessionState, template: Template, *, provider: "llm.Provider | None"
) -> str | None:
    """One LLM call: fill slots the opening idea already answers. Mutates
    ``session.slots`` (source ``extracted``). Returns an error string on failure
    and sets ``session.degraded``; returns ``None`` on success."""
    slot_lines = "\n".join(f"- {s.name}: {s.question_hint}" for s in template.slots)
    user = f'Idea: "{session.original_prompt}"\n\nSlots:\n{slot_lines}\n\nJSON:'

    result, error = capped_complete_json(session, _EXTRACTION_SYSTEM, user, provider=provider)
    if error is not None:
        session.degraded = True
        return error

    valid = {s.name for s in template.slots}
    junk = {"", "unknown", "n/a", "na", "none", "not specified", "unspecified", "tbd"}
    for name, value in (result or {}).items():
        if name not in valid or session.slots.get(name) is not None:
            continue
        phrase = str(value).strip()
        if phrase.lower() in junk:
            continue
        session.slots[name] = SlotValue(value=phrase, source="extracted")
    return None


_QUESTION_SYSTEM = """You are Keel. Ask a software developer ONE clarifying question about their \
project idea, targeting a single dimension they left unspecified, and propose a recommended \
answer they can accept unchanged.

Rules:
- One question only. At most 25 words. Concrete. No preamble, no "it depends".
- The recommended answer is a finished spec line: specific formats, numbers, and names — \
never a menu of options, never a hedge.
- Do not re-ask anything under "Already established".
- Do not name specific library versions, and avoid recommending niche tools whose current \
health you cannot verify.
- Respond with a JSON object: {"question": "...", "recommended": "..."}"""


def _question_user(slot: SlotDef, session: SessionState, template: Template) -> str:
    established = [
        f"- {template.slot(n).label}: {v.value}"
        for n, v in session.slots.items()
        if v.source in ("extracted", "asked", "defaulted") and v.value and template.slot(n)
    ]
    established_block = "\n".join(established) if established else "- nothing yet"
    return (
        f'Project idea: "{session.original_prompt}"\n\n'
        f"Already established:\n{established_block}\n\n"
        f"Target slot: {slot.label} ({slot.name})\n"
        f"What it must pin down: {slot.question_hint}\n"
        f"How to choose the recommended answer: {slot.default_strategy}\n\n"
        f"JSON:"
    )


def next_question(
    session: SessionState, template: Template, *, provider: "llm.Provider | None"
) -> tuple[str, str, str | None]:
    """Generate the question + recommended default for the slot at
    ``current_index``. Returns ``(question, recommended, error)``. On any
    failure, returns the slot's static ``question_hint`` / ``default_text`` and a
    non-null error, and sets ``session.degraded``."""
    slot = current_slot(session, template)
    if slot is None:
        return "", "", "no pending slot"

    result, error = capped_complete_json(
        session, _QUESTION_SYSTEM, _question_user(slot, session, template),
        provider=provider,
    )
    if error is not None:
        session.degraded = True
        return slot.question_hint, slot.default_text, error

    question = str((result or {}).get("question", "")).strip()
    recommended = str((result or {}).get("recommended") or (result or {}).get("default", "")).strip()
    if not question or not recommended:
        session.degraded = True
        return (
            slot.question_hint,
            slot.default_text,
            f"model response missing question/recommended fields: {result!r}",
        )
    return question, recommended, None
