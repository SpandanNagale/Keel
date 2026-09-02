"""Slot state machine, template loading, and the two — and only two — LLM call
sites in Keel.

No Streamlit import here. ``app.py`` owns the widgets and the session-state
mirror; this module owns the logic and is unit-tested as plain Python.

The two call sites:
  1. :func:`extract_prefilled` — pull slots the opening idea already answers.
  2. :func:`next_question`     — generate one question + recommended default.

Both go through :func:`capped_complete_json`, which enforces the per-session call
cap and returns a ``(result, error)`` tuple. Failure is never silent: the error
string is handed back to the caller to surface. Whether that failure *degrades*
the finished document is a separate question — see ``SessionState.degraded``,
which keys off template-default slots and a failed synthesis/conflict call, not
off every transient error. The one-shot synthesis pass in :mod:`keel.render`
uses the same capped helper.
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
# is refused with an error rather than made. Raised from 10 for Phase 2: a
# "thorough" session spends extract(1) + asked(8) + context-defaults(1) +
# check_conflicts(1) + synthesis(1) = 12; Phase 3 adds up to MAX_REGENERATIONS
# regenerations at two calls each, so the ceiling is 14 + 6 = 20.
MAX_LLM_CALLS_PER_SESSION = 20

# Never ask more than this many questions in one session, regardless of how many
# slots the template (and depth) put in play. Slots past this are defaulted from
# accumulated context rather than asked.
MAX_ASKED_QUESTIONS = 8

# Depth -> how many of the optional (required=false) slots to ASK about, in
# priority order. The rest are still filled, just not asked.
DEPTH_OPTIONAL = {"quick": 0, "standard": 2, "thorough": 3}

# How many times a finished spec may be regenerated from edited answers. Each
# regeneration costs two calls (conflict check + synthesis); the per-session cap
# above carries the headroom (14 base + 3 * 2).
MAX_REGENERATIONS = 3

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
def start_session(
    prompt: str, template_name: str, *, created_date: str, depth: str = "standard"
) -> SessionState:
    """Build an empty session. Makes no LLM call."""
    return SessionState(
        original_prompt=prompt.strip(),
        template_name=template_name,
        created_date=created_date,
        depth=depth if depth in DEPTH_OPTIONAL else "standard",
    )


def askable_slots(session: SessionState, template: Template) -> list[SlotDef]:
    """The slots this session's depth puts in play to ASK about: the required
    six, plus the first N optional slots by priority (N from the depth)."""
    n_optional = DEPTH_OPTIONAL.get(session.depth, DEPTH_OPTIONAL["standard"])
    optional = sorted(
        (s for s in template.slots if not s.required),
        key=lambda s: (s.priority, s.name),
    )
    chosen = list(template.required_slots()) + optional[:n_optional]
    return sorted(chosen, key=lambda s: (s.priority, s.name))


def freeze_pending(session: SessionState, template: Template) -> None:
    """Fix the ordered list of slots still to ask about. Call once, after
    extraction, so the ``Question X of N`` counter never shifts. Depth chooses how
    many optional slots are included; the list is then capped at
    :data:`MAX_ASKED_QUESTIONS`. Slots left out are filled by
    :func:`fill_unasked_slots`, not omitted."""
    pending = [
        s.name
        for s in askable_slots(session, template)
        if session.slots.get(s.name) is None
    ]
    session.pending_slots = pending[:MAX_ASKED_QUESTIONS]
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


def accept_answer(
    session: SessionState,
    slot_name: str,
    text: str,
    *,
    recommended: str,
    recommended_source: str = "llm_default",
) -> None:
    """Record an answer.

    If the developer edited the recommendation, the source is ``asked``. If they
    accepted it unchanged (or left the box empty), the source is
    ``recommended_source`` — ``llm_default`` when the recommendation came from the
    model, ``template_default`` when the question call had failed and the box was
    pre-filled with the slot's static ``default_text``. The caller knows which.
    """
    text = text.strip()
    if not text:
        text = recommended.strip()
        source = recommended_source
    else:
        source = recommended_source if text == recommended.strip() else "asked"
    session.slots[slot_name] = SlotValue(value=text, source=source)
    session.questions_asked += 1
    _advance(session)


def skip_slot(session: SessionState, slot_name: str) -> None:
    session.slots[slot_name] = SlotValue(value="", source="skipped")
    _advance(session)


def fill_remaining_defaults(session: SessionState, template: Template) -> None:
    """Backfill every slot — required or optional — that was never resolved with
    its static ``default_text``. Makes no LLM call: this is the "skip the rest,
    use template defaults" path."""
    for s in template.slots:
        if session.slots.get(s.name) is None:
            session.slots[s.name] = SlotValue(value=s.default_text, source="template_default")
    session.finished = True


_CONTEXT_DEFAULT_SYSTEM = """You are Keel. Some specification dimensions were not asked about, to \
keep the session short. Given the project idea and the answers already collected, fill each \
remaining dimension with ONE concrete value the developer would most likely accept — the same \
kind of finished spec line the questions produce.

Rules:
- Be specific and fully consistent with the answers already given. Do not hedge, do not offer \
  options, do not restate the question.
- Do not invent product names, versions, or numeric figures that are not implied by what is \
  already established.
- Respond with a JSON object mapping each requested slot name to its value string. Include \
  every slot asked for."""


def _context_default_user(
    session: SessionState, template: Template, missing: list[SlotDef]
) -> str:
    established = [
        f"- {template.slot(n).label}: {v.value}"
        for n, v in session.slots.items()
        if v.value and template.slot(n)
    ]
    established_block = "\n".join(established) if established else "- nothing yet"
    wanted = "\n".join(f"- {s.name} ({s.label}): {s.question_hint}" for s in missing)
    return (
        f'Project idea: "{session.original_prompt}"\n\n'
        f"Already established:\n{established_block}\n\n"
        f"Fill these dimensions:\n{wanted}\n\nJSON:"
    )


def fill_unasked_slots(
    session: SessionState, template: Template, *, provider: "llm.Provider | None"
) -> str | None:
    """Fill every slot that was never asked (depth or the asked-question cap left
    it out) with a value inferred from the accumulated answers — one LLM call for
    all of them. Any slot the model does not return, or the whole call on failure,
    falls back to the static ``default_text`` and is marked ``template_default``
    (which is what drives ``session.degraded``). Returns an error string if the
    call failed, else ``None``. Always marks the session finished."""
    missing = [
        s
        for s in sorted(template.slots, key=lambda s: (s.priority, s.name))
        if session.slots.get(s.name) is None
    ]
    if not missing:
        session.finished = True
        return None

    error: str | None = None
    if provider is not None:
        result, error = capped_complete_json(
            session,
            _CONTEXT_DEFAULT_SYSTEM,
            _context_default_user(session, template, missing),
            provider=provider,
        )
        if error is None:
            for s in missing:
                value = str((result or {}).get(s.name, "")).strip()
                if value:
                    session.slots[s.name] = SlotValue(value=value, source="llm_default")

    for s in missing:  # static fallback for anything still unfilled
        if session.slots.get(s.name) is None:
            session.slots[s.name] = SlotValue(value=s.default_text, source="template_default")

    session.finished = True
    return error


# --------------------------------------------------------------------------- #
# LLM call sites
# --------------------------------------------------------------------------- #
def apply_answer_edits(
    session: SessionState, edits: dict[str, str], template: Template
) -> int:
    """Overwrite slot values from the review step. A value that changed is marked
    ``source="asked"`` (the developer stands behind it now); an emptied value
    becomes a skip. Returns how many slots actually changed."""
    changed = 0
    for name, raw in edits.items():
        slot = template.slot(name)
        if slot is None:
            continue
        new = raw.strip()
        current = session.slots.get(name)
        old = current.value.strip() if current else None
        if new == old:
            continue
        if not new:
            session.slots[name] = SlotValue(value="", source="skipped")
        else:
            session.slots[name] = SlotValue(value=new, source="asked")
        changed += 1
    return changed


def can_regenerate(session: SessionState) -> tuple[bool, str | None]:
    """Whether another regeneration is allowed. Two calls are needed (conflict
    check + synthesis)."""
    if session.regen_count >= MAX_REGENERATIONS:
        return False, f"regeneration limit reached ({MAX_REGENERATIONS})"
    if session.call_count + 2 > MAX_LLM_CALLS_PER_SESSION:
        return False, "not enough of this session's LLM-call budget left to regenerate"
    return True, None


def regenerations_left(session: SessionState) -> int:
    by_count = MAX_REGENERATIONS - session.regen_count
    by_budget = max((MAX_LLM_CALLS_PER_SESSION - session.call_count) // 2, 0)
    return max(min(by_count, by_budget), 0)


def capped_complete_json(
    session: SessionState,
    system: str,
    user: str,
    *,
    provider: "llm.Provider | None",
    max_tokens: int = llm.MAX_OUTPUT_TOKENS,
    image: "tuple[bytes, str] | None" = None,
) -> tuple[dict | None, str | None]:
    """The only way any part of Keel makes an LLM call. Enforces the per-session
    call cap, increments the counter, and forwards to :func:`keel.llm.complete_json`."""
    if session.call_count >= MAX_LLM_CALLS_PER_SESSION:
        return None, f"session LLM call limit reached ({MAX_LLM_CALLS_PER_SESSION} calls)"
    session.call_count += 1
    return llm.complete_json(
        system, user, provider=provider, max_tokens=max_tokens, image=image
    )


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
    ``session.slots`` (source ``extracted``). Returns an error string on failure,
    ``None`` on success. A failed extraction does not by itself degrade the
    session — every slot it would have filled is still asked normally."""
    slot_lines = "\n".join(f"- {s.name}: {s.question_hint}" for s in template.slots)
    user = f'Idea: "{session.original_prompt}"\n\nSlots:\n{slot_lines}\n\nJSON:'

    result, error = capped_complete_json(session, _EXTRACTION_SYSTEM, user, provider=provider)
    if error is not None:
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
        if v.source in ("extracted", "asked", "llm_default", "template_default")
        and v.value and template.slot(n)
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
    non-null error. The session is not marked degraded here: it only degrades if
    the user then accepts that static recommendation (``accept_answer`` records
    it as ``template_default``)."""
    slot = current_slot(session, template)
    if slot is None:
        return "", "", "no pending slot"

    result, error = capped_complete_json(
        session, _QUESTION_SYSTEM, _question_user(slot, session, template),
        provider=provider,
    )
    if error is not None:
        return slot.question_hint, slot.default_text, error

    question = str((result or {}).get("question", "")).strip()
    recommended = str((result or {}).get("recommended") or (result or {}).get("default", "")).strip()
    if not question or not recommended:
        return (
            slot.question_hint,
            slot.default_text,
            f"model response missing question/recommended fields: {result!r}",
        )
    return question, recommended, None
