"""Turning filled slots into the output document.

Two paths:

* :func:`synthesize_spec` — the primary path. One capped LLM call, made once per
  session after every slot is filled, that reads the original idea plus every
  answer and writes the whole document: sections expanded to the length the
  content warrants, facts moved to the section they belong in, contradictions
  between answers detected and resolved. Its output is then run through
  deterministic Python checks (:func:`_assemble_and_validate`) before it is
  shown — structure, empty sections, hardcoded secrets, leaked template text.

* :func:`render_markdown` — the deterministic fallback. One slot value per line,
  no model. Used whenever synthesis fails or no LLM is available. Fully
  functional on its own.

Section headings and their order are owned here, never by the model: the model
returns section *bodies* keyed by name, and this module assembles them under
fixed headings in :data:`SECTION_ORDER`.

Only a slot's ``value`` is ever emitted. ``default_strategy`` is model-facing and
must never reach the output — there are tests that assert it for both paths.
"""
from __future__ import annotations

import logging
import math
import re

from keel import engine, llm
from keel.models import SessionState, Template

_log = logging.getLogger("keel.render")

# (heading rendered, JSON key). The synthesis model returns a body for every key
# EXCEPT "decisions", which is assembled in Python from the keel_decided slots —
# the model never writes it, exactly as with "open_questions" (which it drafts
# but Python rebuilds). Order and headings are owned here, never by the model.
SECTION_ORDER: list[tuple[str, str]] = [
    ("Context", "context"),
    ("Objective", "objective"),
    ("Input / Output contract", "io_contract"),
    ("Constraints", "constraints"),
    ("Acceptance criteria", "acceptance_criteria"),
    ("Build order", "build_order"),
    ("Project structure", "project_structure"),
    ("Decisions Keel made for you", "decisions"),
    ("Non-goals", "non_goals"),
    ("Open questions", "open_questions"),
]
_HEADINGS = [h for h, _ in SECTION_ORDER]

# Keys the synthesis model is asked to return a body for and that MUST be
# non-empty or the render falls back. "decisions" is assembled in Python.
_REQUIRED_MODEL_KEYS = [
    "context", "objective", "io_contract", "constraints",
    "acceptance_criteria", "non_goals", "open_questions",
]

# Keys the model also returns, but which are additive: absent or empty just means
# the section is omitted, never a fallback. Also suppressed wholesale at Quick
# depth, which lacks the slot data (data_model / interfaces) to support them.
_SOFT_MODEL_KEYS = ["build_order", "project_structure"]

_MODEL_SECTION_KEYS = _REQUIRED_MODEL_KEYS + _SOFT_MODEL_KEYS

# Sections omitted entirely when their body is empty (or, for the two soft keys,
# when the session is Quick depth) rather than failing the render.
_OPTIONAL_SECTION_KEYS = {"decisions", "build_order", "project_structure"}

# Synthesis writes a whole document — and since Phase 2 it enumerates the full
# interface surface — so it needs a generous budget. Too small and a verbose
# model runs out mid-JSON and the call hard-fails to the deterministic renderer.
SYNTHESIS_MAX_TOKENS = 3500

_DEGRADED_NOTE = (
    "> **Note:** this spec was generated without LLM assistance for at least one "
    "step — parts fell back to Keel's generic template defaults. Review the "
    "recommended answers carefully before handing this to an agent."
)

_SCRUB_NOTE = (
    "- A hardcoded secret was detected in the generated text and replaced with an "
    "environment-variable placeholder; provide the real value at runtime, never in "
    "the spec or the code."
)

# A spec is a technical artifact; Keel can lower the barrier to producing one and
# make its own choices legible, but it cannot let someone who does not know what a
# database is validate the answer. Say so once, in the footer — no modal, no
# repetition (spec A7).
_CEILING_NOTE = (
    "*This spec is a starting point for a coding agent and benefits from review by "
    "someone with software experience before you rely on it.*"
)

# The one string "Open questions" is allowed to be, and only when there is
# genuinely nothing to raise (no conflicts, no skipped slots, no failed checks).
_SENTINEL_LINE = "- None — every required dimension was addressed."


# --------------------------------------------------------------------------- #
# Synthesis (primary path)
# --------------------------------------------------------------------------- #
_SYNTHESIS_SYSTEM = """You are Keel. You write a software specification that a coding agent will \
implement literally. You are given a one-line project idea and the answers a developer gave to a \
fixed set of clarifying questions. Turn them into ONE coherent document.

Return ONLY a JSON object with exactly these keys, and no others:
  "context", "objective", "io_contract", "constraints", "acceptance_criteria",
  "build_order", "project_structure", "non_goals", "open_questions", "resolved_conflicts"
All but the last are the markdown BODY of that section — sentences and/or bullet lists. \
Do NOT include the section heading itself. Do NOT add, rename, reorder, or drop keys.
"resolved_conflicts" is a JSON ARRAY (possibly empty) — see the contradiction rule below.

How to write it:

- Write a document, not a form. Give each section as many sentences or bullets as its content \
  warrants. Never paste a single answer verbatim as the whole section.
- "objective": 2 to 4 sentences describing what is being built and the problem it solves. Never \
  echo the idea string back ("Deliver a working implementation of: ...") — that is banned.
- "io_contract": describe the actual inputs and outputs the project implies — the shape of a \
  request and a response, or of an input file and an output file, with the fields that matter. \
  Not just the fragment the developer typed. Then enumerate the interface surface IN FULL: \
  every command, endpoint, function, or screen named or implied by the answers, each on its \
  own line with its inputs and its output. List them all — not one example. If the answers \
  describe a data model, describe each entity and the fields it carries.
- "acceptance_criteria": a markdown bullet list of concrete, individually checkable statements \
  derived from the I/O contract and the definition of done. For anything with more than one \
  feature, give at least three.
- "build_order": a NUMBERED list of 3 to 6 stages, derived from the interface surface, the data \
  model, and the definition of done. Order by DEPENDENCY, not importance: the data model before \
  the routes or commands that read it, read paths before write paths, the happy path before \
  error handling. End each stage with one short clause on how to verify it works. If the answers \
  do not support real stages, give a minimal three-stage skeleton — never invent scope.
- "project_structure": a short directory / file tree derived from the constraints and the \
  interface surface, one line of purpose per entry. No file contents, no code. Begin the section \
  with one sentence stating this is a suggested starting layout, not a requirement, so an agent \
  in an existing repo does not restructure it.
- "constraints": the hard limits (language, offline, no paid APIs, dependency limits) AND the \
  runtime and scale. End it with the error-handling behaviour — what happens on bad input, a \
  missing or unreadable resource, and a partial failure — as its own short paragraph or bullet \
  group, drawn from the error-handling answer.
- Put every fact in the section it belongs to, regardless of which answer it arrived in. \
  Runtime and scale details are CONSTRAINTS — put them under "constraints", never under \
"context". "context" holds the original idea and background only.
- You are given a list of contradictions a separate check already found. For each one your \
  document silences — by writing every section for the more specific and more recently given \
  answer so the document stays internally consistent — add an entry to "resolved_conflicts": \
  {"conflict": "<the contradiction text you were given, verbatim or near>", "how": "<the one \
  choice you made, e.g. 'treated it as a browser app and kept the pages; dropped the no-UI \
  non-goal'>"}. If your document leaves a contradiction genuinely unresolved, do NOT invent a \
  resolution — leave it out of the array. If you were given no contradictions, return []. \
  Never state both sides of a conflict as true, and never list contradictions in \
  "open_questions" yourself — that section is rebuilt mechanically.
- Do NOT invent specifics the developer did not supply. No made-up numbers, versions, quotas, \
  latencies, request rates, or product names. If no traffic figure was given, write "low, \
  single-instance traffic" — not "0.3 requests/second". This is a hard rule. A plausible \
  invented number is worse than a qualitative phrase, because an agent treats it as a \
  requirement — never state a capacity, throughput, or volume figure that no answer supports.
- Architectural shape is in scope and often the most useful thing you can state: \
  server-rendered pages vs a single-page app, one file vs a database, sessions vs tokens, \
  synchronous vs queued. Name the shape and, in one clause, its trade-off. Do NOT name a \
  specific library, framework version, or package as a requirement unless the developer did — \
  library health sits past your knowledge cutoff and a version pin is a hallucination surface.
- Never emit a credential, secret, token, password, or key, even as an example. Use a \
  placeholder such as "the JWT signing secret, read from the JWT_SECRET environment variable". \
  Never state an authentication algorithm, hashing scheme, iteration count, or token lifetime \
  the developer did not supply — record the policy ("password login with server-side sessions") \
  and leave the parameters to implementation.
- "open_questions": a markdown bullet list of dimensions the developer genuinely left \
  unspecified. Do NOT put contradictions here — they are detected and recorded separately. \
  If there is genuinely nothing open, write exactly: \
  "- None — every required dimension was addressed."
"""


def _answers_block(session: SessionState) -> str:
    """The collected answers, one bullet per slot, in question order. Skipped or
    empty slots are marked so the model (and the conflict checker) can see them."""
    template = engine.load_template(session.template_name)
    visible = engine.visible_slots(session, template)
    lines: list[str] = []
    for slot in sorted(visible, key=lambda s: (s.priority, s.name)):
        state = session.slots.get(slot.name)
        if state is None or state.source == "skipped" or not state.value.strip():
            lines.append(f"- {slot.label} [{slot.name}]: (skipped by the developer — leave open)")
        else:
            lines.append(
                f"- {slot.label} [{slot.name}]: {state.value.strip()} (source: {state.source})"
            )
    return "\n".join(lines)


def _synthesis_user(session: SessionState) -> str:
    template = engine.load_template(session.template_name)
    lines = [
        f'Original idea: "{session.original_prompt}"',
        f"Project type: {template.name} — {template.title}",
        "",
        "Answers collected, one per dimension:",
        _answers_block(session),
    ]
    if session.conflicts:
        lines += [
            "",
            "Contradictions already detected by a separate check. Do NOT silently "
            "resolve or hide these — write each section truthfully; they are reported "
            "to the developer separately:",
        ]
        lines += [f"- {c.get('conflict', '').strip()}" for c in session.conflicts]
    lines += ["", "Write the specification now. Return only the JSON object."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Contradiction check (its own LLM call, made BEFORE synthesis)
# --------------------------------------------------------------------------- #
_CONFLICT_MAX_TOKENS = 900

_CONFLICT_SYSTEM = """You are Keel's contradiction checker. You are given a one-line project idea \
and the answers a developer gave to a fixed set of clarifying questions. Your ONLY job is to \
report genuine contradictions and unaddressed premises. You do NOT resolve them and you do NOT \
write any part of the specification.

Return ONLY a JSON object with one key:
  {"conflicts": [ {"slots": [...], "kind": "logical" | "feasibility", "conflict": "...", \
"suggested_resolution": "..."} ]}

For each entry:
- "slots": the names (in brackets above) of the answers that clash. Use the literal string \
  "original idea" when the clash is between the idea and the answers (premise drift, below).
- "kind": "logical" when the answers directly contradict each other or the idea; "feasibility" \
  when no single answer is self-contradictory but the COMBINATION cannot work in practice.
- "conflict": one sentence naming what cannot all be true at once.
- "suggested_resolution": one sentence naming the choice the developer has to make. For a \
  feasibility conflict, name the single answer most likely to need revising. Do not pick for them.

Look for three things:
1. Direct contradictions between answers ("kind": "logical"). For example: one answer says \
   "offline, standard library only" while another needs a hosted model or a paid API; one \
   answer lists authentication as a non-goal while another stores per-user data.
2. Premise drift ("kind": "logical"): a capability named or plainly implied by the ORIGINAL \
   IDEA that NONE of the answers account for — a chat or conversational interface, a \
   user-facing UI, scheduling or recurring runs, multiple users, stored history. Report it \
   with "original idea" in "slots".
3. Technical feasibility of the combination ("kind": "feasibility"): can the stated runtime \
   carry the stated scale (a single-process development server cannot serve hundreds of \
   concurrent users); do the dependency limits permit the stated capabilities; does the \
   storage choice support the stated concurrency; do the interfaces require infrastructure \
   the constraints forbid. Each answer alone is reasonable; together they do not hold up.

Rules:
- Only report a conflict you can state concretely from the given text. Do not pad the list, do \
  not speculate, do not invent facts.
- If there are genuinely none, return {"conflicts": []}.
- Never output specification prose or a resolved version of the spec. Only the list."""


def check_conflicts(
    session: SessionState, *, provider: "llm.Provider | None"
) -> tuple[list[dict], str | None]:
    """One capped LLM call, made before :func:`synthesize_spec`, that looks for
    contradictions between the answers and for capabilities in the original idea
    that no answer covers. Returns ``(conflicts, None)`` on success or
    ``([], reason)`` on failure — the reason is also recorded on
    ``session.conflict_check_error`` so the document notes the check could not run
    and ``session.degraded`` reflects it. Never silently skipped."""
    user = (
        f'Original idea: "{session.original_prompt}"\n\n'
        f"Answers collected, one per dimension:\n{_answers_block(session)}\n\n"
        "Return the JSON object now."
    )
    result, error = engine.capped_complete_json(
        session, _CONFLICT_SYSTEM, user, provider=provider, max_tokens=_CONFLICT_MAX_TOKENS
    )
    if error is not None:
        session.conflict_check_error = error
        return [], error
    session.conflict_check_error = None

    raw = (result or {}).get("conflicts", [])
    conflicts: list[dict] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("conflict", "")).strip()
            if not text:
                continue
            slots_raw = item.get("slots", [])
            slots = (
                [str(s).strip() for s in slots_raw if str(s).strip()]
                if isinstance(slots_raw, list)
                else []
            )
            kind = str(item.get("kind", "")).strip().lower()
            conflicts.append(
                {
                    "slots": slots,
                    "kind": kind if kind in ("logical", "feasibility") else "logical",
                    "conflict": text,
                    "suggested_resolution": str(item.get("suggested_resolution", "")).strip(),
                }
            )
    return conflicts, None


def synthesize_spec(
    session: SessionState,
    *,
    provider: "llm.Provider | None",
    conflicts: list[dict] | None = None,
    conflict_error: str | None = None,
) -> tuple[str | None, str | None]:
    """One capped LLM call, then deterministic validation, producing the full
    markdown document. Returns ``(markdown, None)`` on success or
    ``(None, reason)`` on any failure — the caller then uses
    :func:`render_markdown`.

    ``conflicts`` / ``conflict_error`` come from :func:`check_conflicts`, run
    first. They are given to the synthesis model as context; afterwards each is
    re-validated against the document that was produced (see
    :func:`_revalidate_conflicts`). Survivors are written into "Open questions"
    mechanically here — the model never controls whether a conflict is reported —
    and resolved ones land on ``session.resolved_conflicts``. When ``conflicts``
    is omitted, the value already on ``session`` is used."""
    if conflicts is not None:
        session.conflicts = conflicts
    if conflict_error is not None:
        session.conflict_check_error = conflict_error

    result, error = engine.capped_complete_json(
        session,
        _SYNTHESIS_SYSTEM,
        _synthesis_user(session),
        provider=provider,
        max_tokens=SYNTHESIS_MAX_TOKENS,
    )
    if error is not None:
        return None, error
    return _assemble_and_validate(session, result or {})


# --------------------------------------------------------------------------- #
# Post-synthesis validation (deterministic — no second LLM call)
# --------------------------------------------------------------------------- #
def _assemble_and_validate(
    session: SessionState, sections: dict
) -> tuple[str | None, str | None]:
    template = engine.load_template(session.template_name)

    quick = session.depth == "quick" and session.mode != "guided"

    bodies: dict[str, str] = {}
    for key in _MODEL_SECTION_KEYS:
        if key in _SOFT_MODEL_KEYS and quick:
            continue  # Build order / Project structure are suppressed at Quick depth
        raw_val = sections.get(key, "")
        if isinstance(raw_val, list):  # some models return a bullet array, not a string
            raw_val = "\n".join(f"- {x}" if not str(x).lstrip().startswith(("-", "*")) else str(x)
                                for x in raw_val)
        raw_val = str(raw_val)
        if "\\n" in raw_val and "\n" not in raw_val:  # model escaped its newlines
            raw_val = raw_val.replace("\\n", "\n")
        body = _strip_injected_headings(raw_val.strip())
        if not body:
            if key in _SOFT_MODEL_KEYS:
                continue  # additive section — just omit it, never fall back
            return None, f"synthesis produced no '{key}' section"
        bodies[key] = body

    # "Decisions Keel made for you" is assembled here, never by the model — one
    # entry per slot the user delegated ("decide for me"), with the reason and a
    # revisit condition recorded alongside the value. Empty -> the section is
    # omitted entirely (see _assemble); it is never omitted when a keel_decided
    # slot exists.
    bodies["decisions"] = _build_decisions(session, template)

    # Bug 2: the conflict list was computed against the *answers*. Synthesis may
    # have silenced some of them. Re-validate each against the document it just
    # produced — resolved ones move to session.resolved_conflicts (shown in the
    # UI, never in Open questions); only survivors go on to _build_open_questions.
    surviving, resolved = _revalidate_conflicts(
        session, session.conflicts, bodies, sections.get("resolved_conflicts")
    )
    session.conflicts = surviving
    session.resolved_conflicts = resolved

    # B4: every capacity / throughput / volume figure in the document must trace
    # to an answer. Unsupported ones are rewritten qualitatively here, and each
    # rewrite becomes an "Open questions" note.
    bodies, figure_notes = _scrub_unsupported_figures(bodies, _slot_number_blob(session))

    # "Open questions" is rebuilt here, in Python: the surviving conflict list,
    # plus deterministic checks the model does not run (fabricated numbers,
    # criteria that restate constraints, missing sensitive-domain disclaimer).
    # The model never decides whether a conflict is reported.
    bodies["open_questions"] = _build_open_questions(
        session, template, bodies, extra_notes=figure_notes
    )

    # Hardcoded-secret scan + scrub.
    scrubbed = False
    for key, body in bodies.items():
        cleaned, hit = _scrub_secrets(body)
        bodies[key] = cleaned
        scrubbed = scrubbed or hit
    if scrubbed:
        bodies["open_questions"] = bodies["open_questions"].rstrip() + "\n" + _SCRUB_NOTE

    # No template instruction text may survive into the document.
    for slot in template.slots:
        ds = slot.default_strategy.strip()
        if ds and any(ds in body for body in bodies.values()):
            return None, "synthesis echoed template instruction text"

    md = _assemble(session, bodies)

    # Structure guard: the only H2s are ours, in order — allowing for optional
    # sections (Decisions) that are omitted when empty.
    h2s = [ln[3:].strip() for ln in md.splitlines() if ln.startswith("## ")]
    if h2s != _rendered_headings(bodies):
        return None, "synthesis altered the section structure"
    return md, None


def _rendered_headings(bodies: dict[str, str]) -> list[str]:
    """The headings that should appear, in order: every section with a non-empty
    body. Optional sections (Decisions) drop out when empty; required sections
    are always present by the time this is called."""
    return [
        heading
        for heading, key in SECTION_ORDER
        if key not in _OPTIONAL_SECTION_KEYS or bodies.get(key, "").strip()
    ]


def _assemble(session: SessionState, bodies: dict[str, str]) -> str:
    out: list[str] = [f"# {session.title()}", ""]
    for heading, key in SECTION_ORDER:
        body = bodies.get(key, "").strip()
        if not body and key in _OPTIONAL_SECTION_KEYS:
            continue
        out += [f"## {heading}", "", body, ""]
    if session.degraded:
        out += [_DEGRADED_NOTE, ""]
    out += [
        "---",
        f"*Generated by Keel on {session.created_date}.*",
        "",
        _CEILING_NOTE,
        "",
    ]
    return "\n".join(out)


def _strip_injected_headings(body: str) -> str:
    """Remove any ATX markdown heading the model put inside a section body —
    headings are this module's to own."""
    return re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*", "", body)


# --------------------------------------------------------------------------- #
# "Decisions Keel made for you" — assembled in Python from keel_decided slots
# --------------------------------------------------------------------------- #
def _build_decisions(session: SessionState, template: Template) -> str:
    """One bullet per slot the user delegated to Keel, in priority order: the
    decision, the one-line reason, and a "revisit this if …" clause. Returns
    "" when the user delegated nothing — the section is then omitted. It is
    never omitted while any slot is keel_decided."""
    lines: list[str] = []
    for slot in sorted(template.slots, key=lambda s: (s.priority, s.name)):
        state = session.slots.get(slot.name)
        if state is None or state.source != "keel_decided":
            continue
        value = state.value.strip() or "left to Keel's judgement"
        line = f"- **{slot.label}:** {value}"
        if state.rationale.strip():
            line += f" — {state.rationale.strip()}"
        cond = state.revisit_if.strip()
        if cond:
            cond = re.sub(r"^(revisit this if|revisit if|if)\s+", "", cond, flags=re.I)
            line += f" _Revisit this if {cond.rstrip('.')}._"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# "Open questions" is assembled in Python, never trusted to the model
# --------------------------------------------------------------------------- #
def _conflict_bullets(conflicts: list[dict] | None) -> list[str]:
    out: list[str] = []
    for c in conflicts or []:
        slots = ", ".join(c.get("slots") or []) or "answers"
        res = str(c.get("suggested_resolution") or "").strip()
        if c.get("kind") == "feasibility":
            # Not a self-contradiction the developer made — the combination just
            # does not hold up, and usually one answer needs revising.
            line = f"- **Feasibility ({slots}):** {str(c.get('conflict', '')).strip()}"
            if res:
                line += f" _Likely fix:_ {res}"
        else:
            line = f"- **Conflict ({slots}):** {str(c.get('conflict', '')).strip()}"
            if res:
                line += f" _Suggested resolution:_ {res}"
        out.append(line)
    return out


def _skipped_slot_bullets(session: SessionState, template: Template) -> list[str]:
    out: list[str] = []
    for name in session.skipped_slots():
        slot = template.slot(name)
        label = slot.label if slot else name
        out.append(
            f"- **{label}:** left unanswered — confirm with the developer before implementing."
        )
    return out


_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Figures that read as targets/thresholds rather than incidental counts
# ("exactly one", "at least three" carry no digits and are fine).
_THRESHOLD_RE = re.compile(
    r"(?:[<>]=?|≥|≤|=|within|under|over|at least|at most|no more than|up to|below|above)\s*"
    r"\$?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|k\b|m\b|bn\b|ms\b|s\b|sec\b|seconds|minutes|hours|/s|per second|rps|qps|requests?)?"
    r"|\b\d[\d,]*\.\d+\b"
    r"|\b\d[\d,]*\s*%"
    r"|\b\d{4,}\b",
    re.I,
)


def _numbers_in(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(text or "")}


def _slot_number_blob(session: SessionState) -> set[str]:
    parts = [session.original_prompt] + [v.value for v in session.slots.values() if v.value]
    return _numbers_in(" ".join(parts))


# --------------------------------------------------------------------------- #
# B4: every number in the rendered document must trace to an answer
# --------------------------------------------------------------------------- #
_FIGURE_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_YEARISH = re.compile(r"^(?:19|20)\d{2}$")
_COMPARATOR_BEFORE = re.compile(
    r"(?:[<>]=?|≥|≤|up to|at least|at most|no more than|under|over|within|below|above|"
    r"roughly|around|about|approximately|~)\s*$",
    re.I,
)
_QUANTITY_AFTER = re.compile(
    r"^\s*(?:%|per cent|percent|users?|people|concurren\w*|sessions?|customers?|visitors?|"
    r"requests?|rps|qps|per second|per minute|per hour|per day|/s|queries|hits?|"
    r"transactions?|rows?|records?|items?|entries|bookings?|orders?|rooms?|hotels?|"
    r"documents?|files?|messages?|connections?|threads?|workers?|[kmgt]b\b|bytes?|"
    r"megabytes?|gigabytes?)",
    re.I,
)


def _num_val(bare: str) -> float | None:
    try:
        return float(bare)
    except ValueError:
        return None


def _figure_is_supported(bare: str, known: set[str]) -> bool:
    """Whether a numeric literal traces to an answer — verbatim, or as a plain
    arithmetic derivation of one (a rate from a daily total, a total from a
    per-item count). Errs toward "supported": a false accept is better than
    rewriting a legitimate figure."""
    if bare in known:
        return True
    val = _num_val(bare)
    if val is None:
        return False
    for k in known:
        kv = _num_val(k)
        if not kv:
            continue
        for n in (2, 3, 4, 5, 6, 7, 10, 12, 24, 30, 52, 60, 100, 365, 1000, 3600, 86400):
            if abs(val - kv * n) < 1e-6 or abs(val - kv / n) < 1e-6:
                return True
    return False


def _qualitative_for(after: str, pct: bool) -> str:
    a = after.lower()
    if pct:
        return "a defined threshold"
    if re.search(r"\b(users?|people|concurren\w*|sessions?|customers?|visitors?)\b", a):
        return "a small number of users"
    if re.search(r"\b(requests?|rps|qps|per second|per minute|per hour|/s|queries|hits?|"
                r"transactions?)\b", a):
        return "a low, unspecified rate"
    if re.search(r"\b([kmgt]b|bytes?|megabytes?|gigabytes?)\b", a):
        return "a modest size"
    if re.search(r"\b(rows?|records?|items?|entries|bookings?|orders?|rooms?|hotels?|"
                r"documents?|files?|messages?)\b", a):
        return "an unspecified number"
    return "an unspecified amount"


def _scrub_unsupported_figures(
    bodies: dict[str, str], known: set[str]
) -> tuple[dict[str, str], list[str]]:
    """Replace every capacity / throughput / volume figure in the rendered
    sections that traces to no answer with qualitative wording, and return one
    note per distinct replacement. A plausible invented number is worse than
    vagueness — an agent treats it as a requirement (spec B4)."""
    notes: dict[str, str] = {}

    def scrub(section_key: str, body: str) -> str:
        def repl(m: re.Match) -> str:
            token = m.group(0)
            s, i, j = m.string, m.start(), m.end()
            bare = re.sub(r"[,\s%]", "", token)
            if not bare:
                return token
            # part of an identifier / version / hyphenated code (HS256, v2, P95, SHA-256)
            if i > 0 and (s[i - 1].isalpha() or s[i - 1] == "-"):
                return token
            tail = s[j] if j < len(s) else ""
            if tail.isalpha() or tail == "-":
                return token
            # ordered-list marker "1." / "2." at the start of a line
            line_start = s.rfind("\n", 0, i) + 1
            if s[line_start:i].strip() == "" and tail == ".":
                return token
            if _YEARISH.match(bare) or _figure_is_supported(bare, known):
                return token
            pct = "%" in token
            after = s[j:j + 28]
            before = s[max(0, i - 18):i]
            challenged = (
                pct
                or bool(_COMPARATOR_BEFORE.search(before))
                or bool(_QUANTITY_AFTER.match(after))
                or (_num_val(bare) is not None and _num_val(bare) >= 1000)
            )
            if not challenged:
                return token
            notes.setdefault(token.strip(), section_key)
            return _qualitative_for(after, pct)

        return _FIGURE_RE.sub(repl, body)

    for key in list(bodies):
        if key == "open_questions":
            continue
        bodies[key] = scrub(key, bodies[key])

    note_lines = [
        f'- **Unverified figure:** "{orig}" appeared in the '
        f'{sect.replace("_", " ")} section but no answer specified it; it has been '
        "replaced with qualitative wording. Decide the real value before starting."
        for orig, sect in notes.items()
    ]
    return bodies, note_lines


def _unverified_figure_bullets(acceptance_body: str, known: set[str]) -> list[str]:
    """Numeric thresholds in the acceptance criteria that trace back to no answer.
    Arithmetic derivations of a supplied figure are a tolerable false positive —
    this only flags, it does not fail the render."""
    flagged: list[str] = []
    seen: set[str] = set()
    for m in _THRESHOLD_RE.finditer(acceptance_body or ""):
        frag = m.group(0).strip()
        nums = _numbers_in(frag)
        if not nums or (nums & known):
            continue
        if frag.lower() in seen:
            continue
        seen.add(frag.lower())
        flagged.append(frag)
    if not flagged:
        return []
    joined = ", ".join(f"`{f}`" for f in flagged)
    return [
        f"- **Unverified figure(s):** the acceptance criteria cite {joined}, which no "
        "answer specified. Confirm the target or remove the number."
    ]


_NEG_START = re.compile(
    r"^\s*[-*]\s*(no |not |never |without |does not |doesn't |cannot |can't |won't |will not |there is no |there are no )",
    re.I,
)
_STOP = {
    "the", "a", "an", "is", "are", "be", "to", "of", "and", "or", "no", "not",
    "with", "for", "in", "on", "any", "that", "this", "it", "as", "by",
}


def _line_tokens(line: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", line.lower())) - _STOP


# --------------------------------------------------------------------------- #
# Bug 2: re-validate conflicts against the document synthesis actually produced
# --------------------------------------------------------------------------- #
_RESOLVED_MATCH_THRESHOLD = 0.34   # Jaccard on salient tokens: model-report <-> conflict
_DOC_COVERAGE_THRESHOLD = 0.7      # fraction of a conflict's salient words now in the doc

# A conflict is only ever "resolved by the document" if it is about a missing or
# forbidden capability — the doc can then be shown to describe it. A direct value
# contradiction (offline vs. a hosted model) is never resolved just because both
# terms reappear in the prose. Matched as whole words, so "cannot" is not "not".
_DRIFT_MARKER_WORDS = frozenset({
    "no", "not", "does", "doesn't", "lacks", "lack", "missing", "none", "never",
    "without", "absent", "undefined", "unspecified", "unaddressed", "omits", "omit",
    "forbid", "forbids", "forbidden", "excludes", "rules",
})


def _salient(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", text.lower())} - _STOP


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _model_resolution_for(conflict_text: str, model_resolved: list) -> str | None:
    """The model's own stated resolution for this conflict, if it reported one
    close enough to be the same conflict."""
    ct = _salient(conflict_text)
    best_score, best_how = 0.0, ""
    for r in model_resolved:
        if not isinstance(r, dict):
            continue
        score = _jaccard(ct, _salient(str(r.get("conflict", ""))))
        if score > best_score:
            best_score, best_how = score, str(r.get("how", "")).strip()
    if best_score >= _RESOLVED_MATCH_THRESHOLD:
        return best_how or "resolved during synthesis"
    return None


def _doc_resolves(conflict: dict, doc_blob: str) -> bool:
    """Whether the rendered document plainly now covers what the conflict said
    was missing. Only applied to *premise drift* — a conflict the checker tagged
    with "original idea", i.e. "a capability in the idea that no answer covers".
    A direct value clash between two real slots is never dropped this way, even
    if both its terms happen to reappear in the prose."""
    slots = [str(s).strip().lower() for s in (conflict.get("slots") or [])]
    if "original idea" not in slots:
        return False
    ctext = str(conflict.get("conflict", ""))
    words = set(re.findall(r"[a-z']+", ctext.lower()))
    if not (words & _DRIFT_MARKER_WORDS):
        return False
    tokens = _salient(ctext)
    if len(tokens) < 3:
        return False
    present = sum(1 for t in tokens if t in doc_blob)
    return present / len(tokens) >= _DOC_COVERAGE_THRESHOLD


def _revalidate_conflicts(
    session: SessionState,
    pre_conflicts: list[dict] | None,
    bodies: dict[str, str],
    model_resolved,
) -> tuple[list[dict], list[dict]]:
    """Split the pre-synthesis conflicts into (surviving, resolved).

    A conflict is resolved when the synthesis model reported resolving it, or
    when the rendered document plainly now covers a capability the conflict said
    was missing. Anything the model claims to have resolved that was never raised
    — and anything the document silently resolved without a report — is logged.
    """
    pre = [c for c in (pre_conflicts or []) if isinstance(c, dict)]
    model_resolved = [r for r in (model_resolved or []) if isinstance(r, dict)]
    doc_blob = " ".join(
        bodies.get(k, "")
        for k in ("context", "objective", "io_contract", "constraints",
                  "acceptance_criteria", "non_goals")
    ).lower()

    surviving: list[dict] = []
    resolved: list[dict] = []
    for c in pre:
        ctext = str(c.get("conflict", "")).strip()
        how = _model_resolution_for(ctext, model_resolved)
        if how is not None:
            resolved.append({**c, "resolution": how})
            continue
        if _doc_resolves(c, doc_blob):
            resolved.append({
                **c,
                "resolution": "Resolved in the synthesized document; the check ran "
                "against the answers, before synthesis.",
            })
            _log.warning("synthesis silently resolved a conflict it did not report: %s", ctext)
            continue
        surviving.append(c)

    pre_salient = [_salient(str(c.get("conflict", ""))) for c in pre]
    for r in model_resolved:
        rt = str(r.get("conflict", "")).strip()
        if not rt:
            continue
        if max((_jaccard(_salient(rt), p) for p in pre_salient), default=0.0) < \
                _RESOLVED_MATCH_THRESHOLD:
            _log.warning(
                "synthesis reported resolving a conflict absent from the "
                "pre-synthesis set: %s", rt
            )
    return surviving, resolved


def _restatement_bullets(
    acceptance_body: str, non_goals_body: str, constraints_body: str
) -> list[str]:
    """Acceptance criteria that merely restate a non-goal or a constraint instead
    of describing something observable."""
    ref = [
        _line_tokens(ln)
        for ln in (non_goals_body + "\n" + constraints_body).splitlines()
        if ln.strip()
    ]
    flagged: list[str] = []
    for ln in acceptance_body.splitlines():
        if not ln.strip():
            continue
        at = _line_tokens(ln)
        if not at:
            continue
        neg = bool(_NEG_START.match(ln))
        best = max(
            (len(at & rt) / len(at | rt) for rt in ref if rt),
            default=0.0,
        )
        if best >= 0.7 or (neg and best >= 0.4):
            flagged.append(ln.strip().lstrip("-* ").strip())
    if not flagged:
        return []
    joined = "; ".join(f'"{f}"' for f in flagged[:3])
    return [
        f"- **Non-testable criteria:** {joined} — these restate a constraint or non-goal "
        "rather than describing an observable outcome. Rephrase or move them."
    ]


_SENSITIVE_TERMS = (
    "health", "medical", "clinic", "patient", "diagnos", "symptom", "therap",
    "wellness", "blood pressure", "heart rate", "medication", "prescription",
    "legal advice", "lawyer", "attorney", "litigation", "lawsuit", "contract review",
    "financial", "finance", "investment", "portfolio", "loan", "credit score",
    "trading", "insurance", "retirement", "mortgage",
)
_DISCLAIMER_TERMS = (
    "disclaimer", "not medical advice", "not legal advice", "not financial advice",
    "informational purposes", "consult a", "consult your", "professional advice",
    "scope of use", "not a substitute",
)


def _sensitive_domain_bullet(session: SessionState, doc_bodies: dict[str, str]) -> str | None:
    hay = " ".join(
        [session.original_prompt] + [v.value for v in session.slots.values() if v.value]
    ).lower()
    if not any(term in hay for term in _SENSITIVE_TERMS):
        return None
    doc = " ".join(doc_bodies.values()).lower()
    if any(term in doc for term in _DISCLAIMER_TERMS):
        return None
    return (
        "- **Sensitive domain:** this spec touches health, medical, legal, or financial "
        "matters but states no disclaimer or scope-of-use limitation. Decide whether one "
        "is required — Keel does not draft it for you."
    )


def _conflict_unavailable_bullet(reason: str | None) -> list[str]:
    if not reason:
        return []
    return [
        f"- **Conflict check unavailable:** {reason} — review the answers for "
        "contradictions by hand before handing this to an agent."
    ]


def _reference_bullet(session: SessionState) -> list[str]:
    """Provenance line so a scraped reference travels with the document."""
    ref = getattr(session, "reference", None)
    if not (ref and ref.confirmed):
        return []
    if ref.mode == "image":
        src = f"an uploaded UI image ({ref.query})"
    elif ref.source_urls:
        src = ", ".join(ref.source_urls)
    else:
        return []
    return [
        f"- **Reference used:** structural cues (entities, surfaces, likely non-goals) "
        f"were taken from {src}. Product names, wording, and visual design were not "
        "carried across — confirm the borrowed structure actually fits this build."
    ]


def _build_open_questions(
    session: SessionState,
    template: Template,
    bodies: dict[str, str],
    *,
    extra_notes: list[str] | None = None,
) -> str:
    """Rebuild the "Open questions" body deterministically from the model's draft
    plus everything Python is responsible for. The sole-"None" sentinel survives
    only when there is genuinely nothing to raise."""
    additions: list[str] = []
    additions += _conflict_bullets(session.conflicts)
    additions += _conflict_unavailable_bullet(session.conflict_check_error)
    additions += _reference_bullet(session)
    additions += list(extra_notes or [])
    additions += _unverified_figure_bullets(
        bodies.get("acceptance_criteria", ""), _slot_number_blob(session)
    )
    additions += _restatement_bullets(
        bodies.get("acceptance_criteria", ""),
        bodies.get("non_goals", ""),
        bodies.get("constraints", ""),
    )
    sd = _sensitive_domain_bullet(session, bodies)
    if sd:
        additions.append(sd)

    must_expand = bool(
        session.conflicts
        or session.conflict_check_error
        or session.skipped_slots()
        or additions
    )
    model_lines = [ln for ln in bodies.get("open_questions", "").splitlines() if ln.strip()]
    if must_expand:
        model_lines = [ln for ln in model_lines if _SENTINEL_LINE not in ln]
        if not model_lines:
            model_lines = _skipped_slot_bullets(session, template)

    final = model_lines + additions
    if not final:
        final = [_SENTINEL_LINE]
    return "\n".join(final).strip()


_SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?i)\bsecret\s*[:=]?\s*['\"]([^'\"]{3,})['\"]"),
    re.compile(
        r"(?i)\b(?:password|passwd|pwd|api[_-]?key|secret[_-]?key|client[_-]?secret|"
        r"access[_-]?token|auth[_-]?token|bearer)\s*[:=]\s*([^\s'\"]{4,})"
    ),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{20,}"),
]
_QUOTED_LITERAL = re.compile(r"['\"]([A-Za-z0-9+/=_\-]{24,})['\"]")
_PLACEHOLDER = "«value read from an environment variable»"


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {ch: s.count(ch) for ch in set(s)}
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _scrub_secrets(text: str) -> tuple[str, bool]:
    hit = False

    for pat in _SECRET_PATTERNS:
        if pat.search(text):
            hit = True
            text = pat.sub(_PLACEHOLDER, text)

    def _maybe_secret(m: re.Match) -> str:
        nonlocal hit
        token = m.group(1)
        if _shannon_entropy(token) >= 3.6:
            hit = True
            return _PLACEHOLDER
        return m.group(0)

    text = _QUOTED_LITERAL.sub(_maybe_secret, text)
    return text, hit


# --------------------------------------------------------------------------- #
# Deterministic rendering (fallback path)
# --------------------------------------------------------------------------- #
def _resolved(session: SessionState, names: list[str]) -> list[tuple[str, str]]:
    template = engine.load_template(session.template_name)
    out: list[tuple[str, str]] = []
    for name in names:
        slot = template.slot(name)
        state = session.slots.get(name)
        if slot is None or state is None:
            continue
        if state.source == "skipped" or not state.value.strip():
            continue
        out.append((slot.label, state.value.strip()))
    return out


def _section_slot_names(template: Template, section: str) -> list[str]:
    return [
        s.name
        for s in sorted(template.slots, key=lambda s: (s.priority, s.name))
        if s.section == section
    ]


def render_markdown(session: SessionState) -> str:
    template = engine.load_template(session.template_name)
    lines: list[str] = [f"# {session.title()}", ""]

    lines += ["## Context", ""]
    lines.append(
        f'This spec was compiled by Keel from the idea: "{session.original_prompt}".'
    )
    context_pairs = _resolved(session, _section_slot_names(template, "context"))
    if context_pairs:
        lines.append("")
        for label, value in context_pairs:
            lines.append(f"- **{label}:** {value}")
    lines.append("")

    lines += ["## Objective", ""]
    lines.append(
        f"Deliver a working implementation of: {session.original_prompt.rstrip('.')}."
    )
    lines.append("")

    for heading, section in (
        ("Input / Output contract", "io_contract"),
        ("Constraints", "constraints"),
        ("Acceptance criteria", "acceptance"),
    ):
        lines += [f"## {heading}", ""]
        pairs = _resolved(session, _section_slot_names(template, section))
        if pairs:
            for _, value in pairs:
                lines += [value, ""]
        else:
            lines += ["_Not specified — see Open questions below._", ""]

    decisions = _build_decisions(session, template)
    if decisions.strip():
        lines += ["## Decisions Keel made for you", "", decisions.strip(), ""]

    lines += ["## Non-goals", ""]
    non_goal_pairs = _resolved(session, _section_slot_names(template, "non_goals"))
    if non_goal_pairs:
        for _, value in non_goal_pairs:
            lines += [value, ""]
    else:
        lines += ["_Not specified — see Open questions below._", ""]

    lines += ["## Open questions", ""]
    oq: list[str] = _skipped_slot_bullets(session, template)
    oq += _conflict_bullets(session.conflicts)
    oq += _conflict_unavailable_bullet(session.conflict_check_error)
    oq += _reference_bullet(session)
    sd = _sensitive_domain_bullet(
        session,
        {"blob": " ".join(v.value for v in session.slots.values() if v.value)},
    )
    if sd:
        oq.append(sd)
    if not oq:
        oq = [_SENTINEL_LINE]
    lines += oq
    if session.degraded:
        lines += ["", _DEGRADED_NOTE]
    lines.append("")

    lines += [
        "---",
        f"*Generated by Keel on {session.created_date}.*",
        "",
        _CEILING_NOTE,
        "",
    ]
    return "\n".join(lines)
