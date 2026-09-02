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

# (heading rendered, JSON key the synthesis model must return)
SECTION_ORDER: list[tuple[str, str]] = [
    ("Context", "context"),
    ("Objective", "objective"),
    ("Input / Output contract", "io_contract"),
    ("Constraints", "constraints"),
    ("Acceptance criteria", "acceptance_criteria"),
    ("Non-goals", "non_goals"),
    ("Open questions", "open_questions"),
]
_HEADINGS = [h for h, _ in SECTION_ORDER]

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
  "non_goals", "open_questions", "resolved_conflicts"
The first seven values are the markdown BODY of that section — sentences and/or bullet lists. \
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
  single-instance traffic" — not "0.3 requests/second". This is a hard rule.
- Never emit a credential, secret, token, password, or key, even as an example. Use a \
  placeholder such as "the JWT signing secret, read from the JWT_SECRET environment variable".
- "open_questions": a markdown bullet list of dimensions the developer genuinely left \
  unspecified. Do NOT put contradictions here — they are detected and recorded separately. \
  If there is genuinely nothing open, write exactly: \
  "- None — every required dimension was addressed."
"""


def _answers_block(session: SessionState) -> str:
    """The collected answers, one bullet per slot, in question order. Skipped or
    empty slots are marked so the model (and the conflict checker) can see them."""
    template = engine.load_template(session.template_name)
    lines: list[str] = []
    for slot in sorted(template.slots, key=lambda s: (s.priority, s.name)):
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
  {"conflicts": [ {"slots": [...], "conflict": "...", "suggested_resolution": "..."} ]}

For each entry:
- "slots": the names (in brackets above) of the answers that clash. Use the literal string \
  "original idea" when the clash is between the idea and the answers (premise drift, below).
- "conflict": one sentence naming what cannot all be true at once.
- "suggested_resolution": one sentence naming the choice the developer has to make. Do not \
  pick for them.

Look for exactly two things:
1. Direct contradictions between answers. For example: one answer says "offline, standard \
   library only" while another needs a hosted model or a paid API; one answer lists \
   authentication as a non-goal while another stores per-user data; a stated runtime that \
   cannot deliver a stated capability.
2. Premise drift: a capability named or plainly implied by the ORIGINAL IDEA that NONE of the \
   answers account for — a chat or conversational interface, a user-facing UI, scheduling or \
   recurring runs, multiple users, stored history. Report it with "original idea" in "slots".

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
            conflicts.append(
                {
                    "slots": slots,
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

    bodies: dict[str, str] = {}
    for _, key in SECTION_ORDER:
        raw_val = sections.get(key, "")
        if isinstance(raw_val, list):  # some models return a bullet array, not a string
            raw_val = "\n".join(f"- {x}" if not str(x).lstrip().startswith(("-", "*")) else str(x)
                                for x in raw_val)
        raw_val = str(raw_val)
        if "\\n" in raw_val and "\n" not in raw_val:  # model escaped its newlines
            raw_val = raw_val.replace("\\n", "\n")
        body = _strip_injected_headings(raw_val.strip())
        if not body:
            return None, f"synthesis produced no '{key}' section"
        bodies[key] = body

    # Bug 2: the conflict list was computed against the *answers*. Synthesis may
    # have silenced some of them. Re-validate each against the document it just
    # produced — resolved ones move to session.resolved_conflicts (shown in the
    # UI, never in Open questions); only survivors go on to _build_open_questions.
    surviving, resolved = _revalidate_conflicts(
        session, session.conflicts, bodies, sections.get("resolved_conflicts")
    )
    session.conflicts = surviving
    session.resolved_conflicts = resolved

    # "Open questions" is rebuilt here, in Python: the surviving conflict list,
    # plus deterministic checks the model does not run (fabricated numbers,
    # criteria that restate constraints, missing sensitive-domain disclaimer).
    # The model never decides whether a conflict is reported.
    bodies["open_questions"] = _build_open_questions(session, template, bodies)

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

    # Structure guard: the only H2s are ours, in order.
    h2s = [ln[3:].strip() for ln in md.splitlines() if ln.startswith("## ")]
    if h2s != _HEADINGS:
        return None, "synthesis altered the section structure"
    return md, None


def _assemble(session: SessionState, bodies: dict[str, str]) -> str:
    out: list[str] = [f"# {session.title()}", ""]
    for heading, key in SECTION_ORDER:
        out += [f"## {heading}", "", bodies[key].strip(), ""]
    if session.degraded:
        out += [_DEGRADED_NOTE, ""]
    out += ["---", f"*Generated by Keel on {session.created_date}.*", ""]
    return "\n".join(out)


def _strip_injected_headings(body: str) -> str:
    """Remove any ATX markdown heading the model put inside a section body —
    headings are this module's to own."""
    return re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*", "", body)


# --------------------------------------------------------------------------- #
# "Open questions" is assembled in Python, never trusted to the model
# --------------------------------------------------------------------------- #
def _conflict_bullets(conflicts: list[dict] | None) -> list[str]:
    out: list[str] = []
    for c in conflicts or []:
        slots = ", ".join(c.get("slots") or []) or "answers"
        line = f"- **Conflict ({slots}):** {str(c.get('conflict', '')).strip()}"
        res = str(c.get("suggested_resolution") or "").strip()
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


def _build_open_questions(
    session: SessionState, template: Template, bodies: dict[str, str]
) -> str:
    """Rebuild the "Open questions" body deterministically from the model's draft
    plus everything Python is responsible for. The sole-"None" sentinel survives
    only when there is genuinely nothing to raise."""
    additions: list[str] = []
    additions += _conflict_bullets(session.conflicts)
    additions += _conflict_unavailable_bullet(session.conflict_check_error)
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
        ("Non-goals", "non_goals"),
    ):
        lines += [f"## {heading}", ""]
        pairs = _resolved(session, _section_slot_names(template, section))
        if pairs:
            for _, value in pairs:
                lines += [value, ""]
        else:
            lines += ["_Not specified — see Open questions below._", ""]

    lines += ["## Open questions", ""]
    oq: list[str] = _skipped_slot_bullets(session, template)
    oq += _conflict_bullets(session.conflicts)
    oq += _conflict_unavailable_bullet(session.conflict_check_error)
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

    lines += ["---", f"*Generated by Keel on {session.created_date}.*", ""]
    return "\n".join(lines)
