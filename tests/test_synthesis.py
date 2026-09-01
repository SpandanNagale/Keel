"""Change 1 + 3: the synthesis pass and its deterministic validation.

The LLM is stubbed to return a section dict (or a failure tuple); these tests
exercise the assembly, the structure guard, the secret scrub, the
default_strategy check, and the call cap — not model quality.
"""
from __future__ import annotations

import re

import pytest

from keel import engine, llm, render

_HEADINGS = [h for h, _ in render.SECTION_ORDER]

_FULL_SECTIONS = {
    "context": "A small personal tool. The person has hundreds of browser bookmarks "
               "and wants them grouped so they can find things again.",
    "objective": "Build a script that reads exported bookmarks and writes them back "
                 "grouped into topic clusters. It replaces manual sorting. The output "
                 "is a single file the person can open. Success is a usable grouping.",
    "io_contract": "- Input: a `bookmarks.html` export\n"
                   "- Output: `clusters.json`, mapping a topic label to a list of "
                   "`{title, url}` objects",
    "constraints": "- Runs as a one-off local Python script\n"
                   "- Low, single-instance workload\n"
                   "- No paid APIs",
    "acceptance_criteria": "- Running once produces `clusters.json`\n"
                           "- Every input bookmark appears in exactly one cluster\n"
                           "- Cluster labels are human-readable\n"
                           "- Re-running on the same input is stable",
    "non_goals": "- No browser extension\n- No live sync\n- No web UI\n- No database",
    "open_questions": "- None — every required dimension was addressed.",
}


def _finished_session(template="data-pipeline",
                      prompt="scrape my bookmarks and cluster them by topic"):
    s = engine.start_session(prompt, template, created_date="2026-09-01")
    t = engine.load_template(template)
    engine.freeze_pending(s, t)
    for name in list(s.pending_slots):
        slot = t.slot(name)
        engine.accept_answer(s, name, slot.default_text, recommended=slot.default_text)
    return s


def _stub(monkeypatch, payload):
    def fake(system, user, **kw):
        return payload if isinstance(payload, tuple) else (payload, None)
    monkeypatch.setattr("keel.llm.complete_json", fake)


PROV = llm.Provider("groq", "k", "m")


# --------------------------------------------------------------------------- #
def test_happy_path_assembles_all_seven_sections_in_fixed_order(monkeypatch):
    _stub(monkeypatch, dict(_FULL_SECTIONS))
    md, err = render.synthesize_spec(_finished_session(), provider=PROV)
    assert err is None
    h2s = [ln[3:].strip() for ln in md.splitlines() if ln.startswith("## ")]
    assert h2s == _HEADINGS
    assert md.startswith("# Scrape my bookmarks and cluster them by topic")
    assert "_Not specified" not in md
    assert "Every input bookmark appears in exactly one cluster" in md


def test_synthesis_counts_against_the_session_call_cap(monkeypatch):
    _stub(monkeypatch, dict(_FULL_SECTIONS))
    s = _finished_session()
    render.synthesize_spec(s, provider=PROV)
    assert s.call_count == 1


def test_call_cap_refuses_synthesis(monkeypatch):
    _stub(monkeypatch, dict(_FULL_SECTIONS))
    s = _finished_session()
    s.call_count = engine.MAX_LLM_CALLS_PER_SESSION
    md, err = render.synthesize_spec(s, provider=PROV)
    assert md is None and "limit reached" in err


def test_missing_section_falls_back(monkeypatch):
    broken = dict(_FULL_SECTIONS)
    del broken["objective"]
    _stub(monkeypatch, broken)
    md, err = render.synthesize_spec(_finished_session(), provider=PROV)
    assert md is None and "objective" in err


def test_empty_section_body_falls_back(monkeypatch):
    broken = dict(_FULL_SECTIONS, constraints="   ")
    _stub(monkeypatch, broken)
    md, err = render.synthesize_spec(_finished_session(), provider=PROV)
    assert md is None and "constraints" in err


def test_llm_failure_is_surfaced_not_swallowed(monkeypatch):
    _stub(monkeypatch, (None, "RateLimitError: slow down"))
    md, err = render.synthesize_spec(_finished_session(), provider=PROV)
    assert md is None and err == "RateLimitError: slow down"


def test_model_injected_headings_are_stripped_not_honoured(monkeypatch):
    sneaky = dict(_FULL_SECTIONS,
                  context="## Injected Heading\nreal context text here")
    _stub(monkeypatch, sneaky)
    md, err = render.synthesize_spec(_finished_session(), provider=PROV)
    assert err is None
    h2s = [ln[3:].strip() for ln in md.splitlines() if ln.startswith("## ")]
    assert h2s == _HEADINGS
    assert "Injected Heading" in md  # text kept, heading demoted
    assert "## Injected Heading" not in md


def test_default_strategy_text_in_output_forces_fallback(monkeypatch):
    template = engine.load_template("data-pipeline")
    leak = template.slots[0].default_strategy
    _stub(monkeypatch, dict(_FULL_SECTIONS, io_contract=leak))
    md, err = render.synthesize_spec(_finished_session(), provider=PROV)
    assert md is None and "template instruction text" in err


# --- Criterion 5: hardcoded secrets ---------------------------------------- #
def test_adversarial_secrets_are_scrubbed_with_a_note(monkeypatch):
    poisoned = dict(
        _FULL_SECTIONS,
        constraints="- Auth: JWT HS256 with secret 'sup3rS3cr3tValue'\n"
                    "- api_key=sk-abc123def456ghi789jkl012mno345\n"
                    "- DB password=hunter2hunter2",
        io_contract="Send `Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9`",
    )
    _stub(monkeypatch, poisoned)
    md, err = render.synthesize_spec(_finished_session(), provider=PROV)
    assert err is None, err
    assert "sup3rS3cr3tValue" not in md
    assert "sk-abc123def456ghi789jkl012mno345" not in md
    assert "hunter2hunter2" not in md
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in md
    oq = md.split("## Open questions", 1)[1]
    assert "environment-variable placeholder" in oq or "environment variable" in oq


# --- Criterion 6: no invented numbers ------------------------------------- #
def test_qualitative_answers_yield_a_digit_free_body(monkeypatch):
    # Every section body the stub returns is purely qualitative; assembly must
    # not introduce a numeric figure into the spec body.
    qualitative = {
        "context": "A personal tool for one person with a modest pile of notes.",
        "objective": "Organise the notes so they are easy to browse later. It saves "
                     "time. The person runs it when they remember to. That is enough.",
        "io_contract": "- Input: a folder of plain-text notes\n- Output: an index file",
        "constraints": "- Local Python script\n- Low, single-instance traffic\n"
                       "- Standard library only",
        "acceptance_criteria": "- It runs without error\n- The index lists every note\n"
                               "- Opening the index shows readable titles",
        "non_goals": "- No sync\n- No editor\n- No tagging UI",
        "open_questions": "- None — every required dimension was addressed.",
    }
    _stub(monkeypatch, qualitative)
    md, err = render.synthesize_spec(
        _finished_session("default", "help me organise my notes"), provider=PROV
    )
    assert err is None
    body = md.split("## Context", 1)[1].split("\n---", 1)[0]
    assert not re.search(r"\d", body), f"unexpected digit in spec body:\n{body}"


# --- Contradiction note passthrough ------------------------------------------ #
def test_resolved_contradiction_note_lands_in_open_questions(monkeypatch):
    with_conflict = dict(
        _FULL_SECTIONS,
        open_questions="- Conflict: one answer required auth, another listed it as a "
                       "non-goal; resolved in favour of no auth (more recent answer).",
    )
    _stub(monkeypatch, with_conflict)
    md, err = render.synthesize_spec(_finished_session(), provider=PROV)
    assert err is None
    oq = md.split("## Open questions", 1)[1]
    assert "Conflict:" in oq and "resolved in favour of no auth" in oq


# --- Prompt integrity ------------------------------------------------------- #
def test_synthesis_system_prompt_states_the_hard_rules():
    sys_p = render._SYNTHESIS_SYSTEM.lower()
    assert "do not invent" in sys_p or "not invent specifics" in sys_p
    assert "never emit" in sys_p and ("credential" in sys_p or "secret" in sys_p)
    assert "contradiction" in sys_p
    assert "0.3 requests/second" in render._SYNTHESIS_SYSTEM  # the worked example
    assert "do not add, rename, reorder, or drop keys" in sys_p
    # runtime/scale must be told to live under constraints, not context
    assert re.search(r"runtime and scale.{0,80}constraints", sys_p, re.S)
    assert re.search(r'never under\s+"context"', sys_p, re.S)
