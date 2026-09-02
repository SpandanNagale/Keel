"""Part B — B3 + B4: feasibility conflicts and the numeric-honesty scrub."""
from __future__ import annotations

import pathlib
import re

from keel import engine, llm, render, session_io

PROV = llm.Provider("groq", "k", "m")
_FIX = pathlib.Path("evals/fixtures")

_CORE = {
    "context": "An internal dashboard for company staff.",
    "objective": "Give staff one place to read reports and add notes. It replaces a "
                 "scattered set of spreadsheets. Everyone sees the same data. That is it.",
    "io_contract": "- Pages: report list, report detail, add-note form\n- Entities: Report, Note",
    "constraints": "- Server-rendered Python, one process\n- A single SQLite file\n"
                   "- A bad filter shows an empty state",
    "acceptance_criteria": "- A staff member can filter a report\n- A note is saved and shown\n"
                           "- Reloading shows the saved note",
    "non_goals": "- No mobile app\n- No public access\n- No spreadsheet export",
    "open_questions": "- None — every required dimension was addressed.",
}


def _load(stem):
    session, err = session_io.loads((_FIX / f"{stem}.json").read_text("utf-8"))
    assert err is None
    return session


def _route(monkeypatch, *, conflicts, sections):
    def fake(system, user, **kw):
        if "contradiction checker" in system:
            return {"conflicts": conflicts}, None
        return sections, None
    monkeypatch.setattr("keel.llm.complete_json", fake)


# --------------------------------------------------------------------------- #
# B3: feasibility
# --------------------------------------------------------------------------- #
def test_conflict_prompt_covers_feasibility_and_the_kind_field():
    p = render._CONFLICT_SYSTEM
    low = p.lower()
    assert '"feasibility"' in p and '"logical"' in p
    assert "development server cannot" in low or "development server" in low
    assert "combination" in low


def test_check_conflicts_normalises_the_kind(monkeypatch):
    def fake(system, user, **kw):
        return {"conflicts": [
            {"slots": ["runtime", "scale"], "kind": "feasibility",
             "conflict": "A single-process dev server cannot serve everyone at once.",
             "suggested_resolution": "Use a production server, or lower the concurrency claim."},
            {"slots": ["a", "b"], "conflict": "no kind given -> logical"},
            {"slots": ["c"], "kind": "nonsense", "conflict": "bad kind -> logical"},
        ]}, None
    monkeypatch.setattr("keel.llm.complete_json", fake)

    s = _load("feasibility-devserver-concurrency")
    conflicts, err = render.check_conflicts(s, provider=PROV)
    assert err is None
    assert [c["kind"] for c in conflicts] == ["feasibility", "logical", "logical"]


def test_feasibility_fixture_surfaces_a_feasibility_conflict(monkeypatch):
    """Criterion 8: dev server + high concurrency -> a feasibility conflict that
    renders distinctly in Open questions."""
    conflict = {
        "slots": ["runtime", "scale"],
        "kind": "feasibility",
        "conflict": "The framework's single-process development server cannot serve "
                    "the whole company using the dashboard at the same time.",
        "suggested_resolution": "Run it under a production application server, or state "
                                "a smaller concurrent audience.",
    }
    _route(monkeypatch, conflicts=[conflict], sections=dict(_CORE, resolved_conflicts=[]))
    s = _load("feasibility-devserver-concurrency")
    conflicts, cerr = render.check_conflicts(s, provider=PROV)
    assert conflicts and conflicts[0]["kind"] == "feasibility"

    md, err = render.synthesize_spec(s, provider=PROV, conflicts=conflicts,
                                     conflict_error=cerr)
    assert err is None
    oq = md.split("## Open questions", 1)[1]
    assert "**Feasibility (" in oq
    assert "Likely fix:" in oq
    assert "**Conflict (" not in oq  # this one is not framed as a self-contradiction


# --------------------------------------------------------------------------- #
# B4: numeric honesty
# --------------------------------------------------------------------------- #
def test_invented_capacity_and_volume_figures_are_scrubbed_and_noted(monkeypatch):
    sections = dict(
        _CORE,
        constraints="- Server-rendered Python\n- Must support up to 150 concurrent users\n"
                    "- Runs on the dev server",
        acceptance_criteria="- The catalogue lists 8,000 hotels and 20,000 rooms\n"
                            "- A booking is saved\n- 5000 bookings load in under 2 seconds",
        resolved_conflicts=[],
    )
    _route(monkeypatch, conflicts=[], sections=sections)
    s = _load("qualitative-only")
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=[], conflict_error=None)
    assert err is None

    # the fabricated figures are gone from the section bodies...
    sections = md.split("## Open questions", 1)[0]
    for invented in ("150", "8,000", "8000", "20,000", "20000", "5000", "5,000"):
        assert invented not in sections, invented
    # ...and each is recorded once as an unverified-figure note
    oq = md.split("## Open questions", 1)[1]
    assert oq.count("Unverified figure") >= 3
    assert '"150"' in oq


def test_qualitative_only_fixture_yields_no_invented_figures(monkeypatch):
    """Criterion 9: an entirely-qualitative session produces no capacity /
    throughput / volume numbers."""
    _route(monkeypatch, conflicts=[], sections=dict(_CORE, resolved_conflicts=[]))
    s = _load("qualitative-only")
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=[], conflict_error=None)
    assert err is None
    body = md.split("## Context", 1)[1].split("\n---", 1)[0]
    # the stub bodies are digit-free; assembly must not introduce a figure
    assert not re.search(r"\d", body), body


def test_supported_figures_and_their_derivations_are_left_alone(monkeypatch):
    # 12 rooms is in an answer; 24 (12*2) and 288 (12*24) are derivations.
    sections = dict(
        _CORE,
        io_contract="The hotel has 12 rooms, so at most 24 guests per night and "
                    "roughly 288 room-nights a month.",
        resolved_conflicts=[],
    )
    _route(monkeypatch, conflicts=[], sections=sections)
    s = _load("qualitative-only")
    s.slots["scale"].value = "A small hotel with 12 rooms."
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=[], conflict_error=None)
    assert err is None
    body = md.split("## Input / Output contract", 1)[1].split("\n## ", 1)[0]
    assert "12 rooms" in body and "24 guests" in body and "288 room-nights" in body
