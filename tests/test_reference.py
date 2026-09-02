"""Phase 1: the reference -> evidence -> candidates -> slots pipeline.
Firecrawl and the LLM are always stubbed."""
from __future__ import annotations

import pytest

from keel import engine, firecrawl, llm, reference, render
from keel.models import Evidence, ReferenceState, SlotCandidate

PROV = llm.Provider("groq", "k", "m")


# --------------------------------------------------------------------------- #
# gather_material
# --------------------------------------------------------------------------- #
def _fake_scrape(pages: dict):
    def scrape(url, *, api_key, timeout=45):
        if url in pages:
            return {"markdown": pages[url], "metadata": {}, "url": url}, None
        return None, "not found"
    return scrape


def test_gather_material_scrapes_a_single_rich_page(monkeypatch):
    monkeypatch.setattr(firecrawl, "scrape",
                        _fake_scrape({"https://ex.com": "detailed content " * 200}))
    monkeypatch.setattr(firecrawl, "map_site",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not map")))
    mat, err = reference.gather_material("https://ex.com", api_key="fc-k")
    assert err is None
    assert mat["fetches"] == 1 and mat["urls"] == ["https://ex.com"]
    assert "detailed content" in mat["material"]


def test_gather_material_maps_and_scrapes_subpages_for_a_thin_landing_page(monkeypatch):
    pages = {
        "https://ex.com": "tiny landing page",
        "https://ex.com/features": "feature detail " * 80,
        "https://ex.com/pricing": "pricing detail " * 80,
        "https://ex.com/blog/post": "a blog post " * 80,
    }
    monkeypatch.setattr(firecrawl, "scrape", _fake_scrape(pages))
    monkeypatch.setattr(firecrawl, "map_site",
                        lambda url, *, api_key, timeout=45: (list(pages)[1:], None))
    mat, err = reference.gather_material("https://ex.com", api_key="fc-k", fetch_budget=3)
    assert err is None
    assert mat["fetches"] == 3                       # home + 2 structural subpages
    assert mat["fetches"] <= reference.MAX_REFERENCE_FETCHES
    assert "https://ex.com/features" in mat["urls"]
    assert "https://ex.com/blog/post" not in mat["urls"]   # not structural


def test_gather_material_rejects_a_bad_url_before_any_fetch(monkeypatch):
    monkeypatch.setattr(firecrawl, "scrape",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no fetch")))
    mat, err = reference.gather_material("http://localhost/x", api_key="fc-k")
    assert mat is None and "local and internal" in err


def test_gather_material_surfaces_a_scrape_failure(monkeypatch):
    monkeypatch.setattr(firecrawl, "scrape", lambda *a, **k: (None, "Firecrawl HTTP 500"))
    mat, err = reference.gather_material("https://ex.com", api_key="fc-k")
    assert mat is None and err == "Firecrawl HTTP 500"


# --------------------------------------------------------------------------- #
# extract_evidence
# --------------------------------------------------------------------------- #
def test_extract_evidence_coerces_loose_shapes(monkeypatch):
    monkeypatch.setattr("keel.llm.complete_json", lambda *a, **k: ({
        "product": "  Acme  ",
        "core_entities": ["Widget", "  ", "User"],
        "primary_flows": "browse widgets",                       # a string, not a list
        "surfaces": None,
        "features_likely_out_of_scope_for_a_clone": ["billing", "SSO"],
    }, None))
    s = engine.start_session("x", "web-app", created_date="2026-09-01")
    ev, err = reference.extract_evidence("material", session=s, provider=PROV)
    assert err is None
    assert ev.product == "Acme"
    assert ev.core_entities == ["Widget", "User"]
    assert ev.primary_flows == ["browse widgets"]
    assert ev.surfaces == []
    assert ev.features_likely_out_of_scope == ["billing", "SSO"]
    assert s.call_count == 1                                     # counted against the cap


def test_extract_evidence_surfaces_an_llm_failure(monkeypatch):
    monkeypatch.setattr("keel.llm.complete_json", lambda *a, **k: (None, "RateLimitError"))
    s = engine.start_session("x", "web-app", created_date="2026-09-01")
    ev, err = reference.extract_evidence("m", session=s, provider=PROV)
    assert ev is None and err == "RateLimitError"


# --------------------------------------------------------------------------- #
# evidence_to_candidates / apply_candidates
# --------------------------------------------------------------------------- #
def test_evidence_maps_to_the_right_slots_and_skips_empties():
    ev = Evidence(
        product="Acme",
        core_entities=["Board", "Card"],
        surfaces=["board view"],
        primary_flows=["drag a card"],
        notable_features=[],                       # -> no `done` candidate
        features_likely_out_of_scope=["automation rules"],
    )
    cands = reference.evidence_to_candidates(ev, engine.load_template("web-app"))
    by_slot = {c.slot: c for c in cands}
    assert set(by_slot) == {"data_model", "interfaces", "non_goals"}
    assert "Board; Card" in by_slot["data_model"].evidence
    assert "board view" in by_slot["interfaces"].value and "drag a card" in by_slot["interfaces"].value
    assert "automation rules" in by_slot["non_goals"].value


def test_apply_candidates_writes_reference_source_and_respects_decisions():
    s = engine.start_session("x", "web-app", created_date="2026-09-01")
    t = engine.load_template("web-app")
    cands = [
        SlotCandidate(slot="data_model", value="Board(id, name); Card(id, board_id)",
                      decision="keep"),
        SlotCandidate(slot="non_goals", value="edited by the user", decision="keep"),
        SlotCandidate(slot="interfaces", value="dropped", decision="drop"),
        SlotCandidate(slot="not_a_slot", value="x", decision="keep"),
        SlotCandidate(slot="done", value="   ", decision="keep"),
    ]
    n = reference.apply_candidates(s, cands, t)
    assert n == 2
    assert s.slots["data_model"].source == "reference"
    assert s.slots["non_goals"].value == "edited by the user"
    assert "interfaces" not in s.slots and "not_a_slot" not in s.slots and "done" not in s.slots


# --------------------------------------------------------------------------- #
# provenance: the source URL travels with the document
# --------------------------------------------------------------------------- #
def _finished_with_reference(monkeypatch):
    s = engine.start_session("something like a kanban board", "web-app",
                             created_date="2026-09-01")
    t = engine.load_template("web-app")
    s.reference = ReferenceState(mode="url", query="https://trello.com",
                                 source_urls=["https://trello.com", "https://trello.com/pricing"],
                                 fetch_count=2, confirmed=True)
    engine.freeze_pending(s, t)
    for name in list(s.pending_slots):
        engine.accept_answer(s, name, t.slot(name).default_text,
                             recommended=t.slot(name).default_text)
    engine.fill_unasked_slots(s, t, provider=None)
    return s


def test_reference_provenance_line_lands_in_open_questions_deterministic(monkeypatch):
    s = _finished_with_reference(monkeypatch)
    md = render.render_markdown(s)
    oq = md.split("## Open questions", 1)[1]
    assert "Reference used:" in oq
    assert "https://trello.com/pricing" in oq


def test_reference_provenance_line_lands_in_open_questions_synthesized(monkeypatch):
    s = _finished_with_reference(monkeypatch)
    sections = {
        "context": "A personal kanban tool.", "objective": "Build a board for one person. "
        "It tracks tasks in columns. The person drags cards. That is the scope.",
        "io_contract": "- Board view\n- Card CRUD endpoints",
        "constraints": "- Local web app\n- Low traffic\n- Python",
        "acceptance_criteria": "- A card can be created\n- A card can move columns\n"
        "- The board persists across restarts",
        "non_goals": "- No multi-user\n- No automation", "open_questions": "- None.",
    }
    monkeypatch.setattr("keel.llm.complete_json",
                        lambda system, user, **k: ({"conflicts": []}, None)
                        if "contradiction checker" in system else (sections, None))
    conflicts, cerr = render.check_conflicts(s, provider=PROV)
    md, err = render.synthesize_spec(s, provider=PROV, conflicts=conflicts, conflict_error=cerr)
    assert err is None
    oq = md.split("## Open questions", 1)[1].split("\n---", 1)[0]
    assert "Reference used:" in oq and "trello.com" in oq
