"""Part B — B1 + B2: the auth_model / seed_data slots and the Build order /
Project structure sections.
"""
from __future__ import annotations

import pathlib

from keel import engine, llm, render, session_io
from keel.models import SlotValue

PROV = llm.Provider("groq", "k", "m")
_FIX = pathlib.Path("evals/fixtures")

_CORE = {
    "context": "A small booking site for a twelve-room hotel.",
    "objective": "Let guests search availability and book a room online. It replaces "
                 "phone bookings. Staff see the bookings that come in. That is the scope.",
    "io_contract": "- Pages: availability search, booking form, confirmation\n"
                   "- Entities: Room(id, number, capacity); Booking(id, room_id, dates)",
    "constraints": "- Server-rendered Python, one process\n- SQLite file\n"
                   "- Invalid dates re-render the form with an inline error",
    "acceptance_criteria": "- A guest can complete a booking end to end\n"
                           "- A double-booking is rejected\n- Reloading shows saved state",
    "non_goals": "- No payments\n- No channel-manager sync\n- No housekeeping module",
    "open_questions": "- None — every dimension was addressed.",
}
_WITH_SECTIONS = dict(
    _CORE,
    build_order=(
        "1. Define the Room and Booking tables and load the twelve rooms. "
        "Verify: the schema migrates and the rooms are present.\n"
        "2. Build the availability search (read path). Verify: a date range returns free rooms.\n"
        "3. Build the booking form and POST /book (write path). Verify: a booking is stored.\n"
        "4. Add the confirmation page and double-booking rejection. Verify: a clash is refused."
    ),
    project_structure=(
        "Suggested starting layout, not a requirement:\n"
        "- app.py — routes and startup\n"
        "- models.py — Room and Booking\n"
        "- templates/ — search, form, confirmation pages\n"
        "- hotel.db — the SQLite file"
    ),
    resolved_conflicts=[],
)


def _stub(monkeypatch, payload):
    def fake(system, user, **kw):
        if "contradiction checker" in system:
            return {"conflicts": []}, None
        return (payload, None)
    monkeypatch.setattr("keel.llm.complete_json", fake)


def _hotel_session():
    session, err = session_io.loads((_FIX / "hotel-website-contradiction.json").read_text("utf-8"))
    assert err is None
    return session


# --------------------------------------------------------------------------- #
# B1: the new slots
# --------------------------------------------------------------------------- #
def test_auth_model_and_seed_data_exist_in_every_template():
    for name in engine.list_templates():
        t = engine.load_template(name)
        am, sd = t.slot("auth_model"), t.slot("seed_data")
        assert am and sd
        assert am.required is False and sd.required is False
        assert am.priority == 10 and sd.priority == 11
        assert am.skip_if and len(am.choices) == 4 and len(sd.choices) == 4
        assert am.why_this_matters and sd.why_this_matters


def test_auth_model_is_suppressed_when_non_goals_exclude_auth():
    s = engine.start_session("a small internal tool", "web-app", created_date="2026-09-01")
    t = engine.load_template("web-app")
    s.slots["non_goals"] = SlotValue(
        value="No user accounts or authentication, no payments.", source="asked"
    )
    assert engine.slot_suppressed(t.slot("auth_model"), s) is True
    assert "auth_model" not in {x.name for x in engine.visible_slots(s, t)}
    # and it is not filled by the context-default pass
    engine.freeze_pending(s, t)
    for name in list(s.pending_slots):
        if name == "non_goals":
            continue
        s.slots[name] = SlotValue(value=t.slot(name).default_text, source="asked")
    engine.fill_unasked_slots(s, t, provider=None)
    assert "auth_model" not in s.slots


def test_auth_model_stays_in_play_when_auth_is_not_excluded():
    s = engine.start_session("a members area for my club", "web-app",
                             created_date="2026-09-01")
    t = engine.load_template("web-app")
    s.slots["non_goals"] = SlotValue(value="No payments, no mobile app.", source="asked")
    assert engine.slot_suppressed(t.slot("auth_model"), s) is False


def test_auth_model_choices_carry_no_security_parameters():
    banned = ("bcrypt", "argon2", "pbkdf2", "sha256", "sha-256", "hs256", "rounds",
              "iteration", "expires in", "ttl", "minutes", "hours")
    for name in engine.list_templates():
        for c in engine.load_template(name).slot("auth_model").choices:
            blob = (c.plain_language + " " + c.tradeoff + " " + c.value).lower()
            assert not any(b in blob for b in banned), (name, c.label)


# --------------------------------------------------------------------------- #
# B2: Build order + Project structure
# --------------------------------------------------------------------------- #
def test_synthesis_prompt_asks_for_build_order_and_project_structure():
    p = render._SYNTHESIS_SYSTEM
    assert '"build_order"' in p and '"project_structure"' in p
    assert "DEPENDENCY" in p
    assert "suggested starting layout" in p.lower()
    assert "no file contents" in p.lower() or "no code" in p.lower()


def test_hotel_fixture_gets_build_order_and_project_structure(monkeypatch):
    _stub(monkeypatch, dict(_WITH_SECTIONS))
    s = _hotel_session()
    md, err = render.synthesize_spec(s, provider=PROV)
    assert err is None

    assert "## Build order" in md and "## Project structure" in md
    bo = md.split("## Build order", 1)[1].split("\n## ", 1)[0]
    stages = [ln for ln in bo.splitlines() if ln.strip()[:2] in ("1.", "2.", "3.", "4.", "5.", "6.")]
    assert len(stages) >= 3
    assert "Verify" in bo

    ps = md.split("## Project structure", 1)[1].split("\n## ", 1)[0]
    assert "app.py" in ps and "not a requirement" in ps.lower()

    order = [ln[3:].strip() for ln in md.splitlines() if ln.startswith("## ")]
    assert order.index("Build order") == order.index("Acceptance criteria") + 1
    assert order.index("Project structure") == order.index("Build order") + 1


def test_build_order_and_project_structure_suppressed_at_quick_depth(monkeypatch):
    _stub(monkeypatch, dict(_WITH_SECTIONS))
    s = _hotel_session()
    s.mode = "technical"
    s.depth = "quick"
    md, err = render.synthesize_spec(s, provider=PROV)
    assert err is None
    assert "## Build order" not in md
    assert "## Project structure" not in md


def test_hotel_open_questions_no_longer_asks_about_auth_params_or_seed_data(monkeypatch):
    _stub(monkeypatch, dict(_WITH_SECTIONS))
    s = _hotel_session()
    md, err = render.synthesize_spec(s, provider=PROV)
    assert err is None
    oq = md.split("## Open questions", 1)[1].split("\n---", 1)[0].lower()
    for phrase in ("password hashing", "hashing algorithm", "session lifetime",
                   "how long sessions", "seed data", "ship with seed"):
        assert phrase not in oq
    # auth_model / seed_data are addressed as slots now
    assert s.slots.get("auth_model") is not None
    assert s.slots.get("seed_data") is not None
