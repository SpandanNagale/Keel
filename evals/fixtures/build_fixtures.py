"""Regenerate the frozen regression fixtures.

Run from the repo root:  python evals/fixtures/build_fixtures.py

These two sessions carry answers that contradict each other. They are permanent
regression cases: check_conflicts must keep catching the contradiction after any
synthesis- or conflict-prompt edit. run_evals.py loads them; this script only
rewrites them if the slot taxonomy changes.
"""
from __future__ import annotations

import pathlib

from keel import engine, session_io
from keel.models import SlotValue

_HERE = pathlib.Path(__file__).parent


def _session(prompt, template, depth, answers):
    s = engine.start_session(prompt, template, created_date="2026-09-01", depth=depth)
    t = engine.load_template(template)
    engine.freeze_pending(s, t)
    for name, value in answers.items():
        s.slots[name] = SlotValue(value=value, source="asked")
    engine.fill_unasked_slots(s, t, provider=None)  # static-fill any gap
    s.finished = True
    return s


HOTEL = _session(
    "a booking website for a small hotel", "web-app", "thorough",
    {
        "io_contract": "Guests open pages to search room availability by date and submit a "
                       "booking form; the server renders a confirmation page.",
        "runtime": "A single web server process on one instance.",
        "scale": "Twelve rooms; a handful of bookings a day.",
        "constraints": "Server-rendered Python (Flask) with a SQLite file. No build step.",
        "non_goals": "No web UI or browser-facing pages of any kind; no frontend. Payments "
                     "are also out of scope.",
        "done": "A guest can complete a booking end to end in a web browser and see a "
                "confirmation page.",
        "data_model": "Room(id, number, capacity); Booking(id, room_id, guest_name, "
                      "check_in, check_out, created_at).",
        "interfaces": "GET /  (availability search), POST /book  (create a booking), "
                      "GET /booking/<id>  (confirmation).",
        "error_handling": "An invalid date range re-renders the form with an inline error; "
                          "a double-booking is rejected with a message.",
    },
)

HEALTH = _session(
    "AI chat bot for health monitoring", "default", "thorough",
    {
        "io_contract": "A POST endpoint takes a free-text health message and returns an "
                       "AI-generated interpretation with suggested next steps.",
        "runtime": "A local long-running HTTP server.",
        "scale": "One user, a few messages a day.",
        "constraints": "Python standard library only. Runs fully offline: no external or "
                       "paid APIs, and no model downloads or local model weights.",
        "non_goals": "No authentication, no database, no web UI.",
        "done": "Sending a message returns a relevant AI-generated interpretation of the "
                "described symptoms.",
        "data_model": "Message(id, text, received_at); Reply(id, message_id, text, sent_at).",
        "interfaces": "POST /chat  {text} -> {interpretation, steps}.",
        "error_handling": "An empty or oversized body returns 400 with a JSON error; an "
                          "internal failure returns 500 without a stack trace.",
    },
)


def main() -> None:
    for name, sess in (("hotel-website-contradiction", HOTEL),
                       ("health-monitoring-offline-ai", HEALTH)):
        path = _HERE / f"{name}.json"
        path.write_text(session_io.dumps(sess), encoding="utf-8")
        print(f"wrote {path.relative_to(_HERE.parent.parent)}")


if __name__ == "__main__":
    main()
