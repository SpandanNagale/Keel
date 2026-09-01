from __future__ import annotations

import pytest

from keel import engine, llm
from keel.models import Section

_PROVIDER = llm.Provider("groq", "test-key", "openai/gpt-oss-120b")

REQUIRED = {"io_contract", "scale", "runtime", "constraints", "done", "non_goals"}
SECTIONS = set(Section.__args__)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", engine.list_templates())
def test_every_template_defines_the_six_required_slots(name):
    template = engine.load_template(name)
    required = {s.name for s in template.slots if s.required}
    assert REQUIRED <= required
    for s in template.slots:
        assert s.section in SECTIONS
        assert s.default_text and s.default_strategy and s.question_hint
        assert s.default_text != s.default_strategy


def test_list_templates_is_exactly_the_shipped_set():
    assert engine.list_templates() == ["default", "cli", "data-pipeline", "web-api", "web-app"]


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("scrape my bookmarks and cluster them by topic", "data-pipeline"),
        ("build a REST API for a todo list", "web-api"),
        ("a CLI to rename photos by their EXIF date", "cli"),
        ("help me get my thoughts in order for the week", "default"),
        # Change 4 regressions:
        ("hotel website", "web-app"),
        ("REST API for my notes app", "web-api"),
        ("scrape my bookmarks", "data-pipeline"),
        ("rename my photos by date", "cli"),
        ("a booking website for a small hotel", "web-app"),
        ("landing page for my newsletter", "web-app"),
    ],
)
def test_select_template_keyword_heuristic(prompt, expected):
    assert engine.select_template(prompt) == expected


def test_website_never_routes_to_the_ui_forbidding_api_template():
    # The motivating bug: "hotel website" -> web-api, whose non-goals forbid a UI.
    for prompt in ("hotel website", "restaurant website with online booking",
                   "personal portfolio site", "web app to track my runs"):
        assert engine.select_template(prompt) != "web-api"


def test_slugify():
    assert engine.slugify("Scrape My Bookmarks!") == "scrape-my-bookmarks"
    assert engine.slugify("   ---   ") == "project"
    assert len(engine.slugify("x" * 200)) <= 40


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #
def test_freeze_pending_orders_required_slots_by_priority(make_session):
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    assert session.pending_slots == [s.name for s in template.required_slots()]
    assert session.current_index == 0
    assert session.finished is False


def test_accept_answer_classifies_source_and_advances(make_session):
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    first = session.pending_slots[0]

    engine.accept_answer(session, first, "  ", recommended="the default line")
    assert session.slots[first].value == "the default line"
    assert session.slots[first].source == "defaulted"

    second = session.pending_slots[1]
    engine.accept_answer(session, second, "something the user typed", recommended="rec")
    assert session.slots[second].source == "asked"
    assert session.current_index == 2
    assert session.questions_asked == 2


def test_skip_slot_records_skip_and_advances(make_session):
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    name = session.pending_slots[0]
    engine.skip_slot(session, name)
    assert session.slots[name].source == "skipped"
    assert session.slots[name].value == ""
    assert session.current_index == 1


def test_fill_remaining_defaults_backfills_and_finishes(make_session):
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    engine.fill_remaining_defaults(session, template)
    assert session.finished is True
    for s in template.required_slots():
        assert session.slots[s.name].source == "defaulted"
        assert session.slots[s.name].value == s.default_text


# --------------------------------------------------------------------------- #
# LLM call site: extraction
# --------------------------------------------------------------------------- #
def test_extract_prefilled_fills_slots_and_freeze_pending_skips_them(make_session, stub_llm):
    stub_llm(({"io_contract": "reads bookmark HTML, writes topic clusters"}, None))
    session = make_session("scrape bookmarks and cluster by topic", "data-pipeline")
    template = engine.load_template("data-pipeline")

    error = engine.extract_prefilled(session, template, provider=_PROVIDER)
    assert error is None
    assert session.degraded is False
    assert session.slots["io_contract"].source == "extracted"
    assert session.call_count == 1

    engine.freeze_pending(session, template)
    assert "io_contract" not in session.pending_slots
    assert len(session.pending_slots) == 5


def test_extract_prefilled_surfaces_failure_and_sets_degraded(make_session, stub_llm):
    stub_llm((None, "APIConnectionError: connection refused"))
    session = make_session()
    template = engine.load_template("default")

    error = engine.extract_prefilled(session, template, provider=_PROVIDER)
    assert error == "APIConnectionError: connection refused"
    assert session.degraded is True
    assert session.slots == {}


def test_extract_prefilled_drops_junk_and_unknown_slot_names(make_session, stub_llm):
    stub_llm((
        {"scale": "unknown", "not_a_slot": "x", "runtime": "runs on an hourly cron"},
        None,
    ))
    session = make_session()
    template = engine.load_template("default")
    engine.extract_prefilled(session, template, provider=_PROVIDER)
    assert "scale" not in session.slots
    assert "not_a_slot" not in session.slots
    assert session.slots["runtime"].value == "runs on an hourly cron"


# --------------------------------------------------------------------------- #
# LLM call site: question generation
# --------------------------------------------------------------------------- #
def test_next_question_returns_model_output_on_success(make_session, stub_llm):
    stub_llm(({"question": "How many bookmarks?", "recommended": "About 2,000."}, None))
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)

    q, rec, err = engine.next_question(session, template, provider=_PROVIDER)
    assert (q, rec, err) == ("How many bookmarks?", "About 2,000.", None)
    assert session.call_count == 1
    assert session.degraded is False


def test_next_question_falls_back_to_static_text_on_failure(make_session, stub_llm):
    stub_llm((None, "AuthenticationError: invalid x-api-key"))
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    slot = engine.current_slot(session, template)

    q, rec, err = engine.next_question(session, template, provider=_PROVIDER)
    assert q == slot.question_hint
    assert rec == slot.default_text
    assert rec != slot.default_strategy
    assert "AuthenticationError" in err
    assert session.degraded is True


def test_next_question_treats_missing_fields_as_a_parse_failure(make_session, stub_llm):
    stub_llm(({"question": "only a question, no recommendation"}, None))
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    slot = engine.current_slot(session, template)

    q, rec, err = engine.next_question(session, template, provider=_PROVIDER)
    assert q == slot.question_hint and rec == slot.default_text
    assert "missing question/recommended" in err
    assert session.degraded is True


def test_session_call_cap_refuses_the_eleventh_call(make_session, stub_llm):
    stub_llm(({"question": "q", "recommended": "r"}, None))
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)

    for _ in range(engine.MAX_LLM_CALLS_PER_SESSION):
        engine.next_question(session, template, provider=_PROVIDER)
    assert session.call_count == engine.MAX_LLM_CALLS_PER_SESSION

    q, rec, err = engine.next_question(session, template, provider=_PROVIDER)
    assert "limit reached" in err
    assert session.call_count == engine.MAX_LLM_CALLS_PER_SESSION  # not incremented
    assert session.degraded is True


# --------------------------------------------------------------------------- #
# End to end: accept every default
# --------------------------------------------------------------------------- #
def test_accepting_every_default_finishes_in_under_eight_questions(make_session, stub_llm):
    stub_llm(({"question": "q?", "recommended": "a concrete recommended answer"}, None))
    session = make_session("dedupe my contacts export", "default")
    template = engine.load_template("default")
    engine.extract_prefilled(session, template, provider=_PROVIDER)
    engine.freeze_pending(session, template)

    guard = 0
    while not session.finished:
        guard += 1
        assert guard < 20
        slot = engine.current_slot(session, template)
        q, rec, err = engine.next_question(session, template, provider=_PROVIDER)
        engine.accept_answer(session, slot.name, rec, recommended=rec)

    assert session.questions_asked < 8
    for s in template.required_slots():
        assert session.slots[s.name].value
