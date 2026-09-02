from __future__ import annotations

import pytest

from keel import engine, llm
from keel.models import Section, SlotValue

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
def test_freeze_pending_selects_slots_by_depth(make_session):
    template = engine.load_template("default")
    required = [s.name for s in template.required_slots()]

    quick = make_session()
    quick.depth = "quick"
    engine.freeze_pending(quick, template)
    assert quick.pending_slots == required  # only the six core dimensions

    standard = make_session()  # default depth
    engine.freeze_pending(standard, template)
    assert standard.pending_slots == required + ["data_model", "interfaces"]
    assert standard.current_index == 0 and standard.finished is False

    thorough = make_session()
    thorough.depth = "thorough"
    engine.freeze_pending(thorough, template)
    assert len(thorough.pending_slots) == engine.MAX_ASKED_QUESTIONS  # 9 slots, capped at 8
    assert "error_handling" not in thorough.pending_slots  # lowest priority, trimmed


def test_accept_answer_classifies_source_and_advances(make_session):
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    first = session.pending_slots[0]

    engine.accept_answer(session, first, "  ", recommended="the default line")
    assert session.slots[first].value == "the default line"
    assert session.slots[first].source == "llm_default"  # accepted an LLM recommendation

    second = session.pending_slots[1]
    engine.accept_answer(session, second, "something the user typed", recommended="rec")
    assert session.slots[second].source == "asked"
    assert session.current_index == 2
    assert session.questions_asked == 2

    third = session.pending_slots[2]
    engine.accept_answer(session, third, "", recommended="the static hint",
                         recommended_source="template_default")
    assert session.slots[third].source == "template_default"  # question call had failed


def test_skip_slot_records_skip_and_advances(make_session):
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    name = session.pending_slots[0]
    engine.skip_slot(session, name)
    assert session.slots[name].source == "skipped"
    assert session.slots[name].value == ""
    assert session.current_index == 1


def test_fill_remaining_defaults_backfills_every_slot_and_finishes(make_session):
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    engine.fill_remaining_defaults(session, template)
    assert session.finished is True
    # auth_model is suppressed once non_goals (here its default) already rules
    # authentication out — so check the visible slots, not the raw template.
    for s in engine.visible_slots(session, template):
        assert session.slots[s.name].source == "template_default"
        assert session.slots[s.name].value == s.default_text
    assert "auth_model" not in session.slots  # suppressed by the default non_goals
    assert session.degraded is True  # a template-only spec is a real fallback


def test_fill_unasked_slots_uses_context_then_falls_back(make_session, stub_llm):
    stub_llm(({"data_model": "one Contact record: name, email, phone"}, None))
    session = make_session("dedupe my contacts export", "default")
    session.depth = "quick"  # all three optional slots are unasked
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    for name in list(session.pending_slots):
        engine.accept_answer(session, name, "x", recommended="x")

    err = engine.fill_unasked_slots(session, template, provider=_PROVIDER)
    assert err is None
    assert session.finished is True
    # model-supplied value used where given...
    assert session.slots["data_model"].value == "one Contact record: name, email, phone"
    assert session.slots["data_model"].source == "llm_default"
    # ...static default_text (marked template_default) where the model said nothing
    assert session.slots["error_handling"].value == template.slot("error_handling").default_text
    assert session.slots["error_handling"].source == "template_default"
    assert session.slots["interfaces"].value == template.slot("interfaces").default_text


def test_fill_unasked_slots_surfaces_failure_but_still_fills(make_session, stub_llm):
    stub_llm((None, "APIConnectionError: boom"))
    session = make_session("dedupe my contacts export", "default")
    session.depth = "quick"
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    for name in list(session.pending_slots):
        engine.accept_answer(session, name, "x", recommended="x")

    err = engine.fill_unasked_slots(session, template, provider=_PROVIDER)
    assert err == "APIConnectionError: boom"
    assert session.degraded is True
    assert session.slots["error_handling"].value == template.slot("error_handling").default_text


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
    # 6 required - 1 extracted + 2 optional (standard depth) = 7
    assert len(session.pending_slots) == 7


def test_extract_prefilled_surfaces_failure_without_degrading(make_session, stub_llm):
    # A failed extraction is surfaced, but does not by itself degrade the session:
    # every slot it would have filled is still asked from a blank page.
    stub_llm((None, "APIConnectionError: connection refused"))
    session = make_session()
    template = engine.load_template("default")

    error = engine.extract_prefilled(session, template, provider=_PROVIDER)
    assert error == "APIConnectionError: connection refused"
    assert session.degraded is False
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

    p = engine.next_question(session, template, provider=_PROVIDER)
    q, rec, err = p.question, p.recommended, p.error
    assert (q, rec, err) == ("How many bookmarks?", "About 2,000.", None)
    assert session.call_count == 1
    assert session.degraded is False


def test_next_question_falls_back_to_static_text_on_failure(make_session, stub_llm):
    stub_llm((None, "AuthenticationError: invalid x-api-key"))
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    slot = engine.current_slot(session, template)

    p = engine.next_question(session, template, provider=_PROVIDER)
    q, rec, err = p.question, p.recommended, p.error
    assert q == slot.question_hint
    assert rec == slot.default_text
    assert rec != slot.default_strategy
    assert "AuthenticationError" in err
    # next_question itself does not degrade — only accepting the static fallback does
    assert session.degraded is False
    engine.accept_answer(session, slot.name, rec, recommended=rec,
                         recommended_source="template_default")
    assert session.degraded is True


def test_next_question_treats_missing_fields_as_a_parse_failure(make_session, stub_llm):
    stub_llm(({"question": "only a question, no recommendation"}, None))
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)
    slot = engine.current_slot(session, template)

    p = engine.next_question(session, template, provider=_PROVIDER)
    q, rec, err = p.question, p.recommended, p.error
    assert q == slot.question_hint and rec == slot.default_text
    assert "missing question/recommended" in err
    assert session.degraded is False  # not until the fallback is accepted


def test_session_call_cap_refuses_the_eleventh_call(make_session, stub_llm):
    stub_llm(({"question": "q", "recommended": "r"}, None))
    session = make_session()
    template = engine.load_template("default")
    engine.freeze_pending(session, template)

    for _ in range(engine.MAX_LLM_CALLS_PER_SESSION):
        engine.next_question(session, template, provider=_PROVIDER)
    assert session.call_count == engine.MAX_LLM_CALLS_PER_SESSION

    err = engine.next_question(session, template, provider=_PROVIDER).error
    assert "limit reached" in err
    assert session.call_count == engine.MAX_LLM_CALLS_PER_SESSION  # not incremented
    assert session.degraded is False  # the fallback text has not been accepted


# --------------------------------------------------------------------------- #
# Phase 3: review-step edits and regeneration budget
# --------------------------------------------------------------------------- #
def test_apply_answer_edits_marks_changes_and_empties_as_skips(make_session):
    session = make_session()
    template = engine.load_template("default")
    session.slots = {
        "scale": SlotValue(value="small", source="llm_default"),
        "runtime": SlotValue(value="a local script", source="asked"),
        "done": SlotValue(value="it runs", source="llm_default"),
    }
    changed = engine.apply_answer_edits(
        session,
        {"scale": "about 5000 rows", "runtime": "a local script", "done": "  ",
         "not_a_slot": "ignored"},
        template,
    )
    assert changed == 2
    assert session.slots["scale"].value == "about 5000 rows"
    assert session.slots["scale"].source == "asked"          # edited -> stands behind it
    assert session.slots["runtime"].source == "asked"        # unchanged -> untouched
    assert session.slots["done"].source == "skipped"         # emptied -> skip
    assert session.slots["done"].value == ""


def test_regeneration_budget_respects_both_the_count_and_the_call_cap(make_session):
    session = make_session()

    session.regen_count = engine.MAX_REGENERATIONS
    ok, why = engine.can_regenerate(session)
    assert ok is False and "regeneration limit" in why

    session.regen_count = 0
    session.call_count = engine.MAX_LLM_CALLS_PER_SESSION - 1  # only 1 call left, need 2
    ok, why = engine.can_regenerate(session)
    assert ok is False and "budget" in why
    assert engine.regenerations_left(session) == 0

    session.call_count = engine.MAX_LLM_CALLS_PER_SESSION - 4
    assert engine.regenerations_left(session) == 2
    assert engine.can_regenerate(session)[0] is True


# --------------------------------------------------------------------------- #
# End to end: accept every default
# --------------------------------------------------------------------------- #
def test_accepting_every_default_finishes_in_at_most_eight_questions(make_session, stub_llm):
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
        rec = engine.next_question(session, template, provider=_PROVIDER).recommended
        engine.accept_answer(session, slot.name, rec, recommended=rec)

    assert session.questions_asked <= engine.MAX_ASKED_QUESTIONS
    for s in template.required_slots():
        assert session.slots[s.name].value
