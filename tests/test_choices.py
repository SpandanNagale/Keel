"""Part A — A1 + A2: hand-written choice options and "why this matters" text
live in the template YAML. They must be stable (no LLM call, no per-run
variance) and, in Guided mode, free of unexplained jargon.
"""
from __future__ import annotations

import re

from keel import engine, render
from keel.models import SlotChoice

CHOICE_SLOTS = ("runtime", "scale", "constraints")

# Terms a non-technical user would not know, checked as whole words against the
# option label + its plain-language gloss (the tradeoff line may be more
# technical — it is the "for the curious" detail).
_JARGON = (
    "wsgi", "asgi", "orm", "middleware", "daemon", "stdin", "stdout", "stderr",
    "concurrency", "idempotent", "cdc", "websocket", "backoff", "horizontally",
    "kubernetes", "docker", "reverse proxy", "load balancer",
)


def _has_jargon(text: str) -> list[str]:
    low = text.lower()
    return [t for t in _JARGON if re.search(rf"\b{re.escape(t)}\b", low)]


# --------------------------------------------------------------------------- #
def test_every_slot_has_why_this_matters_in_every_template():
    for name in engine.list_templates():
        for slot in engine.load_template(name).slots:
            assert slot.why_this_matters.strip(), f"{name}.{slot.name}: no why_this_matters"
            # plain language: a couple of sentences, not a one-liner
            assert len(slot.why_this_matters.split()) >= 12, f"{name}.{slot.name}: too terse"


def test_choice_slots_expose_three_or_four_outcome_phrased_options():
    for name in engine.list_templates():
        template = engine.load_template(name)
        for slot_name in CHOICE_SLOTS:
            slot = template.slot(slot_name)
            assert slot is not None
            assert 3 <= len(slot.choices) <= 4, f"{name}.{slot_name}: {len(slot.choices)} choices"
            for c in slot.choices:
                assert c.label.strip() and c.plain_language.strip() and c.tradeoff.strip()
                assert c.as_slot_value(), f"{name}.{slot_name}/{c.label}: empty slot value"


def test_choice_labels_and_glosses_are_free_of_unexplained_jargon():
    offenders = []
    for name in engine.list_templates():
        template = engine.load_template(name)
        for slot_name in CHOICE_SLOTS:
            for c in template.slot(slot_name).choices:
                hits = _has_jargon(c.label + " " + c.plain_language)
                if hits:
                    offenders.append(f"{name}.{slot_name}/{c.label}: {hits}")
    assert not offenders, offenders


def test_slot_choice_value_falls_back_to_plain_language():
    c = SlotChoice(label="x", plain_language="a plain sentence", tradeoff="a cost")
    assert c.as_slot_value() == "a plain sentence"
    c2 = SlotChoice(label="x", plain_language="a plain sentence", tradeoff="a cost",
                    value="the finished spec line")
    assert c2.as_slot_value() == "the finished spec line"


def test_choices_do_not_vary_between_loads():
    a = [c.model_dump() for c in engine.load_template("web-app").slot("runtime").choices]
    engine.load_template.cache_clear()
    b = [c.model_dump() for c in engine.load_template("web-app").slot("runtime").choices]
    assert a == b


# --------------------------------------------------------------------------- #
# A4: architectural shape is back in scope; specific library versions are not.
# --------------------------------------------------------------------------- #
def test_prompts_allow_architectural_shape_but_forbid_version_pins():
    for prompt in (engine._QUESTION_SYSTEM, render._SYNTHESIS_SYSTEM):
        low = prompt.lower()
        assert "architectural shape" in low
        assert "server-rendered" in low and "single-page" in low
        assert "version" in low  # a rule about versions is present
    # the synthesis prompt must tie the version ban to the knowledge cutoff
    assert "knowledge cutoff" in render._SYNTHESIS_SYSTEM.lower()


_VERSION_RE = re.compile(
    r"\b(?:v?\d+\.\d+(?:\.\d+)?)\b(?!\s*(?:requests|rps|%|seconds|users|records))",
)


def _names_a_library_version(text: str) -> list[str]:
    """A crude but useful regression signal: a bare semver-ish token that is not
    obviously a rate or a percentage. Used across phases against fixture output."""
    out = []
    for m in _VERSION_RE.finditer(text):
        # allow dates like 2026-09-02 handled elsewhere; flag x.y / x.y.z
        out.append(m.group(0))
    return out


def test_fixture_output_names_no_library_version():
    """Criterion 5: nothing in a rendered spec pins a library version."""
    from keel.models import SlotValue

    for name in engine.list_templates():
        s = engine.start_session("build a small thing", name, created_date="2026-09-01")
        t = engine.load_template(name)
        engine.freeze_pending(s, t)
        for slot_name in list(s.pending_slots):
            slot = t.slot(slot_name)
            engine.accept_answer(s, slot_name, slot.default_text, recommended=slot.default_text)
        engine.fill_unasked_slots(s, t, provider=None)
        md = render.render_markdown(s)
        assert not _names_a_library_version(md), f"{name}: {_names_a_library_version(md)}"
