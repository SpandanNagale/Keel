"""Acceptance criterion 2 — the specific regression the previous build failed.

Two materially different opening prompts must produce materially different
questions and materially different specs. Here the LLM is faked with a responder
that echoes the idea it was given, so any cross-contamination or hard-coded
question path shows up as identical output.
"""
from __future__ import annotations

import re

from keel import engine, llm
from keel.render import render_markdown

_PROVIDER = llm.Provider("groq", "test-key", "openai/gpt-oss-120b")


def _idea_echo_responder(system, user, **kw):
    idea = re.search(r'Project idea: "([^"]*)"', user)
    target = re.search(r"Target slot: .*\((\w+)\)", user)
    tag = idea.group(1) if idea else "?"
    slot = target.group(1) if target else "?"
    return {
        "question": f"For '{tag}', what is the {slot}?",
        "recommended": f"{slot} answer tuned to: {tag}",
    }, None


def _run(prompt: str, monkeypatch) -> tuple[list[str], str]:
    monkeypatch.setattr("keel.llm.complete_json", _idea_echo_responder)
    template_name = engine.select_template(prompt)
    template = engine.load_template(template_name)
    session = engine.start_session(prompt, template_name, created_date="2026-09-01")
    # No extraction call here: exercise the question path directly.
    engine.freeze_pending(session, template)

    questions: list[str] = []
    guard = 0
    while not session.finished:
        guard += 1
        assert guard < 20
        slot = engine.current_slot(session, template)
        q, rec, err = engine.next_question(session, template, provider=_PROVIDER)
        assert err is None
        questions.append(q)
        engine.accept_answer(session, slot.name, rec, recommended=rec)
    return questions, render_markdown(session)


def test_two_different_prompts_diverge_in_questions_and_spec(monkeypatch):
    q1, spec1 = _run("scrape my bookmarks and cluster them by topic", monkeypatch)
    q2, spec2 = _run("build a REST API for a todo list", monkeypatch)

    assert engine.select_template("scrape my bookmarks and cluster them by topic") != \
        engine.select_template("build a REST API for a todo list")
    assert q1 != q2
    assert set(q1).isdisjoint(set(q2))
    assert spec1 != spec2
    assert "bookmarks" in spec1 and "bookmarks" not in spec2
    assert "todo list" in spec2 and "todo list" not in spec1


def test_same_prompt_is_deterministic_in_structure(monkeypatch):
    q1, spec1 = _run("dedupe my contacts export", monkeypatch)
    q2, spec2 = _run("dedupe my contacts export", monkeypatch)
    assert q1 == q2
    assert spec1 == spec2
