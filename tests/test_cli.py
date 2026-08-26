import json
from pathlib import Path

import pytest

from keel import llm
from keel.cli import amend, run_session
from keel.models import SessionState


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: None)
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: None)


def test_pressing_enter_everywhere_produces_valid_complete_output(tmp_path: Path, monkeypatch):
    # Simulate the user pressing Enter (empty string) at every prompt, and "y" at
    # the repo-detection confirmation.
    from rich.prompt import Prompt

    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: k.get("default", ""))

    run_session(
        "build a tool that renames my photos by date",
        quick=False,
        template_override="default",
        dry_run=False,
        cwd=tmp_path,
    )

    out_dir = tmp_path / "keel"
    md_files = list(out_dir.glob("*.md"))
    json_files = list(out_dir.glob("*.json"))
    assert len(md_files) == 1
    assert len(json_files) == 1

    content = md_files[0].read_text(encoding="utf-8")
    for heading in (
        "## Context",
        "## Objective",
        "## Input / Output contract",
        "## Constraints",
        "## Acceptance criteria",
        "## Non-goals",
        "## Open questions",
    ):
        assert heading in content

    session = SessionState.model_validate(json.loads(json_files[0].read_text(encoding="utf-8")))
    assert session.original_prompt == "build a tool that renames my photos by date"
    assert session.template_name == "default"


def test_amend_changes_one_slot_without_reasking_others(tmp_path: Path, monkeypatch, capsys):
    from rich.prompt import Prompt

    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: k.get("default", ""))
    run_session(
        "build a tool that renames my photos by date",
        quick=False,
        template_override="default",
        dry_run=False,
        cwd=tmp_path,
    )
    json_path = next((tmp_path / "keel").glob("*.json"))
    before = SessionState.model_validate(json.loads(json_path.read_text(encoding="utf-8")))

    answers = iter(["1", "a brand new answer for io_contract"])
    monkeypatch.setattr(Prompt, "ask", lambda *a, **k: next(answers))

    amend(str(json_path))

    after = SessionState.model_validate(json.loads(json_path.read_text(encoding="utf-8")))
    changed_slot = list(before.slots.keys())[0]
    assert after.slots[changed_slot].value == "a brand new answer for io_contract"
    for name in list(before.slots.keys())[1:]:
        assert after.slots[name].value == before.slots[name].value


def test_quick_mode_asks_fewer_questions(tmp_path: Path, monkeypatch):
    from rich.prompt import Prompt

    calls = {"count": 0}

    def fake_ask(*args, **kwargs):
        calls["count"] += 1
        return kwargs.get("default", "")

    monkeypatch.setattr(Prompt, "ask", fake_ask)
    run_session(
        "build a tool that renames my photos by date",
        quick=True,
        template_override="default",
        dry_run=False,
        cwd=tmp_path,
    )
    # 1 confirmation prompt (no project detected -> skipped) + at most 3 slot questions
    assert calls["count"] <= 3
