from pathlib import Path

import pytest
import typer

from keel import llm
from keel.cli import run_session


def test_dry_run_makes_zero_llm_calls(tmp_path: Path, monkeypatch, capsys):
    def _boom(*args, **kwargs):
        raise AssertionError("LLM was called during --dry-run")

    monkeypatch.setattr(llm, "complete", _boom)
    monkeypatch.setattr(llm, "complete_json", _boom)

    # Should not raise, and should not touch the network / LLM module at all.
    run_session(
        "build a tool that renames my photos by date",
        quick=False,
        template_override=None,
        dry_run=True,
        cwd=tmp_path,
    )

    # dry-run must not write any output files either
    assert not (tmp_path / "keel").exists()


def test_dry_run_reports_unknown_template(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no llm")))
    with pytest.raises(typer.Exit):
        run_session(
            "build a tool",
            quick=False,
            template_override="nonexistent",
            dry_run=True,
            cwd=tmp_path,
        )
