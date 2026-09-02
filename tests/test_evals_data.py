"""Phase 5: static checks on the eval dataset — no LLM, no full run.

Guards against a mislabelled slot name or a malformed case sneaking into
cases.json, and against the frozen fixtures drifting out of the schema.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from keel import engine, session_io

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EVALS = _ROOT / "evals"

_ALL_SLOTS = {
    s.name for name in engine.list_templates() for s in engine.load_template(name).slots
}


def _cases():
    return json.loads((_EVALS / "cases.json").read_text("utf-8"))["cases"]


def test_cases_file_is_well_formed_and_has_enough_cases():
    cases = _cases()
    assert len(cases) >= 12
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case id"
    for c in cases:
        for key in ("id", "prompt", "required_slots", "persona", "prompt_facts"):
            assert c.get(key), f"{c.get('id')}: missing {key}"
        assert isinstance(c["required_slots"], list) and c["required_slots"]
        assert isinstance(c["expected_conflicts"], list)


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_every_labelled_required_slot_exists(case):
    for name in case["required_slots"]:
        assert name in _ALL_SLOTS, f"{case['id']}: unknown slot {name!r}"


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_declared_template_is_real_if_present(case):
    if "template" in case:
        assert case["template"] in engine.list_templates()


FIXTURE_STEMS = {
    "hotel-website-contradiction",
    "health-monitoring-offline-ai",
    "feasibility-devserver-concurrency",
    "qualitative-only",
}


def test_frozen_fixtures_load_and_match_the_schema():
    files = sorted((_EVALS / "fixtures").glob("*.json"))
    assert {f.stem for f in files} == FIXTURE_STEMS
    for f in files:
        session, err = session_io.loads(f.read_text("utf-8"))
        assert err is None, f"{f.name}: {err}"
        assert session.finished
        template = engine.load_template(session.template_name)
        # every slot filled with a real slot name
        for name in session.slots:
            assert template.slot(name) is not None
        # every slot in play for this session (auth_model may be suppressed when
        # the non-goals already exclude authentication) is filled
        visible = {s.name for s in engine.visible_slots(session, template)}
        assert set(session.slots) == visible


def test_eval_harness_is_wired_and_runnable():
    """The harness must import and expose its pieces without an LLM. RESULTS.md
    itself is a generated snapshot (``python evals/run_evals.py``) — regenerated
    against the hosted model and committed when the prompts change."""
    import importlib

    mod = importlib.import_module("evals.run_evals")
    for attr in ("run_case", "run_fixture", "write_results", "_spec_body", "_numbers",
                 "_FIXTURE_EXPECT", "_FEASIBILITY_FIXTURES", "_NO_FIGURE_FIXTURES"):
        assert hasattr(mod, attr), f"run_evals is missing {attr}"
    # every fixture on disk is a known regression case (a contradiction fixture
    # in _FIXTURE_EXPECT, or a numeric-honesty fixture in _NO_FIGURE_FIXTURES)
    assert set(mod._FIXTURE_EXPECT) | mod._NO_FIGURE_FIXTURES == FIXTURE_STEMS
    assert mod._FEASIBILITY_FIXTURES <= set(mod._FIXTURE_EXPECT)
    # deterministic helpers work with no model
    assert mod._numbers("handles 2000 rows at 12 MB") == {"2000", "12"}
    assert "## Objective" not in mod._spec_body(
        "# T\n\n## Context\n\nbody\n\n---\n*foot*"
    )
