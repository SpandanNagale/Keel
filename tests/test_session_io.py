"""Phase 3: serialise a session to JSON and back, with a schema-version gate."""
from __future__ import annotations

import json

from keel import engine, session_io
from keel.models import SlotValue


def _session():
    s = engine.start_session(
        "scrape my bookmarks and cluster them", "data-pipeline",
        created_date="2026-09-01", depth="thorough",
    )
    s.slots = {
        "io_contract": SlotValue(value="reads bookmark HTML", source="extracted"),
        "scale": SlotValue(value="about 1000 rows", source="asked"),
    }
    s.conflicts = [{"slots": ["scale", "constraints"], "conflict": "x", "suggested_resolution": "y"}]
    s.conflict_check_error = None
    s.degraded = True
    s.finished = True
    s.call_count = 7
    s.regen_count = 1
    return s


def test_round_trip_preserves_every_field():
    s = _session()
    back, err = session_io.loads(session_io.dumps(s))
    assert err is None
    assert back.model_dump() == s.model_dump()


def test_dump_carries_the_schema_version_and_a_body():
    d = json.loads(session_io.dumps(_session()))
    assert d["schema_version"] == session_io.SCHEMA_VERSION
    assert isinstance(d["keel_session"], dict)


def test_rejects_non_json():
    back, err = session_io.loads("this is not json {")
    assert back is None and "JSON" in err


def test_rejects_a_file_with_no_schema_version():
    back, err = session_io.loads(json.dumps({"keel_session": {}}))
    assert back is None and "schema_version" in err


def test_rejects_a_newer_schema_version():
    payload = json.dumps(
        {"schema_version": session_io.SCHEMA_VERSION + 1, "keel_session": {}}
    )
    back, err = session_io.loads(payload)
    assert back is None and "newer version" in err


def test_rejects_a_body_that_does_not_validate():
    payload = json.dumps(
        {"schema_version": session_io.SCHEMA_VERSION, "keel_session": {"template_name": "x"}}
    )
    back, err = session_io.loads(payload)
    assert back is None and "expected shape" in err


def test_call_count_survives_a_round_trip_so_re_upload_cannot_reset_the_budget():
    s = _session()
    s.call_count = 18
    back, _ = session_io.loads(session_io.dumps(s))
    assert back.call_count == 18
    assert engine.regenerations_left(back) == 1  # (20 - 18) // 2
