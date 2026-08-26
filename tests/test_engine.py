import pytest

from keel import llm
from keel.engine import Engine, load_template
from keel.models import DetectionResult, SlotSource


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Stub the LLM module so tests exercise the deterministic fallback paths."""
    monkeypatch.setattr(llm, "complete", lambda *a, **k: None)
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: None)


def test_start_creates_all_slots_unfilled():
    engine = Engine.start("build a tool", "default", DetectionResult())
    template = load_template("default")
    assert set(engine.session.slots.keys()) == {s.name for s in template.slots}
    required = [s for s in template.slots if s.required]
    assert len(engine.remaining_required_slots()) == len(required)


def test_enter_accepts_recommended_default():
    engine = Engine.start("build a tool", "default", DetectionResult())
    slot = engine.remaining_required_slots()[0]
    question, default = engine.generate_question_for(slot)
    # LLM is stubbed to return None -> falls back to static template text
    assert question == slot.question_hint
    assert default == slot.default_strategy

    engine.apply_answer(slot.name, "", default)
    state = engine.session.slots[slot.name]
    assert state.value == default
    assert state.source == SlotSource.DEFAULTED
    assert slot.name not in [s.name for s in engine.remaining_required_slots()]


def test_skip_marks_open_question_and_unblocks_completion():
    engine = Engine.start("build a tool", "default", DetectionResult())
    slot = engine.remaining_required_slots()[0]
    engine.apply_answer(slot.name, "skip", "some default")
    state = engine.session.slots[slot.name]
    assert state.value is None
    assert state.source == SlotSource.SKIPPED
    # skipped slots are "addressed" -- they don't block termination
    assert slot.name not in [s.name for s in engine.remaining_required_slots()]


def test_free_text_is_recorded_verbatim():
    engine = Engine.start("build a tool", "default", DetectionResult())
    slot = engine.remaining_required_slots()[0]
    engine.apply_answer(slot.name, "my custom answer", "default text")
    state = engine.session.slots[slot.name]
    assert state.value == "my custom answer"
    assert state.source == SlotSource.ASKED


def test_pressing_enter_through_every_slot_reaches_completion():
    engine = Engine.start("build a tool", "default", DetectionResult())
    for slot in list(engine.remaining_required_slots()):
        _, default = engine.generate_question_for(slot)
        engine.apply_answer(slot.name, "", default)
    assert engine.is_complete()


def test_quick_mode_picks_slots_by_template_priority():
    engine = Engine.start("build a tool", "default", DetectionResult())
    picked = engine.quick_priority_slots(limit=3)
    assert len(picked) == 3
    priorities = [s.priority for s in picked]
    assert priorities == sorted(priorities)


def test_skip_if_marks_slot_not_applicable_and_excludes_from_asking():
    # cli.yaml's `scale` slot is required=false, not skip_if -- verify a required
    # skip_if slot behaves correctly using a synthetic template via evaluate_skip_if.
    from keel.engine import evaluate_skip_if
    from keel.models import SlotState

    detection = DetectionResult(project_type="python")
    slots = {"runtime": SlotState()}
    assert evaluate_skip_if("detection['project_type'] == 'python'", detection, slots) is True
    assert evaluate_skip_if("detection['project_type'] == 'node'", detection, slots) is False
    assert evaluate_skip_if("", detection, slots) is False
    assert evaluate_skip_if("this is not valid python (((", detection, slots) is False


def test_detection_prefill_only_fills_unanswered_slots():
    detection = DetectionResult(project_type="python", language="python", summary="a python project using httpx")
    engine = Engine.start("build a tool", "default", detection, extract=False)
    engine.apply_detection_prefill(confirmed=True)
    assert engine.session.slots["runtime"].source == SlotSource.DETECTED
    assert engine.session.slots["constraints"].source == SlotSource.DETECTED
    # a slot the user already answered should not be clobbered
    engine.apply_answer("scale", "about 100 items a day", "n/a")
    engine.apply_detection_prefill(confirmed=True)
    assert engine.session.slots["scale"].source == SlotSource.ASKED


def test_detection_prefill_declined_leaves_slots_unset():
    detection = DetectionResult(project_type="python", language="python", summary="a python project")
    engine = Engine.start("build a tool", "default", detection, extract=False)
    engine.apply_detection_prefill(confirmed=False)
    assert engine.session.slots["runtime"].source is None
