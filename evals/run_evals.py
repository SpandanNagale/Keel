#!/usr/bin/env python
"""Runs every eval case unattended and writes evals/RESULTS.md.

Four metrics, computed without a judge model:
  - slot coverage:        fraction of labelled-required slots filled at termination
  - question efficiency:  mean number of questions asked to reach termination
  - missed-ambiguity rate: labelled-required slots never asked about, extracted, or detected
  - redundancy rate:      questions asked about information already present in the prompt

`--e2e` additionally runs the 5 end-to-end cases: it feeds Keel's generated .md to
Claude playing a coding agent and uses a second LLM call to judge whether the result
matches the case's hand-written expected_output. This is the one place in the harness
that uses a judge model -- it is deliberately kept separate from, and optional
relative to, the four core metrics above.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from keel import llm  # noqa: E402
from keel.detect import suggest_template  # noqa: E402
from keel.engine import Engine  # noqa: E402
from keel.models import DetectionResult, SlotSource  # noqa: E402
from keel.render import render_markdown  # noqa: E402
from simulated_user import SimulatedUser  # noqa: E402

CASES_PATH = Path(__file__).parent / "cases.json"
RESULTS_PATH = Path(__file__).parent / "RESULTS.md"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def run_case(case: dict) -> dict:
    prompt = case["prompt"]
    detection = DetectionResult()
    template_name = suggest_template(detection, prompt)
    engine = Engine.start(prompt, template_name, detection)
    user = SimulatedUser(case["persona"], prompt, seed=abs(hash(case["id"])) % (2**31))

    asked_slots: list[str] = []
    for slot in engine.remaining_required_slots():
        if engine.is_filled(slot.name):
            continue
        question, default = engine.generate_question_for(slot)
        raw_answer = user.answer(question, default)
        engine.apply_answer(slot.name, raw_answer, default)
        asked_slots.append(slot.name)

    template_slot_names = {s.name for s in engine.template.slots}
    required = set(case["required_slots"])
    fillable_required = required & template_slot_names
    unfillable_required = required - template_slot_names

    filled_at_end = {
        name for name in fillable_required
        if engine.session.slots.get(name) and engine.session.slots[name].value
    }
    coverage = (len(filled_at_end) / len(fillable_required)) if fillable_required else 1.0

    never_addressed = {
        name for name in fillable_required
        if name not in asked_slots
        and engine.session.slots[name].source not in (SlotSource.EXTRACTED, SlotSource.DETECTED)
        and not engine.session.slots[name].value
    }
    missed_ambiguity = unfillable_required | never_addressed

    answered_in_prompt = set(case.get("answered_in_prompt", []))
    redundant = {name for name in answered_in_prompt if name in asked_slots}

    return {
        "id": case["id"],
        "template": template_name,
        "questions_asked": len(asked_slots),
        "required_count": len(required),
        "coverage": coverage,
        "missed_ambiguity": sorted(missed_ambiguity),
        "answered_in_prompt": sorted(answered_in_prompt),
        "redundant": sorted(redundant),
        "session": engine.session,
    }


E2E_AGENT_SYSTEM = """You are a coding agent. You will be given a structured project brief.
Describe, concretely, the solution you would build: the main script/module(s), the key
functions, and exactly what the output looks like for a representative input. Be specific
about file formats and data shapes. Do not write out full source code -- a precise design
description is enough. Keep it under 200 words."""

E2E_JUDGE_SYSTEM = """You are grading whether a coding agent's planned solution would satisfy
a developer's actual intent. You will see the developer's persona/expected outcome and the
agent's planned solution. Respond with ONLY a JSON object: {"match": true|false, "reason": "..."}.
"match" is true only if the plan would produce something the persona described as wanting."""


def run_e2e_case(result: dict, case: dict) -> dict:
    md = render_markdown(result["session"])
    plan = llm.complete(E2E_AGENT_SYSTEM, md, max_tokens=500, effort="medium")
    if plan is None:
        return {"id": case["id"], "match": None, "reason": "LLM unavailable"}

    judge_input = (
        f"Persona / expected outcome: {case['persona']}\n"
        f"Expected output: {case.get('expected_output', '')}\n\n"
        f"Agent's planned solution:\n{plan}"
    )
    verdict = llm.complete_json(E2E_JUDGE_SYSTEM, judge_input, max_tokens=300, effort="low")
    if not verdict:
        return {"id": case["id"], "match": None, "reason": "judge call failed"}
    return {"id": case["id"], "match": bool(verdict.get("match")), "reason": str(verdict.get("reason", ""))}


def render_results(results: list[dict], e2e_results: list[dict] | None) -> str:
    n = len(results)
    mean_coverage = sum(r["coverage"] for r in results) / n
    mean_questions = sum(r["questions_asked"] for r in results) / n

    total_required_slots = sum(r["required_count"] for r in results)
    total_missed = sum(len(r["missed_ambiguity"]) for r in results)
    missed_ambiguity_rate = (total_missed / total_required_slots) if total_required_slots else 0.0

    total_answered_in_prompt = sum(len(r["answered_in_prompt"]) for r in results)
    total_redundant = sum(len(r["redundant"]) for r in results)
    redundancy_rate = (total_redundant / total_answered_in_prompt) if total_answered_in_prompt else 0.0

    lines = ["# Keel Eval Results", ""]
    lines.append(f"Cases run: {n}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Slot coverage (mean) | {mean_coverage:.1%} |")
    lines.append(f"| Question efficiency (mean questions/case) | {mean_questions:.2f} |")
    lines.append(f"| Missed-ambiguity rate | {missed_ambiguity_rate:.1%} |")
    lines.append(f"| Redundancy rate | {redundancy_rate:.1%} |")
    lines.append("")

    lines.append("## Per-case detail")
    lines.append("")
    lines.append("| Case | Template | Questions | Coverage | Missed slots | Redundant slots |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['id']} | {r['template']} | {r['questions_asked']} | {r['coverage']:.0%} | "
            f"{', '.join(r['missed_ambiguity']) or '-'} | {', '.join(r['redundant']) or '-'} |"
        )
    lines.append("")

    if e2e_results is not None:
        lines.append("## End-to-end check (5 cases, LLM-judged, opt-in via --e2e)")
        lines.append("")
        lines.append(
            "This step feeds Keel's generated `.md` to an LLM playing a coding agent, then "
            "judges the plan against the case's hand-written `expected_output`. Unlike the four "
            "metrics above, this uses a judge model and is a proxy for actually running a coding "
            "agent end to end."
        )
        lines.append("")
        lines.append("| Case | Match | Reason |")
        lines.append("|---|---|---|")
        for e in e2e_results:
            match_str = "yes" if e["match"] is True else ("no" if e["match"] is False else "n/a")
            lines.append(f"| {e['id']} | {match_str} | {e['reason']} |")
        lines.append("")
        passed = sum(1 for e in e2e_results if e["match"] is True)
        lines.append(f"Passed: {passed}/{len(e2e_results)}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true", help="Also run the 5 end-to-end cases (uses an LLM judge).")
    args = parser.parse_args()

    cases = load_cases()
    results = [run_case(case) for case in cases]

    e2e_results = None
    if args.e2e:
        e2e_cases = [c for c in cases if c.get("end_to_end")]
        by_id = {r["id"]: r for r in results}
        e2e_results = [run_e2e_case(by_id[c["id"]], c) for c in e2e_cases]

    RESULTS_PATH.write_text(render_results(results, e2e_results), encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
