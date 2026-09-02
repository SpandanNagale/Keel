"""Run every eval case unattended and write evals/RESULTS.md.

    python evals/run_evals.py                 # provider auto-picked (env / secrets.toml)
    KEEL_PROVIDER=ollama python evals/run_evals.py    # local model, saves hosted quota
    python evals/run_evals.py --limit 3       # first 3 cases, for a quick check

Metrics are all computed in Python — no judge model. Commit RESULTS.md so a
prompt change that moves a number shows up in the diff. The README notes which
model produced the committed table.
"""
from __future__ import annotations

import argparse
import functools
import json
import pathlib
import re
import sys
import tomllib
from datetime import date

print = functools.partial(print, flush=True)  # noqa: A001 - unbuffered progress

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from keel import engine, llm, render, session_io  # noqa: E402
from evals.simulated_user import SimulatedUser  # noqa: E402

_HERE = pathlib.Path(__file__).parent
_FIXTURES = _HERE / "fixtures"
_RESULTS = _HERE / "RESULTS.md"


# --------------------------------------------------------------------------- #
def _resolve_provider():
    # Also honour KEEL_PROVIDER / KEEL_OLLAMA_MODEL from secrets.toml so a local
    # run can be pinned there instead of exported on every command line.
    _extra = ("KEEL_PROVIDER", "KEEL_OLLAMA_MODEL")
    available: dict[str, str] = {}
    secrets = _ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        data = tomllib.loads(secrets.read_text("utf-8"))
        for k in (*llm.SECRET_KEYS.values(), *_extra):
            if data.get(k):
                available[k] = str(data[k])
    import os
    for k in (*llm.SECRET_KEYS.values(), *_extra):
        if os.environ.get(k):
            available[k] = os.environ[k]
    return llm.resolve_provider(available)


def _numbers(text: str) -> set[str]:
    return render._numbers_in(text or "")


def _spec_body(md: str) -> str:
    """Just the section bodies — drop the title line and the footer."""
    if "## " not in md:
        return md
    body = md.split("## ", 1)[1]
    return body.split("\n---", 1)[0]


# --------------------------------------------------------------------------- #
def run_case(case: dict, provider, persona_provider) -> dict:
    prompt = case["prompt"]
    template_name = case.get("template") or engine.select_template(prompt)
    session = engine.start_session(prompt, template_name, created_date="2026-09-01",
                                   depth="standard")
    template = engine.load_template(template_name)

    engine.extract_prefilled(session, template, provider=provider)
    extracted = {n for n, v in session.slots.items() if v.source == "extracted"}
    engine.freeze_pending(session, template)

    user = SimulatedUser(case["persona"], persona_provider)
    asked: list[str] = []
    questions: list[str] = []
    guard = 0
    while not session.finished and guard < 20:
        guard += 1
        slot = engine.current_slot(session, template)
        if slot is None:
            break
        proposal = engine.next_question(session, template, provider=provider)
        q, rec, q_err = proposal.question, proposal.recommended, proposal.error
        asked.append(slot.name)
        questions.append(q)
        established = [f"{n}: {v.value}" for n, v in session.slots.items() if v.value]
        action, text = user.respond(q, rec, established)
        if action == "skip":
            engine.skip_slot(session, slot.name)
        else:
            engine.accept_answer(
                session, slot.name, text, recommended=rec,
                recommended_source="template_default" if q_err else "llm_default",
            )

    engine.fill_unasked_slots(session, template, provider=provider)
    conflicts, conflict_err = render.check_conflicts(session, provider=provider)
    session.conflicts, session.conflict_check_error = conflicts, conflict_err
    md, synth_err = render.synthesize_spec(session, provider=provider,
                                           conflicts=conflicts, conflict_error=conflict_err)
    md = md or render.render_markdown(session)

    # ---- metrics ----
    req = case["required_slots"]
    filled = [
        n for n in req
        if (v := session.slots.get(n)) and v.source != "skipped" and v.value.strip()
    ]
    coverage = len(filled) / len(req) if req else 1.0

    considered = extracted | set(asked)
    missed = [n for n in req if n not in considered]
    missed_rate = len(missed) / len(req) if req else 0.0

    facts = [f.lower() for f in case.get("prompt_facts", [])]
    redundant = [q for q in questions if any(f in q.lower() for f in facts)]
    redundancy = len(redundant) / len(questions) if questions else 0.0

    expected = case.get("expected_conflicts", [])
    if expected:
        blob = " ".join(c["conflict"] for c in conflicts).lower()
        hits = sum(1 for term in expected if term.lower() in blob)
        conflict_recall = hits / len(expected)
    else:
        conflict_recall = None

    known = _numbers(prompt + " " + " ".join(v.value for v in session.slots.values()))
    doc_nums = _numbers(_spec_body(md))
    untraceable = doc_nums - known
    false_precision = len(untraceable) / len(doc_nums) if doc_nums else 0.0

    return {
        "id": case["id"],
        "template": template_name,
        "asked": session.questions_asked,
        "coverage": coverage,
        "missed": missed,
        "missed_rate": missed_rate,
        "redundancy": redundancy,
        "conflict_recall": conflict_recall,
        "false_precision": false_precision,
        "untraceable_numbers": sorted(untraceable),
        "synth_fell_back": synth_err is not None,
        "degraded": session.degraded,
    }


# Each fixture's contradiction, as term-groups. "Caught" means the conflict text
# hits at least one term in EVERY group — i.e. it names the real contradiction,
# not just some unrelated conflict.
_FIXTURE_EXPECT = {
    "hotel-website-contradiction": [
        ["ui", "frontend", "front-end", "front end", "browser", "web page", "pages",
         "interface", "website"],
    ],
    "health-monitoring-offline-ai": [
        ["offline", "standard library", "standard-library", "stdlib", "no external",
         "local model", "without a model", "no model"],
        ["ai", "ai-generated", "ai generated", "interpretation", "language model",
         "inference"],
    ],
    "feasibility-devserver-concurrency": [
        ["development server", "dev server", "built-in server", "single process",
         "single-process", "one process"],
        ["concurren", "at the same time", "at once", "everyone", "hundreds", "many users",
         "load", "simultaneous"],
    ],
}

# Fixtures whose expected conflict must be reported with kind "feasibility".
_FEASIBILITY_FIXTURES = {"feasibility-devserver-concurrency"}

# Fixtures whose regression check is "the rendered spec invents no capacity /
# throughput / volume figure", not a contradiction. Their answers are entirely
# qualitative, so any bare quantity in the body that traces to no answer fails.
_NO_FIGURE_FIXTURES = {"qualitative-only"}


def run_fixture(path: pathlib.Path, provider) -> dict:
    session, err = session_io.loads(path.read_text("utf-8"))
    if err:
        return {"id": path.stem, "result": "LOAD-FAIL", "detail": err}
    conflicts, cerr = render.check_conflicts(session, provider=provider)
    session.conflicts, session.conflict_check_error = conflicts, cerr
    # synthesize_spec re-validates conflicts against the document and reduces
    # session.conflicts to the survivors (Bug 2). We check detection against the
    # pre-synthesis list; the survivor/resolved counts show the re-validation.
    md, serr = render.synthesize_spec(session, provider=provider,
                                      conflicts=conflicts, conflict_error=cerr)
    blob = " ".join(
        f"{c.get('conflict', '')} {' '.join(c.get('slots') or [])}" for c in conflicts
    ).lower()

    # Numeric-honesty fixture: any bare quantity in the body that traces to no
    # answer is a regression.
    if path.stem in _NO_FIGURE_FIXTURES:
        known = _numbers(
            session.original_prompt
            + " " + " ".join(v.value for v in session.slots.values())
        )
        # Ordered-list markers ("1. ", "2. ") in Build order are structure, not
        # figures — render._scrub_unsupported_figures skips them too.
        body = re.sub(r"(?m)^(\s*)\d+\.\s", r"\1", _spec_body(md or ""))
        strays = sorted(
            n for n in _numbers(body)
            if n not in known and not render._YEARISH.match(n)
        )
        return {
            "id": path.stem,
            "n_conflicts": len(conflicts),
            "survived": len(session.conflicts),
            "resolved": len(session.resolved_conflicts),
            "caught": not strays,
            "synth": "fell-back" if serr else "ok",
            "result": "PASS" if (not strays and cerr is None) else "FAIL",
            "detail": (f"invented figures: {', '.join(strays)}" if strays else "")
            or cerr or serr or "",
        }

    groups = _FIXTURE_EXPECT.get(path.stem, [])
    caught = bool(conflicts) and all(
        any(term in blob for term in group) for group in groups
    )
    if path.stem in _FEASIBILITY_FIXTURES:
        caught = caught and any(c.get("kind") == "feasibility" for c in conflicts)
    return {
        "id": path.stem,
        "n_conflicts": len(conflicts),
        "survived": len(session.conflicts),
        "resolved": len(session.resolved_conflicts),
        "caught": caught,
        "synth": "fell-back" if serr else "ok",
        "result": "PASS" if (caught and cerr is None) else "FAIL",
        "detail": cerr or serr or "",
    }


# --------------------------------------------------------------------------- #
def _fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def write_results(rows: list[dict], fixtures: list[dict], provider, persona_mode: str):
    n = len(rows)
    mean = lambda key: sum(r[key] for r in rows) / n if n else 0.0
    recalls = [r["conflict_recall"] for r in rows if r["conflict_recall"] is not None]

    out: list[str] = []
    out.append("# Keel eval results\n")
    out.append(
        f"_Generated {date.today().isoformat()} · "
        f"provider `{provider.name if provider else 'none'}` · "
        f"model `{provider.model if provider else 'n/a'}` · "
        f"persona `{persona_mode}` · {n} cases_\n"
    )
    out.append("Regenerate with `python evals/run_evals.py`. Numbers move when the "
               "question, conflict, or synthesis prompts change — that is the point.\n")

    out.append("## Aggregate\n")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Slot coverage (labelled-required filled) | {_fmt(mean('coverage'))} |")
    out.append(f"| Question efficiency (mean questions asked) | {_fmt(mean('asked'), 1)} |")
    out.append(f"| Missed-ambiguity rate | {_fmt(mean('missed_rate'))} |")
    out.append(f"| Redundancy rate | {_fmt(mean('redundancy'))} |")
    out.append(
        f"| Conflict recall | {_fmt(sum(recalls) / len(recalls) if recalls else None)} "
        f"({len(recalls)} case{'s' if len(recalls) != 1 else ''}) |"
    )
    out.append(f"| False-precision rate | {_fmt(mean('false_precision'))} |")
    out.append("")

    out.append("## Per case\n")
    out.append("| Case | Template | Asked | Coverage | Missed slots | Redundancy | "
               "Conflicts caught | False precision |")
    out.append("|---|---|--:|--:|---|--:|--:|--:|")
    for r in rows:
        out.append(
            f"| {r['id']} | {r['template']} | {r['asked']} | {_fmt(r['coverage'])} | "
            f"{', '.join(r['missed']) or '—'} | {_fmt(r['redundancy'])} | "
            f"{_fmt(r['conflict_recall'])} | {_fmt(r['false_precision'])} |"
        )
    out.append("")

    flagged = [r for r in rows if r["untraceable_numbers"]]
    if flagged:
        out.append("### Numbers in the spec that trace to no answer\n")
        for r in flagged:
            out.append(f"- **{r['id']}**: {', '.join(r['untraceable_numbers'])}")
        out.append("")

    out.append("## Regression fixtures\n")
    out.append("| Fixture | Conflicts | Survived | Resolved | Caught | Synthesis | Result |")
    out.append("|---|--:|--:|--:|:--:|:--:|:--:|")
    for f in fixtures:
        out.append(
            f"| {f['id']} | {f.get('n_conflicts', '—')} | {f.get('survived', '—')} | "
            f"{f.get('resolved', '—')} | {'yes' if f.get('caught') else 'no'} | "
            f"{f.get('synth', '—')} | {f['result']} |"
        )
    out.append("")

    _RESULTS.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {_RESULTS.relative_to(_ROOT)}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    ap.add_argument("--no-write", action="store_true", help="print, do not touch RESULTS.md")
    ap.add_argument("--persona", choices=["llm", "deterministic", "auto"], default="auto",
                    help="how the simulated user answers (default: llm if a provider is "
                         "configured, else deterministic)")
    args = ap.parse_args()

    provider, reason = _resolve_provider()
    if provider is None:
        print(f"warning: {reason}\n         Keel's own calls will fail; expect degraded specs.")
    if args.persona == "auto":
        persona_mode = "llm" if provider else "deterministic"
    else:
        persona_mode = args.persona
    persona_provider = provider if persona_mode == "llm" else None
    if provider:
        print(f"provider: {provider.name} · model: {provider.model} · persona: {persona_mode}")

    cases = json.loads((_HERE / "cases.json").read_text("utf-8"))["cases"]
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    for c in cases:
        print(f"  · {c['id']}")
        rows.append(run_case(c, provider, persona_provider))

    fixtures = []
    for fx in sorted(_FIXTURES.glob("*.json")):
        print(f"  · fixture {fx.stem}")
        fixtures.append(run_fixture(fx, provider))

    if args.no_write:
        print(json.dumps({"cases": rows, "fixtures": fixtures}, indent=2, default=str))
    else:
        write_results(rows, fixtures, provider, persona_mode)

    failed = [f["id"] for f in fixtures if f["result"] != "PASS"]
    if failed:
        print(f"REGRESSION: fixtures failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
