"""Keel's command-line interface."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt

if sys.platform == "win32":
    # Windows consoles default to a legacy codepage that can't render em-dashes
    # and other punctuation used in generated questions/prompts.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from keel.detect import detect_project, suggest_template
from keel.engine import Engine, list_templates, load_template, slugify
from keel.models import SessionState, SlotSource, SlotState
from keel.render import render_markdown

TEMPLATE_ALIASES = {"api": "web-api"}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
console = Console()


def _output_paths(session: SessionState, out_dir: Path) -> tuple[Path, Path]:
    date = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(session.title)
    return out_dir / f"{date}-{slug}.md", out_dir / f"{date}-{slug}.json"


def _save_json(session: SessionState, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")


def _write_outputs(session: SessionState, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = _output_paths(session, out_dir)
    md_path.write_text(render_markdown(session), encoding="utf-8")
    _save_json(session, json_path)
    return md_path, json_path


@app.command()
def main(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(
        None, help='A short, vague description of what you want to build, e.g. "scrape my bookmarks and cluster them".'
    ),
    quick: bool = typer.Option(False, "--quick", help="Ask only the 2-3 highest-priority required slots; default the rest."),
    template: Optional[str] = typer.Option(None, "--template", help="Force a specific slot template (default, cli, data-pipeline, web-api)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show which slots would be asked, without calling the LLM."),
) -> None:
    # Keel is a flat command, not a Click command group: "amend" is dispatched
    # here rather than as a separate @app.command(), because Typer/Click would
    # otherwise treat the leading positional `prompt` argument and a registered
    # `amend` subcommand as ambiguous (the group-level argument greedily
    # consumes "amend" before subcommand dispatch ever runs).
    if prompt == "amend":
        if not ctx.args:
            console.print("[red]Usage: keel amend <path-to-json>[/red]")
            raise typer.Exit(code=1)
        amend(ctx.args[0])
        return
    if not prompt:
        console.print(ctx.get_help())
        raise typer.Exit(code=1)
    run_session(prompt, quick=quick, template_override=template, dry_run=dry_run)


def run_session(
    prompt: str,
    *,
    quick: bool,
    template_override: Optional[str],
    dry_run: bool,
    cwd: Optional[Path] = None,
) -> None:
    cwd = cwd or Path.cwd()
    detection = detect_project(cwd)
    if template_override:
        template_override = TEMPLATE_ALIASES.get(template_override, template_override)
    template_name = template_override or suggest_template(detection, prompt)
    if template_name not in list_templates():
        console.print(f"[red]Unknown template '{template_name}'. Available: {', '.join(list_templates())}[/red]")
        raise typer.Exit(code=1)

    if dry_run:
        engine = Engine.start(prompt, template_name, detection, extract=False)
        _print_dry_run(engine)
        return

    engine = Engine.start(prompt, template_name, detection)

    if detection.project_type:
        confirm = Prompt.ask(
            f"Detected {detection.summary} — use it for runtime/constraints?",
            choices=["y", "n"],
            default="y",
        )
        engine.apply_detection_prefill(confirmed=confirm.lower() == "y")

    out_dir = cwd / "keel"
    _, json_path = _output_paths(engine.session, out_dir)
    _save_json(engine.session, json_path)

    slots_to_ask = engine.quick_priority_slots() if quick else engine.remaining_required_slots()
    total = len(slots_to_ask)
    for idx, slot in enumerate(slots_to_ask, start=1):
        if engine.is_filled(slot.name):
            continue
        question, default = engine.generate_question_for(slot)
        console.print(f"\n[bold cyan][{idx}/{total}][/bold cyan] {question}")
        console.print(f"      [dim]-> Recommended: {default}[/dim]")
        console.print("      [dim](Enter to accept, or type your own, or `?` for why I'm asking, or `skip`)[/dim]")
        while True:
            raw = Prompt.ask("      >", default="")
            if raw.strip() == "?":
                console.print(f"      [yellow]{slot.question_hint}[/yellow]")
                continue
            break
        engine.apply_answer(slot.name, raw, default)
        _save_json(engine.session, json_path)

    if quick:
        engine.default_remaining_silently()
        _save_json(engine.session, json_path)

    md_path, json_path = _write_outputs(engine.session, out_dir)
    _print_summary(engine.session)
    console.print(f"\n[green]Wrote {md_path}\n      {json_path}[/green]")


def _print_dry_run(engine: Engine) -> None:
    console.print(f"[bold]Template:[/bold] {engine.session.template_name}")
    console.print(f"[bold]Title:[/bold] {engine.session.title}")
    console.print("\n[bold]Slots that would be asked:[/bold]")
    for slot in engine.remaining_required_slots():
        console.print(f"  - {slot.name}: {slot.question_hint}")
    not_applicable = [
        s.name for s in engine.template.slots if engine.session.slots[s.name].source and
        engine.session.slots[s.name].source.value == "not_applicable"
    ]
    if not_applicable:
        console.print("\n[bold]Slots skipped as not applicable:[/bold]")
        for name in not_applicable:
            console.print(f"  - {name}")


def _print_summary(session: SessionState) -> None:
    console.print("\n[bold]Summary[/bold]")
    for name, state in session.slots.items():
        if state.source == SlotSource.SKIPPED:
            console.print(f"  {name}: [yellow]left open[/yellow]")
        elif state.source in (SlotSource.EXTRACTED, SlotSource.DETECTED):
            console.print(f"  {name}: [cyan]assumed[/cyan] — {state.value}")
        elif state.value:
            console.print(f"  {name}: [green]filled[/green] — {state.value}")


def amend(json_path: str) -> None:
    """Reload a prior session's slot state, change one answer, and regenerate the .md."""
    path = Path(json_path)
    if not path.exists():
        console.print(f"[red]No such file: {json_path}[/red]")
        raise typer.Exit(code=1)

    session = SessionState.model_validate(json.loads(path.read_text(encoding="utf-8")))
    template = load_template(session.template_name)
    slot_names = [s.name for s in template.slots]

    console.print("Current slots:")
    for i, name in enumerate(slot_names, start=1):
        state = session.slots.get(name)
        console.print(f"  {i}. {name}: {state.value if state and state.value else '(unset)'}")

    choice = Prompt.ask("Which slot number do you want to change? (Enter to cancel)", default="")
    if not choice.strip():
        console.print("No change made.")
        return

    try:
        slot_name = slot_names[int(choice.strip()) - 1]
    except (ValueError, IndexError):
        console.print("[red]Invalid selection.[/red]")
        raise typer.Exit(code=1)

    new_value = Prompt.ask(f"New value for {slot_name}")
    session.slots[slot_name] = SlotState(value=new_value, source=SlotSource.ASKED)

    md_path = path.with_suffix(".md")
    md_path.write_text(render_markdown(session), encoding="utf-8")
    path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]Updated {slot_name}. Regenerated {md_path}[/green]")


if __name__ == "__main__":
    app()
