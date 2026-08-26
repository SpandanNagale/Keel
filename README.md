# Keel

A CLI prompt compiler. Keel takes a vague, one-line project idea, interrogates you about
the specific things left unstated, and writes a structured, agent-ready prompt as a
markdown file next to your code — so there's no copy-paste step between you and your
coding agent.

## Install

```
pip install -e .
```

Requires Python 3.11+ and an `ANTHROPIC_API_KEY` environment variable.

## Usage

```
keel "scrape my bookmarks and cluster them by topic"
```

Keel inspects the current directory, picks a slot template based on what it finds, and
asks one clarifying question at a time — each with a recommended default you can accept
by pressing Enter. It stops as soon as every required slot is filled and writes:

```
./keel/2026-08-26-bookmark-clusterer.md
./keel/2026-08-26-bookmark-clusterer.json
```

At any prompt you can:
- press **Enter** to accept the recommended default
- type your own answer
- type **`?`** to see why Keel is asking
- type **`skip`** to leave it as an open question in the output

### Other commands

```
keel --quick "<prompt>"              # ask only the 2-3 highest-priority slots, default the rest
keel --template api "<prompt>"       # force a specific slot template
keel --dry-run "<prompt>"            # show which slots would be asked, no LLM calls
keel amend ./keel/2026-08-26-x.json  # reload a session, change one answer, regenerate
```

## How it works

The core abstraction is a **slot** — a named dimension a prompt must pin down before a
coding agent can act on it (input/output contract, scale, runtime, constraints,
definition of done, non-goals). Slots are defined in YAML templates under
`keel/templates/`, never hardcoded. A slot is filled from, in priority order:

1. **Extraction** from your original prompt
2. **Repo detection** (`pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, `.git/config`, ...)
3. **Asking** — one question per turn, only for what's left

Keel terminates when every required slot in the active template has been addressed —
never when a model decides it has asked "enough."

All LLM calls go through the single `complete()` / `complete_json()` functions in
`keel/llm.py`. On any API failure, Keel falls back to the slot's static `question_hint`
and `default_strategy` rather than crashing mid-session.

## Evals

```
python evals/run_evals.py
```

Runs ~20 hand-labelled realistic prompts through a simulated user and writes
`evals/RESULTS.md` with slot coverage, question efficiency, missed-ambiguity rate, and
redundancy rate — no judge model involved. Pass `--e2e` to additionally run the 5
end-to-end cases (feeds the generated prompt to Claude and judges the result against
the case's expected output) — this is slower and uses an LLM judge, so it's opt-in.
