# ⛵ Keel

Keel takes a vague, one-line software project idea, asks a short series of targeted
clarifying questions about what the idea leaves unstated, and produces a structured,
downloadable markdown prompt you can paste into a coding agent (Claude Code, Cursor,
Copilot).

It is a single-page [Streamlit](https://streamlit.io) app.

## Run it locally

```
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add one key
streamlit run app.py
```

`.streamlit/secrets.toml` holds your API key server-side and is gitignored. Keel supports
three providers and picks the first key it finds, in this order:

1. `GROQ_API_KEY` — Groq Cloud (`openai/gpt-oss-120b`)
2. `OLLAMA_API_KEY` — Ollama Cloud (`gpt-oss:120b`)
3. `ANTHROPIC_API_KEY` — Anthropic (`claude-haiku-4-5-20251001`)

Set `KEEL_MODEL` to override the model for whichever provider is active. The app still
runs with no key — degraded, clearly labelled, using template defaults. The sidebar also
takes a per-session key for any of the three providers.

`FIRECRAWL_API_KEY` is optional — it powers the "reference for structure" feature
([Firecrawl](https://firecrawl.dev)). Without it, that input is disabled and everything
else works unchanged.

### Local Ollama for development

Prompt iteration (especially the synthesis prompt) burns hosted quota fast. For local
work, point Keel at an Ollama daemon:

```
ollama serve                                  # in another terminal
ollama pull llama3.2                           # or any small instruct model
KEEL_PROVIDER=ollama streamlit run app.py
```

`KEEL_PROVIDER=ollama` uses `http://localhost:11434` with no API key. Override the model
with `KEEL_OLLAMA_MODEL` (default `llama3.2`). If nothing answers on localhost the error
says so — it never silently pretends the call worked, and the deployed app never takes
this path unless a deployer sets that env var.

**Caveat:** a small local model produces noticeably weaker synthesis than the hosted
models. Use Ollama to iterate on plumbing, flow, and prompt *structure* — but validate
final output quality against a hosted model before judging a prompt change good. Do not
tune the synthesis prompt exclusively against local output.

`KEEL_PROVIDER` also accepts `groq`, `ollama-cloud`, or `anthropic` to force a specific
hosted provider instead of the key-presence default.

### Smoke-test the model connection

```
python doctor.py                        # provider auto-picked from env / secrets.toml
python doctor.py --provider groq        # force a hosted provider
KEEL_PROVIDER=ollama python doctor.py   # test the local daemon
```

Makes exactly one live LLM call and prints the raw failure reason if it fails. Run this
before deploying — it is the check the previous build lacked.

## The slot model

A **slot** is a named dimension a prompt must pin down. Keel asks a question because a
slot is empty and stops when every required slot is addressed — termination is a loop
over the slot list, never "the model decides it has asked enough."

| Slot | Captures | Tier |
|---|---|---|
| `io_contract` | What goes in, what comes out, in what format | core |
| `scale` | Volume, frequency, data size | core |
| `runtime` | Where it runs — CLI, cron, server, notebook, browser | core |
| `constraints` | Hard limits: offline, no paid APIs, language, existing deps | core |
| `done` | Observable definition of done | core |
| `non_goals` | What the agent must NOT build | core |
| `data_model` | The core entities and the fields each record carries | optional |
| `interfaces` | The concrete surface: endpoints, commands, screens, functions | optional |
| `error_handling` | What happens on bad input, failure, or partial success | optional |

An **audience mode** toggle on the start screen decides how questions are asked:

- **Guided** (default) assumes no software background. It asks only the six core
  slots, ignores the depth selector, phrases `runtime` / `scale` / `constraints`
  as plain-language multiple choice (plus "Something else" and "I'm not sure"),
  and shows a *Why does this matter?* note under every question. Everything not
  asked is defaulted — closed-answer slots become `keel_decided`, open ones
  `llm_default`.
- **Technical** keeps today's terse free-text questions and the **Depth**
  selector: `Quick` asks the six core, `Standard` (default) adds `data_model` and
  `interfaces`, `Thorough` adds `error_handling`.

Every question also offers **Decide for me** (records the value as `keel_decided`
with a reason) alongside **Skip this** (leaves it open). No session asks more than
**8** questions; any slot not asked is filled from the answers already given (one
LLM call), never left blank.

Slots live in YAML under [`keel/templates/`](keel/templates/) — `default`, `cli`,
`data-pipeline`, `web-api`, `web-app`. Templates differ in question phrasing, priority,
and section mapping (`web-app` is for browser-facing apps and never lists a UI as a
non-goal; `web-api` is reserved for HTTP services). Keel picks one by keyword heuristic on
your idea, preferring `default` when the prompt is ambiguous; a dropdown lets you override.

Each slot carries a `default_text` (a concrete human-readable fallback answer), a
`default_strategy` (an instruction used *only* inside the LLM prompt to generate a
context-aware default; never rendered — there is a test that asserts it), a hand-written
`why_this_matters` note, and — for the closed-answer slots — a hand-written `choices`
list (`label`, `plain_language`, `tradeoff`). The choice and why text are static: no
LLM call, identical between runs.

## Architecture

```
app.py                    Streamlit UI and flow control only
keel/models.py            pydantic models
keel/engine.py            slot state machine, template loading, the capped LLM helper
keel/render.py            synthesis pass + deterministic fallback -> markdown
keel/llm.py               the one place an LLM is called; returns (result, error)
keel/session_io.py        session <-> JSON, schema-version gate + migrations (v1 -> v3)
keel/firecrawl.py         stdlib HTTP client for the Firecrawl scrape / map API
keel/reference.py         reference intake: scraped markdown -> evidence -> slot candidates
keel/mockup.py            optional static wireframe: LLM HTML -> allowlist sanitizer
keel/templates/*.yaml     five slot templates
.streamlit/config.toml    dark theme palette
assets/style.css          the CSS the theme block can't reach (chips, conflict banner)
doctor.py                 one-live-call smoke script
tests/
```

`keel/engine.py`, `keel/render.py`, `keel/session_io.py`, `keel/firecrawl.py`,
`keel/reference.py`, and `keel/mockup.py` import and pass their tests without Streamlit
installed.

The result screen renders the spec section by section (each collapsible), shows a conflict
banner above it when `check_conflicts` found anything, and colour-codes every answer by
where it came from — the same chips appear in the sidebar progress panel, which tracks each
slot's state live through the session.

### LLM integration

All calls go through `keel/llm.complete_json(system, user, *, provider, max_tokens)`,
which returns a `(result, error)` tuple — never a bare `None`, never a raised exception.
`provider` is a `Provider(name, api_key, model, host)`; `llm.resolve_provider(secrets)`
builds one. Groq and Ollama enforce JSON natively; Anthropic via an assistant-turn prefill
— a prose reply becomes a distinguishable parse error. On failure the UI surfaces the
reason and falls back to template text.

Every answer carries a **source**: `extracted` (from your idea), `asked` (you typed it),
`reference` (confirmed from a scraped reference — see below), `llm_default` (the model
suggested it and you accepted), `template_default` (static YAML fallback, the model was
unavailable), `keel_decided` (you pressed **Decide for me** — Keel chose and recorded a
one-line reason plus a *revisit this if …* condition, both surfaced in the
*Decisions Keel made for you* section), or `skipped` (you left it deliberately open).
`SessionState.degraded` is *derived* from that state — it is true only when a slot is on
`template_default`, or the synthesis or conflict-check call actually failed; `keel_decided`
is not a degradation. A transient earlier error that did not change the finished document
(a missed extraction, one re-tried question) does not trip the banner.

The sidebar never shows a bare `n / n resolved`: it breaks the count into
`answered · Keel decided · skipped · pending`, so a spec that is mostly Keel's choices
reads as exactly that.

Call sites, all routed through `engine.capped_complete_json` (per-session cap of 14):

1. **Extract** pre-answered slots from the opening idea.
2. **Question** — generate one question + recommended default per unfilled slot (at most 8).
3. **Context defaults** — one call that fills any slot the depth or the 8-question cap left
   unasked, from the answers already given.
4. **Conflict check** (`render.check_conflicts`) — its own call, *before* synthesis, looking
   for contradictions between answers and for a capability in the original idea that no
   answer covers (premise drift). Its result is written into *Open questions* mechanically
   in Python; the synthesis model never decides whether a conflict is reported.
5. **Synthesis** — once, after every slot is filled: one call that reads the idea plus
   every answer and writes the whole document. It keeps each section consistent with the
   most specific answer, moves facts to the right section, enumerates the full interface
   surface, and expands each section to the length its content warrants. Section headings
   and their order are owned by `render.py`, never the model: the model returns section
   *bodies* keyed by name. It also returns `resolved_conflicts` — which of the
   contradictions it was handed its document silenced, and how.

After synthesis, each pre-synthesis conflict is **re-validated against the document that
was actually produced** (`render._revalidate_conflicts`). A conflict the model reported
resolving — or a premise-drift conflict the finished document now plainly covers — moves
to a *Resolved during synthesis* list (shown in the UI, never in *Open questions*). Only
conflicts that survive re-validation reach *Open questions*. Anything the model claims to
have resolved that was never raised, or resolved without reporting, is logged.

The synthesized document is then run through deterministic Python checks (no extra LLM
call): structure and ordering, empty sections, hardcoded-secret scrub (with a note added
to *Open questions*), a `default_strategy`-leak assertion, plus flags for fabricated
numeric thresholds, acceptance criteria that merely restate a constraint or non-goal, and
a missing disclaimer in a health/legal/financial spec. Any structural failure falls back
to the deterministic one-value-per-line renderer, with a visible warning and the degraded
note (`synthesis_failed` is set). That renderer stays fully functional with no LLM available.

### Caps (all constants)

| Cap | Value | Where |
|---|---|---|
| LLM calls per session | 20 | `keel/engine.py` |
| Questions asked per session | 8 | `keel/engine.py` |
| Regenerations per session | 3 | `keel/engine.py` |
| Output tokens — question | 800 | `keel/llm.py` |
| Output tokens — synthesis | 3500 | `keel/render.py` |
| Output tokens — wireframe | 4000 | `keel/mockup.py` |
| Shared-key calls per day | 500 | `app.py` |
| Opening prompt length | 500 chars | `keel/engine.py` |

Past the daily ceiling the shared key is disabled and users are directed to supply their
own via the sidebar **API key** field (used for that session only, never stored).

## Reference for structure

When you have only a rough notion ("something like Trello"), a blank slot page is hard to
answer. In the reference field on the start screen, give Keel either a **URL** or a
**product name**:

- a **URL** is scraped directly;
- a **name** ("something like Harvest") is searched first ([Firecrawl](https://firecrawl.dev)
  `/v1/search`), and you pick which of the top results is the real site — a wrong
  resolution silently poisoning every slot is the main failure mode, so the choice is
  always yours.

Either way Keel then scrapes for *structure only* — via Firecrawl, keyed from
`FIRECRAWL_API_KEY` in secrets — entities, screens and endpoints, primary flows, and the
features a first build would deliberately skip. One LLM call turns that into candidate slot
values.

You can also **upload a screenshot or a hand-drawn sketch** of a UI (PNG/JPEG/WebP, ≤ 4 MB).
A vision model reads the screens, form fields, and navigation — structure only, no copy or
branding. This needs a multimodal model: `ANTHROPIC_API_KEY` (Claude Haiku is multimodal),
a local Ollama vision model (`KEEL_PROVIDER=ollama`), or `KEEL_VISION_MODEL` naming a vision
model on your Groq key. Without one, the image field is disabled and says why — the default
Groq model (`openai/gpt-oss-120b`) is text-only.

Nothing enters the spec unconfirmed. A confirm step shows every candidate with the
reference phrases it came from; you keep, edit, or drop each one (clear the box to drop it).
Kept values fill their slot with source `reference`; the rest of the slots are asked as
normal. The source URL is recorded in *Open questions* so the provenance travels with the
document — product names, wording, and visual design are never carried across.

A reference costs one Firecrawl fetch (a thin landing page triggers up to two more
structural sub-pages, hard-capped at three) plus one LLM call, both counted against the
session caps. Non-HTTP schemes, IP literals, and loopback / internal hosts are refused
before anything is fetched. Every reference mode is optional; the blank-prompt flow is the
default and works with Firecrawl unavailable. `keel/firecrawl.py` (the HTTP client, stdlib
only) and `keel/reference.py` (evidence → candidates → slots) import and test without
Streamlit.

## Wireframe preview

The result screen has an opt-in **Generate wireframe preview** button. It makes one extra
LLM call that turns the `interfaces` and `data_model` answers into a greyscale, static HTML
sketch — one block per screen with its route and fields — so you can eyeball the structure.
It is not part of the spec and is offered as a separate `.html` download.

Keel does not execute generated code, ever. The model's HTML is untrusted: `keel/mockup.py`
rebuilds it from a tag/attribute **allowlist**, dropping `<script>`, `on*` handlers,
`<iframe>`/`<object>`/`<embed>`, `<form action>`, `<style>`, `<img>`, and any
`javascript:` / `data:` / external URL. The sanitized result is framed with
`st.components.v1.html` and wrapped in a fixed greyscale stylesheet. An injected `<script>`
in the model's response is stripped — there is a test that proves it.

## Save, edit, regenerate

The result screen offers **Download session (.json)** next to Download .md. The file holds
the whole session — prompt, template, depth, every answer with its `source`, the conflict
list, the call count, and a `schema_version` — and nothing else; Keel never stores a session
server-side. **Resume a saved session** on the start screen loads one back and drops you at
the review step (a file from a newer Keel is refused with a clear message rather than
half-read).

The review step lists every dimension that fed the spec, each editable, tagged with where
it came from (`from your idea` / `you answered` / `Keel suggested` / `template fallback` /
`skipped`). **Regenerate spec** re-runs the conflict check and synthesis from the edited
answers — no question is ever re-asked. Each regeneration costs two calls and is capped (`engine.MAX_REGENERATIONS`); the
UI shows how many are left. `keel/session_io.py` owns the format; it imports and tests
without Streamlit.

## Deploy to Streamlit Community Cloud

Point a new app at `app.py`, set one of `GROQ_API_KEY` / `OLLAMA_API_KEY` /
`ANTHROPIC_API_KEY` (and optionally `KEEL_MODEL`) in the app's **Secrets**, and deploy.
`requirements.txt` pins the dependencies. Do **not** set `KEEL_PROVIDER=ollama` on a
deployed app — there is no local daemon there and every call will fail loudly.

## Evals

```
python evals/run_evals.py                       # provider auto-picked; writes evals/RESULTS.md
KEEL_PROVIDER=ollama python evals/run_evals.py  # local model — no hosted quota, for iteration
python evals/run_evals.py --persona deterministic --limit 3   # fast partial pass
```

[`evals/cases.json`](evals/cases.json) holds ~15 hand-labelled vague prompts: for each, the
slots a competent engineer would insist on pinning down, a persona for the simulated user,
the facts the prompt already states, and any contradiction `check_conflicts` should catch.
[`evals/simulated_user.py`](evals/simulated_user.py) answers each question in character (or,
with no provider, accepts the recommendation and skips on a fixed schedule).

`run_evals.py` runs every case end to end and writes `evals/RESULTS.md`. Commit that file
when you tune a prompt, so the next run's diff shows what moved. Metrics are pure Python, no
judge model: slot coverage, mean questions asked, missed-ambiguity rate, redundancy rate,
conflict recall, and false-precision rate (numbers in the spec that trace to no answer). The
table stamps the model and persona mode that produced it.

[`evals/fixtures/`](evals/fixtures/) holds two frozen sessions whose answers contradict each
other — the hotel-website "no UI" case and the health-monitoring "offline but AI-generated"
case. They are permanent regression cases: re-run them after any edit to the question,
conflict, or synthesis prompt. `run_evals.py` exits non-zero if either stops being caught.
Rebuild them with `python evals/fixtures/build_fixtures.py` only if the slot taxonomy changes.

The committed `RESULTS.md` records which model and persona mode produced it. A small local
model scores differently from a hosted one — iterate locally, but regenerate the committed
table against the hosted model before trusting the numbers.

## Tests

```
pip install pytest
pytest
```
