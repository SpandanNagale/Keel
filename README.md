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

The six core slots are always asked. The three optional slots are governed by a
**Depth** selector on the start screen: `Quick` asks the six core, `Standard`
(default) adds `data_model` and `interfaces`, `Thorough` adds `error_handling`.
No session asks more than **8** questions; any slot not asked is filled from the
answers already given (one LLM call), never left blank.

Slots live in YAML under [`keel/templates/`](keel/templates/) — `default`, `cli`,
`data-pipeline`, `web-api`, `web-app`. Templates differ in question phrasing, priority,
and section mapping (`web-app` is for browser-facing apps and never lists a UI as a
non-goal; `web-api` is reserved for HTTP services). Keel picks one by keyword heuristic on
your idea, preferring `default` when the prompt is ambiguous; a dropdown lets you override.

Each slot carries a `default_text` (a concrete human-readable fallback answer) and a
`default_strategy` (an instruction used *only* inside the LLM prompt to generate a
context-aware default). `default_strategy` never appears in rendered output — there is a
test that asserts it.

## Architecture

```
app.py                 Streamlit UI and flow control only
keel/models.py         pydantic models
keel/engine.py         slot state machine, template loading, the capped LLM helper
keel/render.py         synthesis pass + deterministic fallback -> markdown
keel/llm.py            the one place an LLM is called; returns (result, error)
keel/templates/*.yaml  five slot templates
doctor.py              one-live-call smoke script
tests/
```

`keel/engine.py` and `keel/render.py` import and pass their tests without Streamlit
installed.

### LLM integration

All calls go through `keel/llm.complete_json(system, user, *, provider, max_tokens)`,
which returns a `(result, error)` tuple — never a bare `None`, never a raised exception.
`provider` is a `Provider(name, api_key, model, host)`; `llm.resolve_provider(secrets)`
builds one. Groq and Ollama enforce JSON natively; Anthropic via an assistant-turn prefill
— a prose reply becomes a distinguishable parse error. On failure the UI surfaces the
reason, falls back to template text, and sets a `degraded` flag the output notes.

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
   *bodies* keyed by name.

The synthesized document is then run through deterministic Python checks (no extra LLM
call): structure and ordering, empty sections, hardcoded-secret scrub (with a note added
to *Open questions*), a `default_strategy`-leak assertion, plus flags for fabricated
numeric thresholds, acceptance criteria that merely restate a constraint or non-goal, and
a missing disclaimer in a health/legal/financial spec. Any structural failure falls back
to the deterministic one-value-per-line renderer, with a visible warning and the degraded
note. That renderer stays fully functional with no LLM available.

### Caps (all constants)

| Cap | Value | Where |
|---|---|---|
| LLM calls per session | 14 | `keel/engine.py` |
| Questions asked per session | 8 | `keel/engine.py` |
| Output tokens — question | 800 | `keel/llm.py` |
| Output tokens — synthesis | 2000 | `keel/render.py` |
| Shared-key calls per day | 500 | `app.py` |
| Opening prompt length | 500 chars | `keel/engine.py` |

Past the daily ceiling the shared key is disabled and users are directed to supply their
own via the sidebar **API key** field (used for that session only, never stored).

## Deploy to Streamlit Community Cloud

Point a new app at `app.py`, set one of `GROQ_API_KEY` / `OLLAMA_API_KEY` /
`ANTHROPIC_API_KEY` (and optionally `KEEL_MODEL`) in the app's **Secrets**, and deploy.
`requirements.txt` pins the dependencies. Do **not** set `KEEL_PROVIDER=ollama` on a
deployed app — there is no local daemon there and every call will fail loudly.

## Tests

```
pip install pytest
pytest
```
