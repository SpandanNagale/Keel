"""Keel — turn a vague project idea into an agent-ready prompt.

UI and flow control only. Every piece of logic lives in ``keel/`` and is tested
without Streamlit. Rules this file obeys:

  * All session data lives in ``st.session_state``, initialised once behind a
    guard. The LLM client is never stored there.
  * State is mutated only inside button branches — never during render. Question
    generation happens at the tail of the branch that advances to a new slot, so
    the render pass only ever *reads* ``pending_q``.
  * Every LLM failure is surfaced with ``st.warning``. Whether it *degrades* the
    finished document is derived (``SessionState.degraded``): a slot left on
    ``template_default``, or a failed synthesis / conflict call — not every
    transient error.
"""
from __future__ import annotations

import html
from datetime import date
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from keel import engine, llm, mockup, reference, render, session_io
from keel.models import ReferenceState
from keel.render import render_markdown

_ASSETS = Path(__file__).parent / "assets"

# --------------------------------------------------------------------------- #
# Caps (all configurable here)
# --------------------------------------------------------------------------- #
MAX_PROMPT_CHARS = engine.MAX_PROMPT_CHARS          # 500
MAX_CALLS_PER_SESSION = engine.MAX_LLM_CALLS_PER_SESSION  # 10
SHARED_KEY_DAILY_CALL_CEILING = 500                 # shared-key spend guard

BYOK_PROVIDERS = list(llm.PROVIDER_ORDER)           # ["groq", "ollama-cloud", "anthropic"]

# Module-level (process-global) daily counter for the shared (app-configured) key.
_daily_usage = {"date": "", "count": 0}


def _shared_calls_left() -> int:
    today = date.today().isoformat()
    if _daily_usage["date"] != today:
        _daily_usage["date"] = today
        _daily_usage["count"] = 0
    return SHARED_KEY_DAILY_CALL_CEILING - _daily_usage["count"]


def _record_shared_call() -> None:
    _shared_calls_left()  # roll the date over if needed
    _daily_usage["count"] += 1


# --------------------------------------------------------------------------- #
# Secrets / config
# --------------------------------------------------------------------------- #
def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def _model_override() -> str | None:
    return _secret("KEEL_MODEL") or None


def _shared_provider() -> tuple["llm.Provider | None", str | None]:
    """Provider built from the app's own secrets (Groq -> Ollama -> Anthropic)."""
    available = {name: _secret(name) for name in llm.SECRET_KEYS.values()}
    return llm.resolve_provider(available, model_override=_model_override())


def _vision_provider() -> tuple["llm.Provider | None", str | None]:
    """A multimodal provider for reference Mode C (image), from secrets."""
    available = {name: _secret(name) for name in llm.SECRET_KEYS.values()}
    return llm.resolve_vision_provider(
        available, model_override=_secret("KEEL_VISION_MODEL") or None
    )


@st.cache_data(show_spinner=False)
def _load_template(name: str):
    return engine.load_template(name)


@st.cache_data(show_spinner=False)
def _style() -> str:
    try:
        return (_ASSETS / "style.css").read_text("utf-8")
    except OSError:
        return ""


def _inject_css() -> None:
    css = _style()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# source -> (human label, chip modifier). Used identically in the sidebar
# progress panel and the review step so the colour always means the same thing.
# llm_default and template_default are deliberately distinct: only the second is
# a real fallback, and only it should read as a warning colour.
_SOURCE_META = {
    "extracted": ("from your idea", "extracted"),
    "asked": ("you answered", "asked"),
    "reference": ("from a reference", "reference"),
    "llm_default": ("Keel suggested", "llm-default"),
    "template_default": ("template fallback", "template-default"),
    "keel_decided": ("Keel decided", "keel-decided"),
    "skipped": ("skipped", "skipped"),
    "pending": ("pending", "pending"),
}

# How the progress breakdown groups the seven sources. "answered" is the user's
# own input; "decided" is anything Keel chose (a suggestion they accepted, a
# static fallback, or an explicit "decide for me"); "skipped" is a deliberate
# open question.
_ANSWERED_SOURCES = ("extracted", "asked", "reference")
_DECIDED_SOURCES = ("llm_default", "template_default", "keel_decided")


def _chip(source: str) -> str:
    label, mod = _SOURCE_META.get(source, ("pending", "pending"))
    return f'<span class="keel-chip keel-chip--{mod}">{html.escape(label)}</span>'


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    if "session" not in st.session_state:
        st.session_state.session = None          # keel.models.SessionState | None
        st.session_state.pending_session = None  # session awaiting reference confirmation
        st.session_state.pending_q = None        # {slot, question, recommended, error}
        st.session_state.start_error = None      # extraction / upload / reference error banner
        st.session_state.final_md = None         # synthesized (or fallback) document
        st.session_state.synth_error = None      # synthesis failure reason, if any
        st.session_state.mockup_html = None      # sanitized wireframe HTML, once generated
        st.session_state.mockup_error = None     # wireframe generation failure reason


def _provider_for_call(byok: str, byok_provider: str) -> tuple["llm.Provider | None", bool, str | None]:
    """Return (provider, is_shared, blocking_reason).

    A user-supplied key (``byok``) always wins and is never rate-limited here.
    Otherwise the app's own provider is used, subject to the daily ceiling.
    """
    if byok.strip():
        model = _model_override() or llm.DEFAULT_MODELS.get(byok_provider, "")
        return llm.Provider(name=byok_provider, api_key=byok.strip(), model=model), False, None

    provider, reason = _shared_provider()
    if provider is None:
        return None, True, reason
    if _shared_calls_left() <= 0:
        return None, True, (
            "Keel's shared key has reached today's usage limit — add your own "
            "API key in the sidebar to keep going"
        )
    return provider, True, None


# --------------------------------------------------------------------------- #
# Flow (only ever called from button branches)
# --------------------------------------------------------------------------- #
def _generate_pending_question(byok: str, byok_provider: str) -> None:
    """Produce pending_q for the session's current slot, or clear it if the
    session is finished. Called at the tail of every advancing branch."""
    session = st.session_state.session
    template = _load_template(session.template_name)
    slot = engine.current_slot(session, template)
    if slot is None:
        st.session_state.pending_q = None
        return

    provider, is_shared, blocking = _provider_for_call(byok, byok_provider)
    if blocking is not None:
        st.session_state.pending_q = {
            "slot": slot.name,
            "question": slot.question_hint,
            "recommended": slot.default_text,
            "rationale": "",
            "revisit_if": "",
            "error": blocking,
        }
        return

    if is_shared:
        _record_shared_call()
    proposal = engine.next_question(session, template, provider=provider)
    st.session_state.pending_q = {
        "slot": slot.name,
        "question": proposal.question,
        "recommended": proposal.recommended,
        "rationale": proposal.rationale,
        "revisit_if": proposal.revisit_if,
        "error": proposal.error,
    }


def _start(
    prompt: str, template_name: str, depth: str, mode: str, byok: str, byok_provider: str
) -> None:
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        st.session_state.start_error = (
            f"The idea must be between 1 and {MAX_PROMPT_CHARS} characters."
        )
        return

    session = engine.start_session(
        prompt, template_name, created_date=date.today().isoformat(),
        depth=depth, mode=mode,
    )
    template = _load_template(template_name)

    provider, is_shared, blocking = _provider_for_call(byok, byok_provider)
    if blocking is not None:
        st.session_state.start_error = blocking
    else:
        if is_shared:
            _record_shared_call()
        err = engine.extract_prefilled(session, template, provider=provider)
        st.session_state.start_error = err

    engine.freeze_pending(session, template)
    st.session_state.session = session
    st.session_state.pending_q = None
    _generate_pending_question(byok, byok_provider)


def _begin_questions(session, byok: str, byok_provider: str, *, extract: bool) -> None:
    """Promote a fully-prepared session to the live one and produce the first
    question. Shared by the plain start path and the reference-confirm path."""
    template = _load_template(session.template_name)
    if extract:
        provider, is_shared, blocking = _provider_for_call(byok, byok_provider)
        if blocking is None:
            if is_shared:
                _record_shared_call()
            engine.extract_prefilled(session, template, provider=provider)  # skips filled slots
    engine.freeze_pending(session, template)
    st.session_state.session = session
    st.session_state.pending_session = None
    st.session_state.pending_q = None
    st.session_state.pop("_ref_edits", None)
    _generate_pending_question(byok, byok_provider)


def _fetch_reference(
    prompt: str, template_name: str, depth: str, mode: str, ref_input: str,
    byok: str, byok_provider: str,
) -> None:
    """Stage a session for reference confirmation. A URL goes straight to the
    scrape pipeline (Mode B); anything else is treated as a product name and
    searched first, so the user can pick which site to borrow from (Mode A).
    Costs one Firecrawl call plus (for a URL) one LLM call, counted against caps."""
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        st.session_state.start_error = (
            f"The idea must be between 1 and {MAX_PROMPT_CHARS} characters."
        )
        return

    fc_key = _secret("FIRECRAWL_API_KEY")
    if not fc_key.strip():
        st.session_state.start_error = (
            "Reference intake needs a Firecrawl API key (set FIRECRAWL_API_KEY in "
            "secrets). You can Start without a reference."
        )
        return

    provider, is_shared, blocking = _provider_for_call(byok, byok_provider)
    if blocking is not None:
        st.session_state.start_error = f"Reference intake needs an LLM — {blocking}"
        return

    session = engine.start_session(
        prompt, template_name, created_date=date.today().isoformat(),
        depth=depth, mode=mode,
    )
    ref_input = ref_input.strip()

    if reference.looks_like_url(ref_input):
        session.reference = ReferenceState(mode="url", query=ref_input,
                                           chosen_url=ref_input)
        err = _build_reference_from_url(session, ref_input, fc_key,
                                        byok, byok_provider)
        if err:
            st.session_state.start_error = err
            return
    else:
        sites, err = reference.resolve_product(ref_input, api_key=fc_key)
        if err:
            st.session_state.start_error = f"Could not search for that product — {err}"
            return
        if not sites:
            st.session_state.start_error = (
                f"No official site found for “{ref_input}”. Start without a reference, "
                "or paste a URL directly."
            )
            return
        session.reference = ReferenceState(mode="name", query=ref_input,
                                           site_candidates=sites)

    _clear_candidate_widgets()
    st.session_state.pending_session = session
    st.session_state.start_error = None


def _build_reference_from_url(
    session, url: str, fc_key: str, byok: str, byok_provider: str
) -> str | None:
    """Scrape ``url`` -> evidence -> candidates, onto ``session.reference``.
    Returns an error string or None. Shared by Mode B and Mode A's site pick."""
    provider, is_shared, blocking = _provider_for_call(byok, byok_provider)
    if blocking is not None:
        return f"Reference intake needs an LLM — {blocking}"

    material, err = reference.gather_material(
        url, api_key=fc_key, fetch_budget=reference.MAX_REFERENCE_FETCHES
    )
    if err:
        return f"Could not use that reference — {err}"

    if is_shared:
        _record_shared_call()
    evidence, err = reference.extract_evidence(
        material["material"], session=session, provider=provider
    )
    if err:
        return f"Could not read that reference — {err}"

    template = _load_template(session.template_name)
    candidates = reference.evidence_to_candidates(evidence, template)
    if not candidates:
        return "That reference did not yield anything concrete to borrow. Start without it."

    ref = session.reference
    ref.chosen_url = material["urls"][0] if material["urls"] else url
    ref.site_candidates = []
    ref.source_urls = material["urls"]
    ref.fetch_count = material["fetches"]
    ref.evidence = evidence
    ref.candidates = candidates
    return None


def _resolve_reference_pick(url: str, byok: str, byok_provider: str) -> None:
    """Mode A: the user chose one of the searched sites; run the scrape pipeline."""
    session = st.session_state.pending_session
    if not session or not session.reference:
        return
    err = _build_reference_from_url(
        session, url, _secret("FIRECRAWL_API_KEY"), byok, byok_provider
    )
    if err:
        st.session_state.start_error = err
        session.reference.site_candidates = []  # don't loop on the picker
        return
    st.session_state.start_error = None


def _fetch_reference_image(
    prompt: str, template_name: str, depth: str, mode: str, uploaded,
    byok: str, byok_provider: str,
) -> None:
    """Mode C: read a screenshot / sketch with a vision model and stage the
    session with candidate slot values for confirmation."""
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        st.session_state.start_error = (
            f"The idea must be between 1 and {MAX_PROMPT_CHARS} characters."
        )
        return

    mime = reference.image_mime(getattr(uploaded, "name", ""))
    if not mime:
        st.session_state.start_error = "Upload a PNG, JPEG, or WebP image."
        return
    data = uploaded.getvalue()
    if len(data) > reference.MAX_IMAGE_BYTES:
        st.session_state.start_error = (
            f"That image is over {reference.MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )
        return

    vprov, vreason = _vision_provider()
    if vprov is None:
        st.session_state.start_error = f"Image intake needs a vision model — {vreason}"
        return

    session = engine.start_session(
        prompt, template_name, created_date=date.today().isoformat(),
        depth=depth, mode=mode,
    )
    session.reference = ReferenceState(mode="image", query=getattr(uploaded, "name", "image"))
    _record_shared_call()
    evidence, err = reference.extract_evidence_from_image(
        data, mime, session=session, provider=vprov
    )
    if err:
        st.session_state.start_error = f"Could not read that image — {err}"
        return
    candidates = reference.evidence_to_candidates(evidence, _load_template(template_name))
    if not candidates:
        st.session_state.start_error = (
            "That image did not yield anything concrete to borrow. Start without it."
        )
        return
    session.reference.evidence = evidence
    session.reference.candidates = candidates
    _clear_candidate_widgets()
    st.session_state.pending_session = session
    st.session_state.start_error = None


def _confirm_reference(edits: dict[str, str], byok: str, byok_provider: str) -> None:
    session = st.session_state.pending_session
    if not session or not session.reference:
        return
    ref = session.reference
    template = _load_template(session.template_name)
    for c in ref.candidates:
        c.value = str(edits.get(c.slot, c.value)).strip()
        c.decision = "keep" if c.value else "drop"   # an emptied box drops the candidate
    ref.applied = reference.apply_candidates(session, ref.candidates, template)
    ref.confirmed = True
    _begin_questions(session, byok, byok_provider, extract=True)


def _skip_reference(byok: str, byok_provider: str) -> None:
    """Keep the staged session (same idea / template / depth) but drop the
    reference entirely."""
    session = st.session_state.pending_session
    if not session:
        return
    session.reference = None
    _begin_questions(session, byok, byok_provider, extract=True)


def _discard_pending() -> None:
    st.session_state.pending_session = None
    st.session_state.start_error = None
    _clear_candidate_widgets()


def _finalize(byok: str, byok_provider: str) -> None:
    """Runs once, when the session first reaches ``finished``: one synthesis LLM
    call to write the whole document, with the deterministic renderer as the
    fallback. The result is cached on ``st.session_state.final_md`` so Streamlit
    reruns never repeat the call."""
    session = st.session_state.session
    if not session or not session.finished or st.session_state.final_md is not None:
        return
    template = _load_template(session.template_name)

    needs_fill = any(
        session.slots.get(s.name) is None
        for s in engine.visible_slots(session, template)
    )

    provider, is_shared, blocking = _provider_for_call(byok, byok_provider)
    if blocking is not None:
        session.synthesis_failed = True
        st.session_state.synth_error = blocking
        if needs_fill:
            engine.fill_unasked_slots(session, template, provider=None)  # static
        st.session_state.final_md = render_markdown(session)
        return

    # Slots that depth or the asked-question cap left unasked are filled here,
    # from the accumulated answers — one call for all of them, never re-asked.
    if needs_fill:
        if is_shared:
            _record_shared_call()
        engine.fill_unasked_slots(session, template, provider=provider)

    # Contradiction check first, as its own call — a model already committed to
    # writing a coherent document is motivated not to notice the inputs cannot be
    # made coherent. Its result is written into "Open questions" mechanically.
    if is_shared:
        _record_shared_call()
    conflicts, conflict_error = render.check_conflicts(session, provider=provider)
    session.conflicts = conflicts
    session.conflict_check_error = conflict_error

    if is_shared:
        _record_shared_call()
    md, error = render.synthesize_spec(
        session, provider=provider, conflicts=conflicts, conflict_error=conflict_error
    )
    if error is not None:
        session.synthesis_failed = True
        st.session_state.synth_error = error
        st.session_state.final_md = render_markdown(session)
    else:
        session.synthesis_failed = False
        st.session_state.synth_error = None
        st.session_state.final_md = md


def _advance_after(byok: str, byok_provider: str) -> None:
    """Common tail for accept/skip: generate the next question, or finalize."""
    session = st.session_state.session
    if session and session.finished:
        _finalize(byok, byok_provider)
    else:
        _generate_pending_question(byok, byok_provider)


def _accept(text: str, byok: str, byok_provider: str) -> None:
    pq = st.session_state.pending_q
    session = st.session_state.session
    if not pq or not session:
        return
    # If the question call had failed, pq["recommended"] is the slot's static
    # default_text — accepting it unchanged is a genuine template fallback.
    rec_source = "template_default" if pq.get("error") else "llm_default"
    engine.accept_answer(
        session, pq["slot"], text,
        recommended=pq["recommended"], recommended_source=rec_source,
    )
    st.session_state.pending_q = None
    _advance_after(byok, byok_provider)


def _skip(byok: str, byok_provider: str) -> None:
    pq = st.session_state.pending_q
    session = st.session_state.session
    if not pq or not session:
        return
    engine.skip_slot(session, pq["slot"])
    st.session_state.pending_q = None
    _advance_after(byok, byok_provider)


def _decide(text: str, byok: str, byok_provider: str) -> None:
    """"Decide for me": Keel keeps the recommended value but records that the
    user delegated the choice, with the one-line reason and revisit condition
    that came back on the same question call (no extra LLM call)."""
    pq = st.session_state.pending_q
    session = st.session_state.session
    if not pq or not session:
        return
    value = (text or "").strip() or pq["recommended"]
    engine.decide_for_me(
        session, pq["slot"], value,
        rationale=pq.get("rationale", ""), revisit_if=pq.get("revisit_if", ""),
    )
    st.session_state.pending_q = None
    _advance_after(byok, byok_provider)


def _finish_with_defaults(byok: str, byok_provider: str) -> None:
    session = st.session_state.session
    if session:
        engine.fill_remaining_defaults(session, _load_template(session.template_name))
    st.session_state.pending_q = None
    _finalize(byok, byok_provider)


def _load_session(raw: bytes) -> None:
    """Restore a downloaded session and jump straight to the review step."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        st.session_state.start_error = "That file is not a readable text/JSON file."
        return
    session, error = session_io.loads(text)
    if error is not None:
        st.session_state.start_error = error
        return
    try:
        _load_template(session.template_name)
    except Exception:
        st.session_state.start_error = (
            f"The session names an unknown template ({session.template_name!r})."
        )
        return
    session.finished = True
    session.synthesis_failed = False  # recomputed on the next synthesis
    st.session_state.session = session
    st.session_state.pending_q = None
    st.session_state.start_error = None
    st.session_state.final_md = None
    st.session_state.synth_error = None
    st.session_state.mockup_html = None
    st.session_state.mockup_error = None


def _regenerate(edits: dict[str, str], byok: str, byok_provider: str) -> None:
    """Review step: apply the edited answers, then re-run the conflict check and
    synthesis only — questions are never re-asked."""
    session = st.session_state.session
    if not session:
        return
    ok, reason = engine.can_regenerate(session)
    if not ok:
        st.session_state.synth_error = reason
        return
    template = _load_template(session.template_name)
    engine.apply_answer_edits(session, edits, template)
    session.regen_count += 1
    session.conflicts = []
    session.resolved_conflicts = []
    session.conflict_check_error = None
    session.synthesis_failed = False
    st.session_state.final_md = None
    st.session_state.synth_error = None
    st.session_state.mockup_html = None       # the old wireframe no longer matches
    st.session_state.mockup_error = None
    _finalize(byok, byok_provider)


def _generate_mockup(byok: str, byok_provider: str) -> None:
    """Opt-in: one extra LLM call turning the interfaces/data-model answers into a
    sanitized static wireframe. Never automatic — many users only want the spec."""
    session = st.session_state.session
    if not session:
        return
    provider, is_shared, blocking = _provider_for_call(byok, byok_provider)
    if blocking is not None:
        st.session_state.mockup_error = blocking
        st.session_state.mockup_html = None
        return
    if is_shared:
        _record_shared_call()
    html_doc, error = mockup.build_mockup(session, provider=provider)
    st.session_state.mockup_html = html_doc
    st.session_state.mockup_error = error


def _clear_edit_widgets() -> None:
    for k in [k for k in st.session_state if k.startswith("edit_")]:
        del st.session_state[k]


def _clear_candidate_widgets() -> None:
    # plain state key (not a widget key) holding in-progress reference edits
    st.session_state.pop("_ref_edits", None)


def _reset() -> None:
    st.session_state.session = None
    st.session_state.pending_session = None
    st.session_state.pending_q = None
    st.session_state.start_error = None
    st.session_state.final_md = None
    st.session_state.synth_error = None
    st.session_state.mockup_html = None
    st.session_state.mockup_error = None
    _clear_edit_widgets()
    _clear_candidate_widgets()


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def _sidebar_byok() -> tuple[str, str]:
    with st.sidebar:
        st.subheader("API key")
        shared, reason = _shared_provider()
        if shared is not None:
            st.caption(
                f"Keel is running on the maintainer's **{shared.name}** key "
                f"(`{shared.model}`), with per-session and daily limits. Supply "
                "your own key below to bypass them — it is used only for this "
                "session and never stored."
            )
        else:
            st.caption(
                "No shared key is configured. Supply your own key below, or run "
                "Keel in degraded mode with template defaults only."
            )
        provider = st.selectbox(
            "Provider for your key",
            BYOK_PROVIDERS,
            format_func=lambda p: {"groq": "Groq", "ollama-cloud": "Ollama Cloud",
                                   "anthropic": "Anthropic"}.get(p, p),
            key="byok_provider",
        )
        key = st.text_input("Your API key (optional)", type="password", key="byok")
    return key, provider


def _progress_breakdown(session, slots) -> str:
    """Honest progress: never a bare ``n / n``. Splits the count into what the
    user answered, what Keel decided, what was skipped, and what is still
    pending — the "9 / 9 resolved" line over a fully-defaulted spec was the
    product lying about its own output (spec A6)."""
    answered = decided = skipped = pending = 0
    for s in slots:
        cur = session.slots.get(s.name)
        if cur is None:
            pending += 1
        elif cur.source in _ANSWERED_SOURCES:
            answered += 1
        elif cur.source in _DECIDED_SOURCES:
            decided += 1
        elif cur.source == "skipped":
            skipped += 1
    parts = [f"{answered} answered", f"{decided} Keel decided", f"{skipped} skipped"]
    if pending:
        parts.append(f"{pending} pending")
    return " · ".join(parts)


def _answered_minority(session, slots) -> bool:
    """True when the user's own answers are a minority of the resolved slots —
    the cue for a neutral "most of this is Keel's suggestions" line."""
    answered = resolved = 0
    for s in slots:
        cur = session.slots.get(s.name)
        if cur is None:
            continue
        resolved += 1
        if cur.source in _ANSWERED_SOURCES:
            answered += 1
    return resolved > 0 and answered * 2 < resolved


def _sidebar_slot_panel() -> None:
    """Live state of every slot — the slot model made visible instead of hidden."""
    session = st.session_state.session
    if session is None:
        return
    template = _load_template(session.template_name)
    slots = sorted(engine.visible_slots(session, template),
                   key=lambda s: (s.priority, s.name))
    with st.sidebar:
        st.divider()
        st.subheader("Progress")
        rows = []
        for slot in slots:
            cur = session.slots.get(slot.name)
            source = cur.source if cur else "pending"
            rows.append(
                f'<div class="keel-slot"><span>{html.escape(slot.label)}</span>'
                f"{_chip(source)}</div>"
            )
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption(_progress_breakdown(session, slots))


def _view_intro(byok: str, byok_provider: str) -> None:
    st.write(
        "Type a vague, one-line software idea. Keel asks a few targeted questions "
        "about what it leaves unstated, then hands you a structured markdown spec "
        "to paste into a coding agent."
    )

    mode_label = st.radio(
        "How would you like the questions?",
        ["Guided", "Technical"],
        index=0,
        horizontal=True,
        key="mode_choice",
        help="Guided assumes no software background: multiple-choice questions in "
        "plain language, six questions, and aggressive defaulting. Technical uses "
        "terse free-text questions and the full slot set.",
    )
    mode = {"Guided": "guided", "Technical": "technical"}[mode_label]
    if mode == "guided":
        st.caption(
            "Guided mode — pick an option, or press **Decide for me** for anything "
            "you're unsure about. Keel records what it chose and why."
        )

    prompt = st.text_area(
        "Your project idea",
        key="idea_input",
        max_chars=MAX_PROMPT_CHARS,
        height=90,
        placeholder="a website where people can book rooms at my hotel"
        if mode == "guided"
        else "scrape my bookmarks and cluster them by topic",
    )

    auto = engine.select_template(prompt) if prompt.strip() else "default"
    names = engine.list_templates()
    choice = st.selectbox(
        "Template",
        names,
        index=names.index(auto),
        format_func=lambda n: f"{n}  ·  {_load_template(n).title}",
        help="Auto-selected from your idea by keyword. Override if it guessed wrong.",
    )
    if prompt.strip() and choice == auto:
        st.caption(f"Auto-detected: **{auto}**")

    if mode == "technical":
        depth_label = st.radio(
            "Depth",
            ["Quick", "Standard", "Thorough"],
            index=1,
            horizontal=True,
            key="depth_choice",
            help="Quick asks the 6 core dimensions. Standard adds the data model and "
            "interface surface. Thorough also asks about error handling. Slots not "
            "asked are filled from context, never left blank.",
        )
        depth = {"Quick": "quick", "Standard": "standard", "Thorough": "thorough"}[depth_label]
    else:
        # Guided ignores depth entirely — it always asks its six core questions
        # and defaults the rest (spec A5).
        depth = "standard"

    fc_configured = bool(_secret("FIRECRAWL_API_KEY").strip())
    vision_provider, vision_reason = _vision_provider()
    ref_url = st.text_input(
        "Reference for structure — a URL or a product name (optional)",
        key="ref_url", placeholder="https://linear.app  ·  or  ·  something like Trello",
        disabled=not fc_configured,
        help="A URL is scraped directly; a name is searched first so you can pick the "
        "site. Keel takes structure only — entities, screens, likely non-goals — and "
        "shows every candidate to keep, edit, or drop before it enters the spec. Names, "
        "wording, and visual design are never carried across.",
    )
    ref_image = st.file_uploader(
        "…or upload a screenshot / sketch of a UI (optional)",
        type=["png", "jpg", "jpeg", "webp"], key="ref_image",
        disabled=vision_provider is None,
        help="A vision model reads the screens, fields, and navigation — structure "
        "only, no copy or branding. Max 4 MB.",
    )
    if not fc_configured and vision_provider is None:
        st.caption(
            "Reference intake needs `FIRECRAWL_API_KEY` (URL / name) or a vision "
            f"model for images — {vision_reason}."
        )
    elif vision_provider is None:
        st.caption(f"Image intake is off — {vision_reason}.")

    shared, _ = _shared_provider()
    if shared is None and not byok.strip():
        st.info(
            "No API key is configured. You can still run Keel — it will fall back "
            "to generic template questions and mark the spec as generated without "
            "LLM assistance."
        )

    use_image = ref_image is not None
    use_url = bool(ref_url.strip()) and fc_configured and not use_image
    start_label = (
        "Analyze image & continue" if use_image
        else "Fetch reference & continue" if use_url
        else "Start"
    )
    if st.button(start_label, type="primary", key="start_btn", disabled=not prompt.strip()):
        _clear_edit_widgets()
        if use_image:
            _fetch_reference_image(prompt, choice, depth, mode, ref_image, byok, byok_provider)
        elif use_url:
            _fetch_reference(prompt, choice, depth, mode, ref_url, byok, byok_provider)
        else:
            _start(prompt, choice, depth, mode, byok, byok_provider)
        st.rerun()

    if st.session_state.start_error:
        st.warning(st.session_state.start_error)

    with st.expander("About Keel"):
        st.caption(
            "Keel turns a vague idea into a structured spec for a coding agent. It "
            "lowers the barrier to writing one and makes its own choices legible, but "
            "it cannot replace judgement: this spec is a starting point and benefits "
            "from review by someone with software experience before you rely on it."
        )

    with st.expander("Resume a saved session"):
        st.caption(
            "Upload a `.json` file downloaded from a previous run to jump straight "
            "to the review step — no questions to answer again."
        )
        up = st.file_uploader("Session file", type=["json"], key="session_upload",
                              label_visibility="collapsed")
        if st.button("Load session", key="load_session_btn", disabled=up is None):
            _clear_edit_widgets()
            _load_session(up.getvalue())
            st.rerun()


def _view_reference_confirm(byok: str, byok_provider: str) -> None:
    session = st.session_state.pending_session
    ref = session.reference
    st.subheader("Reference for structure")

    # Mode A, disambiguation phase: a product name was searched, no site chosen yet.
    if ref.evidence is None and ref.site_candidates:
        st.markdown(
            f"Searched for **{html.escape(ref.query)}**. Pick the site to borrow "
            "structure from:"
        )
        for i, cand in enumerate(ref.site_candidates):
            title = cand.title or cand.url
            st.markdown(
                f"**{html.escape(title)}**  \n{html.escape(cand.url)}"
                + (f"  \n_{html.escape(cand.description[:200])}_" if cand.description else "")
            )
            if st.button("Use this site", key=f"ref_pick_{i}"):
                _resolve_reference_pick(cand.url, byok, byok_provider)
                st.rerun()
            st.divider()
        if st.button("None of these — start without a reference", key="ref_pick_none"):
            _skip_reference(byok, byok_provider)
            st.rerun()
        if st.button("Back to the idea", key="ref_pick_back"):
            _discard_pending()
            st.rerun()
        if st.session_state.start_error:
            st.warning(st.session_state.start_error)
        return

    ev = ref.evidence
    if ev and ev.product:
        st.markdown(f"Resolved to **{html.escape(ev.product)}**.")
    if ref.mode == "image":
        st.caption(
            f"Structural cues read from your uploaded image (`{html.escape(ref.query)}`). "
            "Keep, edit, or drop each candidate — nothing here enters the spec until you do."
        )
    else:
        st.caption(
            "Structural cues from "
            + ", ".join(f"[{html.escape(u)}]({u})" for u in ref.source_urls)
            + f" · {ref.fetch_count} page fetch"
            + ("es" if ref.fetch_count != 1 else "")
            + ". Keep, edit, or drop each candidate — nothing here enters the spec until you do."
        )

    st.caption("Edit any candidate, or clear a box to drop it.")
    edits: dict[str, str] = {}
    for c in ref.candidates:
        slot = _load_template(session.template_name).slot(c.slot)
        label = slot.label if slot else c.slot
        st.markdown(f"**{html.escape(label)}** &nbsp; {_chip('reference')}",
                    unsafe_allow_html=True)
        st.caption(f"From the reference: {html.escape(c.evidence)}")
        # No widget key: a keyed widget that vanishes when this view is torn down
        # trips AppTest, and the value is captured here anyway.
        seed = st.session_state.get("_ref_edits", {}).get(c.slot, c.value)
        edits[c.slot] = st.text_area(label, value=seed, height=80,
                                     label_visibility="collapsed")
        st.divider()
    st.session_state["_ref_edits"] = edits  # survives this view's own reruns

    c1, c2 = st.columns(2)
    if c1.button("Use selected & continue", type="primary", key="ref_use_btn"):
        _confirm_reference(dict(edits), byok, byok_provider)
        st.rerun()
    if c2.button("Skip the reference, keep going", key="ref_skip_btn"):
        _skip_reference(byok, byok_provider)
        st.rerun()
    if st.button("Back to the idea", key="ref_back_btn"):
        _discard_pending()
        st.rerun()

    if st.session_state.start_error:
        st.warning(st.session_state.start_error)


def _view_questions(byok: str, byok_provider: str) -> None:
    session = st.session_state.session
    pq = st.session_state.pending_q
    total = len(session.pending_slots)
    idx = min(session.current_index, total - 1)

    st.progress((session.current_index) / total if total else 1.0)
    st.markdown(f"**Question {idx + 1} of {total}**")

    if pq is None:  # defensive: should not happen
        _generate_pending_question(byok, byok_provider)
        pq = st.session_state.pending_q

    if pq.get("error"):
        st.warning(f"LLM unavailable: {pq['error']} — using template defaults")
    elif session.degraded:
        st.caption("Running in degraded mode — some answers are template defaults.")

    template = _load_template(session.template_name)
    slot = template.slot(pq["slot"])

    st.markdown(f"### {pq['question']}")
    if slot and slot.why_this_matters:
        with st.expander("Why does this matter?"):
            st.write(slot.why_this_matters)

    # Guided mode + a slot with a known answer space -> choice-first. Anything
    # else keeps the free-text box.
    guided_choices = (
        session.mode == "guided" and slot is not None and bool(slot.choices)
    )
    unsure = False
    if guided_choices:
        labels = [c.label for c in slot.choices]
        options = labels + ["Something else", "I'm not sure"]
        pick = st.radio(
            "Pick the closest option",
            options,
            key=f"choice_{pq['slot']}_{session.current_index}",
        )
        if pick in labels:
            chosen = slot.choices[labels.index(pick)]
            st.caption(f"{chosen.plain_language}  \n_Trade-off: {chosen.tradeoff}_")
            answer = chosen.as_slot_value()
        elif pick == "Something else":
            answer = st.text_area(
                "Describe it in your own words",
                value="",
                key=f"answer_{pq['slot']}_{session.current_index}",
                height=100,
            )
        else:  # "I'm not sure"
            unsure = True
            answer = ""
            st.caption(
                "Keel will choose a sensible default and record what it picked and "
                "why in the *Decisions Keel made for you* section."
            )
    else:
        answer = st.text_area(
            "Recommended answer — accept, edit, or skip",
            value=pq["recommended"],
            key=f"answer_{pq['slot']}_{session.current_index}",
            height=120,
        )

    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    if c1.button("Accept & continue", type="primary", key="accept_btn"):
        if unsure:
            _decide(answer, byok, byok_provider)
        else:
            _accept(answer, byok, byok_provider)
        st.rerun()
    if c2.button("Decide for me", key="decide_btn",
                 help="You don't know — let Keel choose and record why."):
        _decide(answer, byok, byok_provider)
        st.rerun()
    if c3.button("Skip this", key="skip_btn",
                 help="Leave this deliberately open — it goes to Open questions."):
        _skip(byok, byok_provider)
        st.rerun()
    if c4.button("Skip the rest, use defaults", key="finish_btn"):
        _finish_with_defaults(byok, byok_provider)
        st.rerun()

    st.divider()
    if st.button("Start over", key="startover_q"):
        _reset()
        st.rerun()

    calls = session.call_count
    st.caption(
        f"LLM calls this session: {calls} / {MAX_CALLS_PER_SESSION}"
        + ("" if byok.strip() else f"  ·  shared-key calls left today: {max(_shared_calls_left(), 0)}")
    )


def _conflict_banner(session) -> None:
    if not session.conflicts:
        return
    items = []
    for c in session.conflicts:
        slots = ", ".join(html.escape(s) for s in (c.get("slots") or [])) or "answers"
        conflict = html.escape(str(c.get("conflict", "")).strip())
        res = html.escape(str(c.get("suggested_resolution") or "").strip())
        feasibility = c.get("kind") == "feasibility"
        res_label = "Likely fix" if feasibility else "Suggested resolution"
        res_html = f'<br><span class="keel-res">{res_label}: {res}</span>' if res else ""
        tag = '<em>(feasibility)</em> ' if feasibility else ""
        items.append(f"<li>{tag}<strong>{slots}</strong> — {conflict}{res_html}</li>")
    n = len(session.conflicts)
    any_feas = any(c.get("kind") == "feasibility" for c in session.conflicts)
    heading = (
        f"{n} unresolved {'issue' if n == 1 else 'issues'} in your answers"
        if any_feas
        else f"{n} unresolved {'conflict' if n == 1 else 'conflicts'} in your answers"
    )
    st.markdown(
        f'<div class="keel-conflict"><h4>{heading}</h4>'
        f'<ul>{"".join(items)}</ul>'
        '<a href="#review-edit-answers">Jump to the answers to resolve them &darr;</a></div>',
        unsafe_allow_html=True,
    )


def _resolved_conflicts_panel(session) -> None:
    """Conflicts the synthesis pass reconciled. Shown so the resolution is
    visible, but kept out of Open questions — nothing is outstanding here."""
    resolved = getattr(session, "resolved_conflicts", None)
    if not resolved:
        return
    with st.expander(f"Resolved during synthesis ({len(resolved)})"):
        for c in resolved:
            slots = ", ".join(c.get("slots") or []) or "answers"
            st.markdown(
                f"- **{html.escape(slots)}** — {html.escape(str(c.get('conflict', '')).strip())}  \n"
                f"  _{html.escape(str(c.get('resolution', '')).strip())}_"
            )


def _render_spec_sections(md: str) -> None:
    """Progressive reveal: one collapsible block per section instead of one wall."""
    lines = md.splitlines()
    if lines and lines[0].startswith("# "):
        st.markdown(f"### {lines[0][2:].strip()}")
        lines = lines[1:]
    sections: list[tuple[str, list[str]]] = []
    for ln in lines:
        if ln.startswith("## "):
            sections.append((ln[3:].strip(), []))
        elif ln.strip() == "---":
            break
        elif sections:
            sections[-1][1].append(ln)
    for heading, body in sections:
        with st.expander(heading, expanded=True):
            st.markdown("\n".join(body).strip() or "_(nothing)_")


def _view_result(byok: str, byok_provider: str) -> None:
    session = st.session_state.session

    # Belt-and-suspenders: if the session is finished but finalize never ran
    # (e.g. a mid-session refresh landed here), run it now.
    if st.session_state.final_md is None:
        _finalize(byok, byok_provider)
    md = st.session_state.final_md or render_markdown(session)

    if st.session_state.synth_error:
        st.warning(
            f"Synthesis unavailable: {st.session_state.synth_error} — showing a basic spec"
        )
    elif session.degraded:
        st.warning(
            "This spec was generated without full LLM assistance. Recommended "
            "answers fell back to Keel's generic template defaults — review them "
            "before handing this to an agent."
        )

    _conflict_banner(session)
    _resolved_conflicts_panel(session)

    template = _load_template(session.template_name)
    panel_slots = sorted(engine.visible_slots(session, template),
                         key=lambda s: (s.priority, s.name))
    if _answered_minority(session, panel_slots):
        st.caption(
            "Most of this spec is Keel's suggestions rather than your decisions — "
            "worth reviewing the Decisions section before handing it to an agent."
        )

    slug = engine.slugify(session.title())
    base = f"keel-{slug}-{session.created_date}"
    d1, d2 = st.columns(2)
    d1.download_button("Download .md", md, file_name=f"{base}.md",
                       mime="text/markdown", type="primary")
    d2.download_button("Download session (.json)", session_io.dumps(session),
                       file_name=f"{base}.json", mime="application/json",
                       help="Reload this on the start screen to edit and regenerate later.")

    _render_spec_sections(md)

    st.caption("Copy the whole spec from the box below — the copy icon is top-right.")
    st.code(md, language="markdown")

    _wireframe_preview(byok, byok_provider, base)

    _review_and_regenerate(byok, byok_provider)

    if st.button("Start over", key="startover_r"):
        _reset()
        st.rerun()


def _wireframe_preview(byok: str, byok_provider: str, base: str) -> None:
    """Opt-in static wireframe of the screens in the spec. One extra LLM call;
    the model's HTML is sanitized before it is framed or offered for download."""
    session = st.session_state.session
    st.subheader("Wireframe preview")
    st.caption(
        "A greyscale, static sketch of the screens the spec implies — structure to "
        "check, not a design. One extra LLM call. Nothing is executed; the HTML is "
        "sanitized (scripts, handlers, embeds, and external URLs stripped) before it "
        "is shown or downloaded."
    )
    budget_left = session.call_count + 1 <= engine.MAX_LLM_CALLS_PER_SESSION
    label = "Regenerate wireframe" if st.session_state.mockup_html else "Generate wireframe preview"
    if st.button(label, key="mockup_btn", disabled=not budget_left):
        _generate_mockup(byok, byok_provider)
        st.rerun()
    if not budget_left and not st.session_state.mockup_html:
        st.caption("Not enough of this session's LLM-call budget left for the wireframe.")

    if st.session_state.mockup_error:
        st.warning(f"Wireframe unavailable: {st.session_state.mockup_error}")
    if st.session_state.mockup_html:
        components.html(st.session_state.mockup_html, height=600, scrolling=True)
        st.download_button(
            "Download wireframe (.html)", st.session_state.mockup_html,
            file_name=f"{base}-mockup.html", mime="text/html",
            help="A standalone, sanitized HTML wireframe — not part of the spec.",
        )


def _review_and_regenerate(byok: str, byok_provider: str) -> None:
    session = st.session_state.session
    template = _load_template(session.template_name)
    slots = sorted(engine.visible_slots(session, template),
                   key=lambda s: (s.priority, s.name))

    for slot in slots:  # seed each editor once; the user's edits then persist
        k = f"edit_{slot.name}"
        if k not in st.session_state:
            cur = session.slots.get(slot.name)
            st.session_state[k] = cur.value if cur else ""

    st.subheader("Review & edit answers", anchor="review-edit-answers")
    st.caption(
        "Every dimension that fed the spec above. Edit any of them and regenerate — "
        "Keel re-checks for contradictions and rewrites the document without asking a "
        "single question again."
    )
    st.markdown(
        " &nbsp; ".join(
            _chip(s) for s in
            ("extracted", "asked", "reference", "llm_default", "template_default",
             "keel_decided", "skipped")
        ),
        unsafe_allow_html=True,
    )
    for slot in slots:
        cur = session.slots.get(slot.name)
        source = cur.source if cur else "pending"
        st.markdown(
            f"**{html.escape(slot.label)}** &nbsp; {_chip(source)}",
            unsafe_allow_html=True,
        )
        st.text_area(slot.label, key=f"edit_{slot.name}", height=80,
                     label_visibility="collapsed")

    left = engine.regenerations_left(session)
    ok, reason = engine.can_regenerate(session)
    if st.button("Regenerate spec", key="regen_btn", type="primary", disabled=not ok):
        edits = {s.name: st.session_state.get(f"edit_{s.name}", "") for s in slots}
        _regenerate(edits, byok, byok_provider)
        st.rerun()
    st.caption(
        f"Regenerations left: {left}" if ok else f"Regeneration unavailable — {reason}."
    )


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Keel", page_icon="⛵", layout="centered")
    _init_state()
    _inject_css()
    st.title("⛵ Keel")

    byok, byok_provider = _sidebar_byok()
    _sidebar_slot_panel()
    session = st.session_state.session

    if st.session_state.get("pending_session") is not None:
        _view_reference_confirm(byok, byok_provider)
    elif session is None:
        _view_intro(byok, byok_provider)
    elif not session.finished:
        _view_questions(byok, byok_provider)
    else:
        _view_result(byok, byok_provider)


if __name__ == "__main__":
    main()
