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

from keel import engine, llm, reference, render, session_io
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
    "skipped": ("skipped", "skipped"),
    "pending": ("pending", "pending"),
}


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
            "error": blocking,
        }
        return

    if is_shared:
        _record_shared_call()
    question, recommended, error = engine.next_question(session, template, provider=provider)
    st.session_state.pending_q = {
        "slot": slot.name,
        "question": question,
        "recommended": recommended,
        "error": error,
    }


def _start(
    prompt: str, template_name: str, depth: str, byok: str, byok_provider: str
) -> None:
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        st.session_state.start_error = (
            f"The idea must be between 1 and {MAX_PROMPT_CHARS} characters."
        )
        return

    session = engine.start_session(
        prompt, template_name, created_date=date.today().isoformat(), depth=depth
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
    prompt: str, template_name: str, depth: str, ref_url: str,
    byok: str, byok_provider: str,
) -> None:
    """Mode B: scrape a pasted URL, extract structural evidence, and stage a
    session with candidate slot values for the user to confirm. Costs one
    Firecrawl fetch (up to 3) plus one LLM call, all counted against the caps."""
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
        prompt, template_name, created_date=date.today().isoformat(), depth=depth
    )
    material, err = reference.gather_material(
        ref_url, api_key=fc_key, fetch_budget=reference.MAX_REFERENCE_FETCHES
    )
    if err:
        st.session_state.start_error = f"Could not use that reference — {err}"
        return

    if is_shared:
        _record_shared_call()
    evidence, err = reference.extract_evidence(
        material["material"], session=session, provider=provider
    )
    if err:
        st.session_state.start_error = f"Could not read that reference — {err}"
        return

    template = _load_template(template_name)
    candidates = reference.evidence_to_candidates(evidence, template)
    if not candidates:
        st.session_state.start_error = (
            "That reference did not yield anything concrete to borrow. Start without it."
        )
        return

    session.reference = ReferenceState(
        mode="url", query=ref_url.strip(), source_urls=material["urls"],
        fetch_count=material["fetches"], evidence=evidence, candidates=candidates,
    )
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

    needs_fill = any(session.slots.get(s.name) is None for s in template.slots)

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
    _finalize(byok, byok_provider)


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


def _sidebar_slot_panel() -> None:
    """Live state of every slot — the slot model made visible instead of hidden."""
    session = st.session_state.session
    if session is None:
        return
    template = _load_template(session.template_name)
    slots = sorted(template.slots, key=lambda s: (s.priority, s.name))
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
        done = sum(1 for s in slots if session.slots.get(s.name))
        st.caption(f"{done} / {len(slots)} dimensions resolved")


def _view_intro(byok: str, byok_provider: str) -> None:
    st.write(
        "Type a vague, one-line software idea. Keel asks a few targeted questions "
        "about what it leaves unstated, then hands you a structured markdown spec "
        "to paste into a coding agent."
    )
    prompt = st.text_area(
        "Your project idea",
        key="idea_input",
        max_chars=MAX_PROMPT_CHARS,
        height=90,
        placeholder="scrape my bookmarks and cluster them by topic",
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

    fc_configured = bool(_secret("FIRECRAWL_API_KEY").strip())
    ref_url = st.text_input(
        "Reference URL for structure (optional)",
        key="ref_url", placeholder="https://linear.app — a product to borrow structure from",
        disabled=not fc_configured,
        help="Keel scrapes it for structure only — entities, screens, likely non-goals — "
        "and shows every candidate to keep, edit, or drop before it enters the spec. "
        "Names, wording, and visual design are never carried across.",
    )
    if not fc_configured:
        st.caption("Set `FIRECRAWL_API_KEY` in secrets to enable reference intake.")

    shared, _ = _shared_provider()
    if shared is None and not byok.strip():
        st.info(
            "No API key is configured. You can still run Keel — it will fall back "
            "to generic template questions and mark the spec as generated without "
            "LLM assistance."
        )

    start_label = "Fetch reference & continue" if (ref_url.strip() and fc_configured) else "Start"
    if st.button(start_label, type="primary", key="start_btn", disabled=not prompt.strip()):
        _clear_edit_widgets()
        if ref_url.strip() and fc_configured:
            _fetch_reference(prompt, choice, depth, ref_url, byok, byok_provider)
        else:
            _start(prompt, choice, depth, byok, byok_provider)
        st.rerun()

    if st.session_state.start_error:
        st.warning(st.session_state.start_error)

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
    ev = ref.evidence
    if ev and ev.product:
        st.markdown(f"Resolved to **{html.escape(ev.product)}**.")
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

    st.markdown(f"### {pq['question']}")
    answer = st.text_area(
        "Recommended answer — accept, edit, or skip",
        value=pq["recommended"],
        key=f"answer_{pq['slot']}_{session.current_index}",
        height=120,
    )

    c1, c2, c3 = st.columns([2, 1, 2])
    if c1.button("Accept & continue", type="primary", key="accept_btn"):
        _accept(answer, byok, byok_provider)
        st.rerun()
    if c2.button("Skip this", key="skip_btn"):
        _skip(byok, byok_provider)
        st.rerun()
    if c3.button("Skip the rest, use defaults", key="finish_btn"):
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
        res_html = f'<br><span class="keel-res">Suggested resolution: {res}</span>' if res else ""
        items.append(f"<li><strong>{slots}</strong> — {conflict}{res_html}</li>")
    n = len(session.conflicts)
    st.markdown(
        f'<div class="keel-conflict"><h4>{n} unresolved '
        f'{"conflict" if n == 1 else "conflicts"} in your answers</h4>'
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

    _review_and_regenerate(byok, byok_provider)

    if st.button("Start over", key="startover_r"):
        _reset()
        st.rerun()


def _review_and_regenerate(byok: str, byok_provider: str) -> None:
    session = st.session_state.session
    template = _load_template(session.template_name)
    slots = sorted(template.slots, key=lambda s: (s.priority, s.name))

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
            ("extracted", "asked", "reference", "llm_default", "template_default", "skipped")
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
