"""Keel — turn a vague project idea into an agent-ready prompt.

UI and flow control only. Every piece of logic lives in ``keel/`` and is tested
without Streamlit. Rules this file obeys:

  * All session data lives in ``st.session_state``, initialised once behind a
    guard. The LLM client is never stored there.
  * State is mutated only inside button branches — never during render. Question
    generation happens at the tail of the branch that advances to a new slot, so
    the render pass only ever *reads* ``pending_q``.
  * Every LLM failure is surfaced with ``st.warning`` and recorded on the
    session's ``degraded`` flag, which the rendered spec notes.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from keel import engine, llm, render
from keel.render import render_markdown

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


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    if "session" not in st.session_state:
        st.session_state.session = None      # keel.models.SessionState | None
        st.session_state.pending_q = None    # {slot, question, recommended, error}
        st.session_state.start_error = None  # extraction error banner
        st.session_state.final_md = None     # synthesized (or fallback) document
        st.session_state.synth_error = None  # synthesis failure reason, if any


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
        session.degraded = True
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


def _start(prompt: str, template_name: str, byok: str, byok_provider: str) -> None:
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        st.session_state.start_error = (
            f"The idea must be between 1 and {MAX_PROMPT_CHARS} characters."
        )
        return

    session = engine.start_session(prompt, template_name, created_date=date.today().isoformat())
    template = _load_template(template_name)

    provider, is_shared, blocking = _provider_for_call(byok, byok_provider)
    if blocking is not None:
        session.degraded = True
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


def _finalize(byok: str, byok_provider: str) -> None:
    """Runs once, when the session first reaches ``finished``: one synthesis LLM
    call to write the whole document, with the deterministic renderer as the
    fallback. The result is cached on ``st.session_state.final_md`` so Streamlit
    reruns never repeat the call."""
    session = st.session_state.session
    if not session or not session.finished or st.session_state.final_md is not None:
        return

    provider, is_shared, blocking = _provider_for_call(byok, byok_provider)
    if blocking is not None:
        session.degraded = True
        st.session_state.synth_error = blocking
        st.session_state.final_md = render_markdown(session)
        return

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
        session.degraded = True
        st.session_state.synth_error = error
        st.session_state.final_md = render_markdown(session)
    else:
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
    engine.accept_answer(session, pq["slot"], text, recommended=pq["recommended"])
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


def _reset() -> None:
    st.session_state.session = None
    st.session_state.pending_q = None
    st.session_state.start_error = None
    st.session_state.final_md = None
    st.session_state.synth_error = None


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

    shared, _ = _shared_provider()
    if shared is None and not byok.strip():
        st.info(
            "No API key is configured. You can still run Keel — it will fall back "
            "to generic template questions and mark the spec as generated without "
            "LLM assistance."
        )

    if st.button("Start", type="primary", key="start_btn", disabled=not prompt.strip()):
        _start(prompt, choice, byok, byok_provider)
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

    slug = engine.slugify(session.title())
    fname = f"keel-{slug}-{session.created_date}.md"

    st.download_button("Download .md", md, file_name=fname, type="primary")
    st.markdown(md)

    with st.expander("Raw markdown (select all to copy)"):
        st.code(md, language="markdown")

    if st.button("Start over", key="startover_r"):
        _reset()
        st.rerun()


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Keel", page_icon="⛵", layout="centered")
    _init_state()
    st.title("⛵ Keel")

    byok, byok_provider = _sidebar_byok()
    session = st.session_state.session

    if session is None:
        _view_intro(byok, byok_provider)
    elif not session.finished:
        _view_questions(byok, byok_provider)
    else:
        _view_result(byok, byok_provider)


if __name__ == "__main__":
    main()
