"""Phase 2: a static, sanitized HTML wireframe of the screens the spec describes.

This is a preview artifact, never part of the spec, and **no generated code is
executed on the host** — Keel does not run generated code, full stop. The model's
HTML is untrusted input: :func:`sanitize_html` rebuilds it from a tag/attribute
*allowlist*, dropping ``<script>``, event handlers, framed/embedded content,
external resource references, and ``javascript:`` / ``data:`` URLs. The result is
rendered inside a sandboxed iframe (``st.components.v1.html``) and offered as a
separate ``.html`` download.

No Streamlit import here — ``app.py`` wires the button and the frame.
"""
from __future__ import annotations

import html as _html
import re
from html.parser import HTMLParser

from keel import engine
from keel.models import SessionState

MOCKUP_MAX_TOKENS = 4000

# Tags kept (rebuilt with filtered attributes). Everything else is unwrapped
# (children kept) unless it is in _DROP_TREE, whose whole subtree is discarded.
_ALLOWED_TAGS = frozenset({
    "div", "section", "header", "footer", "nav", "main", "article", "aside",
    "figure", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6", "p", "span",
    "strong", "em", "b", "i", "u", "small", "code", "pre", "blockquote", "hr", "br",
    "ul", "ol", "li", "dl", "dt", "dd",
    "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup", "col",
    "form", "label", "fieldset", "legend", "input", "textarea", "select", "option",
    "optgroup", "button", "a",
})
# Whole subtree discarded. Structural wrappers (html/head/body) and any unlisted
# tag are instead *unwrapped* — the tag goes, its children stay. All styling comes
# from the fixed shell below, so <style> is dropped outright.
_DROP_TREE = frozenset({
    "script", "style", "iframe", "object", "embed", "applet", "noscript",
    "template", "link", "meta", "base", "frame", "frameset", "svg", "math", "img",
    "picture", "source", "video", "audio", "canvas", "map", "area", "title",
})
_ALLOWED_ATTRS = frozenset({
    "class", "id", "style", "type", "placeholder", "disabled", "readonly",
    "checked", "selected", "multiple", "rows", "cols", "colspan", "rowspan",
    "for", "name", "value", "role", "scope", "headers", "abbr", "align", "valign",
    "width", "height", "aria-label", "aria-hidden", "aria-describedby", "title",
})
# Void elements: no end tag, so a dropped one must not open a "skip subtree".
_VOID = frozenset({"img", "source", "area", "base", "link", "meta", "embed",
                   "frame", "param", "track", "wbr", "col", "br", "hr", "input"})
_URLISH_ATTRS = frozenset({
    "href", "src", "srcset", "action", "formaction", "background", "data",
    "codebase", "cite", "longdesc", "usemap", "poster", "ping", "dynsrc",
    "lowsrc", "xlink:href", "profile", "manifest", "archive",
})
_BAD_URL_SCHEME = re.compile(r"(?i)(?:javascript|vbscript|data|file|about|blob)\s*:")
_CSS_BAD = re.compile(
    r"(?is)@import|url\s*\(|expression\s*\(|javascript\s*:|vbscript\s*:|"
    r"-moz-binding|behaviou?r\s*:|<\s*/?\s*script"
)

_WIREFRAME_SHELL = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 20px; background: #f4f4f4; color: #222;
         font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
  .keel-wf-note {{ background: #e6e6e6; border: 1px solid #bbb; color: #444;
    font-size: 12px; padding: 6px 10px; margin-bottom: 16px; border-radius: 4px; }}
  h1, h2, h3, h4 {{ margin: 0 0 8px; font-weight: 600; color: #111; }}
  section, .screen {{ background: #fff; border: 1px solid #ccc; border-radius: 6px;
    padding: 16px; margin: 0 0 20px; }}
  input, textarea, select, button {{ font: inherit; border: 1px solid #bbb;
    background: #fafafa; color: #333; padding: 6px 8px; border-radius: 4px; }}
  button {{ background: #ddd; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
  a {{ color: #333; text-decoration: underline; }}
  img {{ display: none; }}
</style></head><body>
<div class="keel-wf-note">Wireframe preview — structure only. Not a visual design; \
no styling, copy, or branding is implied.</div>
{body}
</body></html>"""


class _Sanitizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._drop_depth = 0

    # -- helpers ----------------------------------------------------------- #
    def _clean_attrs(self, tag: str, attrs: list[tuple[str, str | None]]) -> str:
        kept: list[str] = []
        for name, value in attrs:
            name = (name or "").lower().strip()
            value = value or ""
            if name.startswith("on"):
                continue
            if name in _URLISH_ATTRS:
                if tag == "a" and name == "href":
                    kept.append('href="#"')   # keep it clickable-looking, go nowhere
                continue
            if not (name in _ALLOWED_ATTRS or name.startswith("aria-") or name.startswith("data-")):
                continue
            if _BAD_URL_SCHEME.search(value.replace("\x00", "").replace("\t", "").replace("\n", "")):
                continue
            if name == "style":
                if _CSS_BAD.search(value):
                    continue
            kept.append(f'{name}="{_html.escape(value, quote=True)}"')
        return (" " + " ".join(kept)) if kept else ""

    # -- HTMLParser hooks ------------------------------------------------- #
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._drop_depth or tag in _DROP_TREE:
            if tag in _DROP_TREE and tag not in _VOID:
                self._drop_depth += 1
            return
        if tag in _ALLOWED_TAGS:
            self.out.append(f"<{tag}{self._clean_attrs(tag, attrs)}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if self._drop_depth or tag in _DROP_TREE:
            return
        if tag in _ALLOWED_TAGS:
            self.out.append(f"<{tag}{self._clean_attrs(tag, attrs)}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _DROP_TREE:
            if tag not in _VOID and self._drop_depth:
                self._drop_depth -= 1
            return
        if self._drop_depth:
            return
        if tag in _ALLOWED_TAGS:
            self.out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        if self._drop_depth:
            return
        self.out.append(_html.escape(data, quote=False))

    def handle_comment(self, data: str) -> None:  # drop all comments
        return


def sanitize_html(raw: str) -> str:
    """Rebuild ``raw`` from the tag/attribute allowlist. Never raises."""
    p = _Sanitizer()
    try:
        p.feed(raw or "")
        p.close()
    except Exception:  # noqa: BLE001 - a broken parse must not crash the app
        return ""
    body = "".join(p.out).strip()
    # belt-and-suspenders: no stray "<script" survives even mangled
    body = re.sub(r"(?is)<\s*/?\s*script[^>]*>", "", body)
    return body


def _wrap(body: str) -> str:
    return _WIREFRAME_SHELL.format(body=body or "<p>The model produced no usable structure.</p>")


# --------------------------------------------------------------------------- #
_MOCKUP_SYSTEM = """You produce a STATIC HTML WIREFRAME of the screens a software spec \
implies. Greyscale boxes only — no colour, no imagery, no logos, no brand or product names, \
no real copy. The goal is structure the developer can check, not a design.

Return ONLY a JSON object: {"html": "<the wireframe as one self-contained HTML fragment>"}.

Rules for the HTML:
- One <section class="screen"> per screen / route / endpoint named in the answers. Start each \
  with an <h3> giving its route or name, then the fields, controls, or columns it shows.
- Use only: section, div, header, nav, h1-h4, p, span, ul/ol/li, table/tr/th/td, form, label, \
  input, textarea, select, option, button, strong, small, hr. Placeholder text in inputs is \
  fine; real content is not.
- Show navigation between screens as a plain <nav> list of links (href="#").
- NO <script>, NO event handlers (onclick=...), NO <style>, NO <img>, NO external URLs, NO \
  <iframe>. Inline nothing that loads a resource.
- Keep it compact. A dozen short sections at most."""


def build_mockup(
    session: SessionState, *, provider: "object | None"
) -> tuple[str | None, str | None]:
    """One capped LLM call producing a sanitized, self-contained wireframe HTML
    document. Returns ``(html, None)`` or ``(None, reason)``. The output is always
    run through :func:`sanitize_html` before it is returned."""
    dm = session.slots.get("data_model")
    itf = session.slots.get("interfaces")
    parts = [f'Project idea: "{session.original_prompt}"']
    if itf and itf.value.strip():
        parts.append(f"Interfaces / screens:\n{itf.value.strip()}")
    if dm and dm.value.strip():
        parts.append(f"Data model:\n{dm.value.strip()}")
    parts.append("Produce the wireframe JSON now.")
    user = "\n\n".join(parts)

    result, error = engine.capped_complete_json(
        session, _MOCKUP_SYSTEM, user, provider=provider, max_tokens=MOCKUP_MAX_TOKENS
    )
    if error is not None:
        return None, error
    raw = str((result or {}).get("html") or "").strip()
    if not raw:
        return None, "the model returned no wireframe HTML"
    return _wrap(sanitize_html(raw)), None
