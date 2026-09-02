"""Reference intake: turn scraped evidence about a product into candidate slot
values the developer confirms before they become answers.

A reference informs slots; it never silently becomes one. The pipeline:

    raw markdown (from keel.firecrawl)
      -> extract_evidence      one capped LLM call -> Evidence
      -> evidence_to_candidates                    -> list[SlotCandidate]
      -> (the confirm UI: keep / edit / drop each)
      -> apply_candidates                          -> slots, source="reference"

No Streamlit import. ``app.py`` drives the buttons and the confirm view; this
module is plain, testable Python.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from keel import engine, firecrawl, llm
from keel.models import Evidence, SessionState, SiteCandidate, SlotCandidate, Template

MAX_REFERENCE_FETCHES = 3
MAX_SITE_CANDIDATES = 3
MAX_IMAGE_BYTES = 4 * 1024 * 1024
_IMAGE_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "webp": "image/webp"}
_THIN_PAGE_CHARS = 1500       # below this, the home page is treated as a thin landing page
_TOTAL_MATERIAL_CHARS = 14000
# The evidence JSON is small, but a reasoning model (Groq gpt-oss) spends most of
# its budget thinking before it emits — too low and the call dies mid-JSON.
_EVIDENCE_MAX_TOKENS = 2500


_IMAGE_EVIDENCE_SYSTEM = """You are Keel's reference analyst. You are given a screenshot or a \
hand-drawn sketch of a software UI. Extract ONLY structural information a developer would use \
to scope a similar build.

Return ONLY a JSON object with exactly these keys:
  "product"          - "" (an image carries no reliable product name)
  "core_entities"    - the data objects implied by what each screen shows (e.g. "Order", "Item")
  "primary_flows"    - what a user does across the screens, as short verb phrases
  "surfaces"         - each screen or view, named by what it is (e.g. "list screen", "detail
                       screen", "settings form"); include the fields on any form you can see
  "notable_features" - interactions or capabilities the layout implies (search, filters, tabs)
  "features_likely_out_of_scope_for_a_clone" - []

Rules:
- Structure only. Do NOT transcribe copy, headings, labels verbatim, brand names, colours, \
  logos, or imagery. Describe the SHAPE of each screen, not its content.
- Each list item is a short phrase. If the image does not support a key, use an empty list.
- Do not invent screens or fields that are not visible."""


_EVIDENCE_SYSTEM = """You are Keel's reference analyst. You are given text scraped from a \
product's website or documentation. Extract ONLY structural information a developer would use \
to scope a similar build.

Return ONLY a JSON object with exactly these keys:
  "product"          - the product's name, as a short string
  "core_entities"    - the main data objects it manages (e.g. "Issue", "Project", "Comment")
  "primary_flows"    - the key things a user does, as short verb phrases
  "surfaces"         - concrete UI / API surfaces: screens, pages, endpoints, CLI commands
  "notable_features" - distinguishing capabilities worth noting
  "features_likely_out_of_scope_for_a_clone" - large features a first build would deliberately skip

Rules:
- Structure only. Do NOT extract or reproduce marketing copy, taglines, slogans, colours, \
  imagery, pricing numbers, or brand language.
- Each list item is a short noun or verb phrase, not a sentence. 3-8 items per list is plenty.
- If the text does not support a key, use an empty list (or "" for "product").
- Do not invent features the text does not mention."""


# --------------------------------------------------------------------------- #
# Mode A: a product name -> a handful of candidate official sites to pick from
# --------------------------------------------------------------------------- #
_NAME_STOP = {
    "something", "like", "similar", "to", "a", "an", "the", "app", "application",
    "tool", "clone", "of", "kinda", "sort", "for", "site", "website", "service",
    "platform", "my", "own",
}
# Sites that are never the product's own structural reference.
_OFF_TOPIC_HOSTS = (
    "reddit.com", "substack.com", "medium.com", "youtube.com", "youtu.be",
    "wikipedia.org", "twitter.com", "x.com", "facebook.com", "linkedin.com",
    "news.ycombinator.com", "quora.com", "producthunt.com", "g2.com",
    "capterra.com", "crunchbase.com", "glassdoor.com", "indeed.com",
    "play.google.com", "apps.apple.com", "github.com", "gitlab.com",
    "trustpilot.com", "getapp.com", "softwareadvice.com",
)


def looks_like_url(text: str) -> bool:
    t = (text or "").strip().lower()
    if re.match(r"^[a-z][a-z0-9+.\-]*://", t):
        return True
    return bool(re.match(r"^[a-z0-9][a-z0-9-]*(\.[a-z0-9-]+)+(/|\?|#|$)", t))


def product_query(name: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9.+&-]+", name or "")
             if w.lower() not in _NAME_STOP]
    core = " ".join(words).strip() or (name or "").strip()
    return f"{core} official website"


def _rank_official(name: str, results: list[dict]) -> list[dict]:
    tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", name or "")
              if w.lower() not in _NAME_STOP and len(w) > 1]
    best: dict[str, tuple[int, dict]] = {}   # host -> (score, result), shallowest wins
    for i, r in enumerate(results):
        host = (urlparse(r["url"]).hostname or "").lower()
        host = host[4:] if host.startswith("www.") else host
        if not host or any(host == h or host.endswith("." + h) for h in _OFF_TOPIC_HOSTS):
            continue
        depth = urlparse(r["url"]).path.rstrip("/").count("/")
        name_hit = any(tok in host for tok in tokens)
        score = (5 if name_hit else 0) - depth - i * 0.1  # search order breaks ties
        if host not in best or score > best[host][0]:
            best[host] = (score, r)
    return [r for _, r in sorted(best.values(), key=lambda t: t[0], reverse=True)]


def resolve_product(
    name: str, *, api_key: str, limit: int = MAX_SITE_CANDIDATES
) -> tuple[list[SiteCandidate] | None, str | None]:
    """Search for a product's own site. Returns up to ``limit`` candidates for the
    user to choose from (possibly empty — then they proceed without a reference),
    or ``(None, reason)`` on a search failure."""
    hits, err = firecrawl.search(product_query(name), api_key=api_key, limit=limit + 4)
    if err:
        return None, err
    ranked = _rank_official(name, hits or [])[:limit]
    return [SiteCandidate(**r) for r in ranked], None


# --------------------------------------------------------------------------- #
def _looks_thin(markdown: str) -> bool:
    return len(markdown or "") < _THIN_PAGE_CHARS


def gather_material(
    raw_url: str, *, api_key: str, fetch_budget: int = MAX_REFERENCE_FETCHES
) -> tuple[dict | None, str | None]:
    """Scrape ``raw_url`` and, if it is a thin landing page, up to two structural
    sub-pages, into one markdown blob. Never exceeds ``MAX_REFERENCE_FETCHES``
    fetches. Returns ``({"material", "urls", "fetches"}, None)`` or
    ``(None, reason)``."""
    url, reason = firecrawl.validate_url(raw_url)
    if reason:
        return None, reason
    budget = max(1, min(fetch_budget, MAX_REFERENCE_FETCHES))

    home, err = firecrawl.scrape(url, api_key=api_key)
    if err:
        return None, err
    pages = [home]
    fetches = 1

    if _looks_thin(home["markdown"]) and fetches < budget:
        links, map_err = firecrawl.map_site(url, api_key=api_key)
        if not map_err and links:
            for sub in firecrawl.rank_subpages(url, links, limit=budget - fetches):
                page, sub_err = firecrawl.scrape(sub, api_key=api_key)
                fetches += 1
                if not sub_err:
                    pages.append(page)
                if fetches >= budget:
                    break

    parts: list[str] = []
    used: list[str] = []
    total = 0
    for pg in pages:
        chunk = f"# Source: {pg['url']}\n\n{pg['markdown']}"
        if total + len(chunk) > _TOTAL_MATERIAL_CHARS:
            chunk = chunk[: max(0, _TOTAL_MATERIAL_CHARS - total)]
        if chunk.strip():
            parts.append(chunk)
            used.append(pg["url"])
            total += len(chunk)
        if total >= _TOTAL_MATERIAL_CHARS:
            break
    return {"material": "\n\n---\n\n".join(parts), "urls": used, "fetches": fetches}, None


# --------------------------------------------------------------------------- #
def _coerce_evidence(d: dict) -> dict:
    def _list(x) -> list[str]:
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str) and x.strip():
            return [x.strip()]
        return []

    return {
        "product": str(d.get("product") or "").strip(),
        "core_entities": _list(d.get("core_entities")),
        "primary_flows": _list(d.get("primary_flows")),
        "surfaces": _list(d.get("surfaces")),
        "notable_features": _list(d.get("notable_features")),
        "features_likely_out_of_scope": _list(
            d.get("features_likely_out_of_scope_for_a_clone")
            or d.get("features_likely_out_of_scope")
        ),
    }


def extract_evidence(
    material: str, *, session: SessionState, provider: "llm.Provider | None"
) -> tuple[Evidence | None, str | None]:
    """One capped LLM call (counts against the per-session budget) that turns the
    scraped material into structured :class:`Evidence`."""
    result, error = engine.capped_complete_json(
        session,
        _EVIDENCE_SYSTEM,
        f"Scraped material:\n\n{material}\n\nReturn the JSON object now.",
        provider=provider,
        max_tokens=_EVIDENCE_MAX_TOKENS,
    )
    return _evidence_from_result(result, error)


def image_mime(filename: str) -> str | None:
    ext = (filename or "").rsplit(".", 1)[-1].lower()
    return _IMAGE_MIME.get(ext)


def extract_evidence_from_image(
    image_bytes: bytes, mime: str, *, session: SessionState,
    provider: "llm.Provider | None",
) -> tuple[Evidence | None, str | None]:
    """Mode C: one capped *vision* LLM call turning a UI screenshot or sketch into
    structured :class:`Evidence` (structure only — no copy, branding, or colour)."""
    if not image_bytes:
        return None, "the image is empty"
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return None, f"the image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB"
    if mime not in llm.VISION_MIME_TYPES:
        return None, "use a PNG, JPEG, or WebP image"
    result, error = engine.capped_complete_json(
        session,
        _IMAGE_EVIDENCE_SYSTEM,
        "Describe the structure of this UI as the JSON object. Structure only.",
        provider=provider,
        max_tokens=_EVIDENCE_MAX_TOKENS,
        image=(image_bytes, mime),
    )
    return _evidence_from_result(result, error)


def _evidence_from_result(result, error) -> tuple[Evidence | None, str | None]:
    if error:
        return None, error
    try:
        return Evidence.model_validate(_coerce_evidence(result or {})), None
    except Exception as exc:  # noqa: BLE001
        return None, f"could not read the evidence the model returned: {exc}"


# --------------------------------------------------------------------------- #
# evidence -> candidate slot values
#
# The last of these is the highest-value output: a reference is the best possible
# source of concrete non-goals, because it shows the developer everything they
# *could* build and lets them strike most of it out.
_CANDIDATE_PLAN: list[tuple[str, tuple[str, ...], str]] = [
    ("data_model", ("core_entities",),
     "Model these entities, each with the fields it implies: {}."),
    ("interfaces", ("surfaces", "primary_flows"),
     "Provide surfaces and flows covering: {}."),
    ("non_goals", ("features_likely_out_of_scope",),
     "A first build deliberately leaves out: {}."),
    ("done", ("notable_features",),
     "Decide whether \"done\" requires supporting: {}."),
]


def evidence_to_candidates(evidence: Evidence, template: Template) -> list[SlotCandidate]:
    out: list[SlotCandidate] = []
    for slot_name, keys, shape in _CANDIDATE_PLAN:
        if template.slot(slot_name) is None:
            continue
        items: list[str] = []
        for k in keys:
            items += list(getattr(evidence, k, []) or [])
        items = list(dict.fromkeys(i for i in items if i))  # de-dupe, keep order
        if not items:
            continue
        out.append(
            SlotCandidate(
                slot=slot_name,
                value=shape.format("; ".join(items)),
                evidence="; ".join(items),
            )
        )
    return out


def apply_candidates(
    session: SessionState, candidates: list[SlotCandidate], template: Template
) -> int:
    """Write every kept candidate into a slot with source ``reference``. A
    dropped candidate, an emptied value, or an unknown slot is ignored. Returns
    how many slots were filled."""
    from keel.models import SlotValue  # local: keep module import surface small

    applied = 0
    for c in candidates:
        if c.decision != "keep":
            continue
        value = (c.value or "").strip()
        if not value or template.slot(c.slot) is None:
            continue
        session.slots[c.slot] = SlotValue(value=value, source="reference")
        applied += 1
    return applied
