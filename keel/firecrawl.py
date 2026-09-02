"""Thin client for the Firecrawl HTTP API — reference intake (Phase 1).

No new dependency: stdlib ``urllib`` only, and the single host this module ever
connects to is ``api.firecrawl.dev``. Every function returns a ``(result, error)``
tuple like :func:`keel.llm.complete_json` — it never raises and never returns a
bare ``None`` on failure.

The SSRF surface is the *target* URL the user pastes, which Firecrawl then
fetches on our behalf. :func:`validate_url` rejects non-HTTP(S) schemes, IP
literals, and loopback / internal hostnames before the URL is ever sent.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

API_BASE = "https://api.firecrawl.dev"
USER_AGENT = "KeelReferenceBot/1.0 (+https://github.com/SpandanNagale/Keel)"

_TIMEOUT = 45
MAX_MARKDOWN_CHARS = 8000     # per page, before it is handed to the LLM
_MAP_SCAN_LIMIT = 400         # how many mapped links we bother to rank

_BLOCKED_HOST_RE = re.compile(
    r"^(localhost|.*\.local|.*\.internal|.*\.lan|"
    r"metadata\.google\.internal|instance-data.*)$",
    re.I,
)

# feature / pricing / api / docs / changelog pages carry structure; the rest is noise
_STRUCTURAL_HINTS = (
    "feature", "pricing", "docs", "documentation", "api", "developer", "developers",
    "changelog", "product", "how-it-works", "use-case", "use-cases", "guide",
    "reference", "capabilities", "platform", "overview", "workflow", "integrations",
)


# --------------------------------------------------------------------------- #
def validate_url(raw: str) -> tuple[str | None, str | None]:
    """Normalise and vet a user-supplied URL. Returns ``(url, None)`` when it is
    a plain public http(s) address, or ``(None, reason)``."""
    raw = (raw or "").strip()
    if not raw:
        return None, "enter a URL"
    if "://" not in raw:
        raw = "https://" + raw
    try:
        p = urlparse(raw)
    except ValueError as exc:
        return None, f"could not parse that URL ({exc})"

    if p.scheme not in ("http", "https"):
        return None, f"only http and https URLs are allowed (got {p.scheme or 'no'} scheme)"
    host = (p.hostname or "").strip().rstrip(".")
    if not host:
        return None, "that URL has no host"
    if _BLOCKED_HOST_RE.match(host):
        return None, "local and internal hosts are not allowed"
    try:
        ipaddress.ip_address(host)
        return None, "raw IP addresses are not allowed — use a hostname"
    except ValueError:
        pass
    if "." not in host:
        return None, "that host does not look like a public domain"
    return raw, None


# --------------------------------------------------------------------------- #
def scrape(url: str, *, api_key: str, timeout: int = _TIMEOUT) -> tuple[dict | None, str | None]:
    """Scrape one page to markdown. ``url`` is assumed to have passed
    :func:`validate_url`. Returns ``({"markdown", "metadata", "url"}, None)`` or
    ``(None, reason)``."""
    if not (api_key or "").strip():
        return None, "no Firecrawl API key configured"
    body = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "headers": {"User-Agent": USER_AGENT},
    }
    data, err = _post("/v1/scrape", body, api_key, timeout)
    if err:
        return None, err
    payload = (data or {}).get("data") or {}
    md = (payload.get("markdown") or "").strip()
    if not md:
        return None, "Firecrawl returned no readable content for that page"
    meta = payload.get("metadata") or {}
    return {
        "markdown": md[:MAX_MARKDOWN_CHARS],
        "metadata": meta,
        "url": str(meta.get("sourceURL") or meta.get("og:url") or url),
    }, None


def search(
    query: str, *, api_key: str, limit: int = 5, timeout: int = _TIMEOUT
) -> tuple[list[dict] | None, str | None]:
    """Web search. Returns ``([{"url","title","description"}, ...], None)`` or
    ``(None, reason)``. Used by reference Mode A to resolve a product name."""
    if not (api_key or "").strip():
        return None, "no Firecrawl API key configured"
    data, err = _post("/v1/search", {"query": query, "limit": max(1, min(limit, 10))},
                      api_key, timeout)
    if err:
        return None, err
    raw = (data or {}).get("data") or (data or {}).get("web") or []
    out: list[dict] = []
    for it in raw if isinstance(raw, list) else []:
        if not isinstance(it, dict):
            continue
        u = str(it.get("url") or "").strip()
        if not u:
            continue
        out.append({
            "url": u,
            "title": str(it.get("title") or "").strip(),
            "description": str(it.get("description") or it.get("snippet") or "").strip(),
        })
    return out, None


def map_site(url: str, *, api_key: str, timeout: int = _TIMEOUT) -> tuple[list[str] | None, str | None]:
    """List the URLs Firecrawl can see on the same site. Returns ``(links, None)``
    or ``(None, reason)``."""
    if not (api_key or "").strip():
        return None, "no Firecrawl API key configured"
    data, err = _post("/v1/map", {"url": url}, api_key, timeout)
    if err:
        return None, err
    links = (data or {}).get("links") or (data or {}).get("data") or []
    return [str(u) for u in links if isinstance(u, str)], None


def rank_subpages(home_url: str, links: list[str], *, limit: int) -> list[str]:
    """Pick up to ``limit`` structural sub-pages of the same site, most
    promising first. Deterministic — no network."""
    if limit <= 0:
        return []
    home = urlparse(home_url)
    home_path = home.path.rstrip("/")
    seen: set[str] = set()
    scored: list[tuple[int, str]] = []
    for link in links[:_MAP_SCAN_LIMIT]:
        p = urlparse(link)
        if p.netloc != home.netloc:
            continue
        path = p.path.rstrip("/")
        if not path or path == home_path or path in seen:
            continue
        low = path.lower()
        hits = sum(1 for h in _STRUCTURAL_HINTS if h in low)
        if hits == 0:
            continue
        seen.add(path)
        scored.append((hits * 10 - path.count("/"), link))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [u for _, u in scored[:limit]]


# --------------------------------------------------------------------------- #
def _post(path: str, body: dict, api_key: str, timeout: int) -> tuple[dict | None, str | None]:
    req = urllib.request.Request(
        API_BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = str(json.loads(exc.read()).get("error", ""))[:200]
        except Exception:  # noqa: BLE001
            pass
        if exc.code in (401, 403):
            return None, "Firecrawl rejected the API key (check FIRECRAWL_API_KEY)"
        if exc.code == 402:
            return None, "the Firecrawl account is out of credits"
        if exc.code == 429:
            return None, "Firecrawl rate limit reached — try again in a minute"
        return None, f"Firecrawl HTTP {exc.code}" + (f": {detail}" if detail else "")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        return None, f"could not reach Firecrawl ({reason})"
    except Exception as exc:  # noqa: BLE001 - every failure becomes a reason string
        return None, f"Firecrawl call failed: {type(exc).__name__}: {exc}"

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None, "Firecrawl returned a non-JSON response"
    if isinstance(data, dict) and data.get("success") is False:
        return None, f"Firecrawl: {data.get('error') or 'request failed'}"
    return data, None
