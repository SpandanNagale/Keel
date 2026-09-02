"""Phase 1: the Firecrawl client. Network is always stubbed at ``_post``; these
tests are about URL vetting, response shaping, and sub-page ranking."""
from __future__ import annotations

import pytest

from keel import firecrawl


# --------------------------------------------------------------------------- #
# validate_url
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, expected", [
    ("https://linear.app", "https://linear.app"),
    ("linear.app/features", "https://linear.app/features"),
    ("http://example.com/x", "http://example.com/x"),
])
def test_validate_url_accepts_and_normalises_public_https(raw, expected):
    url, err = firecrawl.validate_url(raw)
    assert err is None and url == expected


@pytest.mark.parametrize("raw, needle", [
    ("ftp://example.com", "http and https"),
    ("file:///etc/passwd", "http and https"),
    ("http://localhost:8000", "local and internal"),
    ("https://foo.local", "local and internal"),
    ("https://169.254.169.254/latest/meta-data/", "raw IP addresses"),
    ("http://127.0.0.1", "raw IP addresses"),
    ("https://[::1]/", "raw IP addresses"),
    ("https://10.0.0.5", "raw IP addresses"),
    ("https://intranet", "public domain"),
    ("", "enter a URL"),
])
def test_validate_url_rejects_non_public_targets(raw, needle):
    url, err = firecrawl.validate_url(raw)
    assert url is None and needle in err


# --------------------------------------------------------------------------- #
# scrape / map_site response shaping
# --------------------------------------------------------------------------- #
def _stub_post(monkeypatch, result):
    calls = []

    def fake(path, body, api_key, timeout):
        calls.append((path, body))
        return result if isinstance(result, tuple) else (result, None)

    monkeypatch.setattr(firecrawl, "_post", fake)
    return calls


def test_scrape_returns_trimmed_markdown_and_source_url(monkeypatch):
    _stub_post(monkeypatch, {"data": {
        "markdown": "x" * (firecrawl.MAX_MARKDOWN_CHARS + 500),
        "metadata": {"sourceURL": "https://ex.com/real"},
    }})
    page, err = firecrawl.scrape("https://ex.com", api_key="fc-k")
    assert err is None
    assert len(page["markdown"]) == firecrawl.MAX_MARKDOWN_CHARS
    assert page["url"] == "https://ex.com/real"


def test_scrape_empty_body_is_an_error(monkeypatch):
    _stub_post(monkeypatch, {"data": {"markdown": "   ", "metadata": {}}})
    page, err = firecrawl.scrape("https://ex.com", api_key="fc-k")
    assert page is None and "no readable content" in err


def test_scrape_without_a_key_never_calls_the_api(monkeypatch):
    calls = _stub_post(monkeypatch, {"data": {"markdown": "hi", "metadata": {}}})
    page, err = firecrawl.scrape("https://ex.com", api_key="")
    assert page is None and "no Firecrawl API key" in err
    assert calls == []


def test_search_returns_url_title_description_triples(monkeypatch):
    _stub_post(monkeypatch, {"data": [
        {"url": "https://a.com", "title": "A", "description": "the a"},
        {"url": "https://b.com", "snippet": "the b"},
        {"title": "no url"},
        "junk",
    ]})
    hits, err = firecrawl.search("a product", api_key="fc-k")
    assert err is None
    assert hits == [
        {"url": "https://a.com", "title": "A", "description": "the a"},
        {"url": "https://b.com", "title": "", "description": "the b"},
    ]


def test_search_without_a_key_is_an_error(monkeypatch):
    calls = _stub_post(monkeypatch, {"data": []})
    hits, err = firecrawl.search("x", api_key="")
    assert hits is None and "no Firecrawl API key" in err and calls == []


def test_map_site_normalises_the_link_list(monkeypatch):
    _stub_post(monkeypatch, {"links": ["https://ex.com", "https://ex.com/pricing", 42, None]})
    links, err = firecrawl.map_site("https://ex.com", api_key="fc-k")
    assert err is None and links == ["https://ex.com", "https://ex.com/pricing"]


def test_post_maps_http_errors_to_reasons(monkeypatch):
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    data, err = firecrawl._post("/v1/scrape", {}, "bad", 5)
    assert data is None and "rejected the API key" in err


# --------------------------------------------------------------------------- #
# rank_subpages
# --------------------------------------------------------------------------- #
def test_rank_subpages_prefers_structural_same_domain_pages():
    links = [
        "https://site.com/",                       # home, skipped
        "https://site.com/features/boards",        # structural, depth 2
        "https://site.com/pricing",                # structural, depth 1
        "https://site.com/blog/2024/a-post",       # not structural
        "https://other.com/api",                   # off-domain
        "https://site.com/changelog",              # structural
    ]
    picks = firecrawl.rank_subpages("https://site.com/", links, limit=2)
    assert picks == ["https://site.com/pricing", "https://site.com/changelog"] or \
           picks == ["https://site.com/changelog", "https://site.com/pricing"]
    assert all("other.com" not in u for u in picks)
    assert firecrawl.rank_subpages("https://site.com/", links, limit=0) == []
