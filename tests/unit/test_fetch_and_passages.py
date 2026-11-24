import asyncio

import httpx
import pytest

from research_analyst.retrieval import fetch
from research_analyst.retrieval.passages import bm25_rank, select_passages, split_passages

PAGE = """<html><head><title>t</title></head><body><nav>menu menu</nav><article>
<p>Machine learning accelerates drug discovery by predicting protein binding affinity before any wet-lab work is done, saving months of screening.</p>
<p>The weather in Lisbon was pleasant last Tuesday and the football match ended late in the evening for most fans.</p>
<p>Clinical trials for AI-designed molecules entered phase two in several companies, which analysts describe as a milestone for the sector overall.</p>
</article><footer>cookies</footer></body></html>"""


async def public(host):
    return ["93.184.216.34"]


async def private(host):
    return ["10.0.0.5"]


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


def run(coro):
    return asyncio.run(coro)


def test_fetch_extracts_main_text():
    h = lambda req: httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=PAGE)  # noqa: E731
    text = run(fetch.fetch_text("https://example.com/a", client=_client(h), resolver=public))
    assert "protein binding" in text and "menu menu" not in text


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "http://", "javascript:alert(1)"])
def test_non_http_urls_refused(url):
    assert run(fetch.is_safe_url(url, public)) is False


def test_private_ip_refused_and_no_request_made():
    hits = []
    h = lambda req: hits.append(req) or httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)  # noqa: E731
    assert run(fetch.fetch_text("http://internal.example/", client=_client(h), resolver=private)) is None
    assert hits == []


@pytest.mark.parametrize("ip", ["127.0.0.1", "169.254.169.254", "192.168.1.1", "::1", "0.0.0.0"])
def test_non_public_addresses(ip):
    assert fetch._is_public(ip) is False


def test_redirect_to_private_host_is_blocked():
    async def resolver(host):
        return ["10.0.0.1"] if host == "evil.internal" else ["93.184.216.34"]

    def h(req):
        if req.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://evil.internal/secret"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)

    assert run(fetch.fetch_text("https://example.com/", client=_client(h), resolver=resolver)) is None


def test_redirect_loop_gives_up():
    h = lambda req: httpx.Response(302, headers={"location": "https://example.com/again"})  # noqa: E731
    assert run(fetch.fetch_text("https://example.com/", client=_client(h), resolver=public)) is None


def test_non_html_and_errors_return_none():
    pdf = lambda req: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")  # noqa: E731
    err = lambda req: httpx.Response(500)  # noqa: E731
    boom = lambda req: (_ for _ in ()).throw(httpx.ConnectError("down"))  # noqa: E731
    for h in (pdf, err, boom):
        assert run(fetch.fetch_text("https://example.com/", client=_client(h), resolver=public)) is None


def test_byte_cap_truncates(monkeypatch):
    monkeypatch.setattr(fetch.settings, "fetch_max_bytes", 300)
    big = "<html><body><article><p>" + ("Machine learning drug discovery sentence here. " * 500) + "</p></article></body></html>"
    h = lambda req: httpx.Response(200, headers={"content-type": "text/html"}, text=big)  # noqa: E731
    text = run(fetch.fetch_text("https://example.com/", client=_client(h), resolver=public))
    assert text is None or len(text) < 400


def test_fetch_many_returns_only_successes():
    def h(req):
        return httpx.Response(200 if "ok" in req.url.path else 404, headers={"content-type": "text/html"}, text=PAGE)

    out = run(fetch.fetch_many(["https://e.com/ok", "https://e.com/bad", "https://e.com/ok"], client=_client(h), resolver=public))
    assert list(out) == ["https://e.com/ok"]


# ── passages ──────────────────────────────────────────────────────────────────

LONG = "\n\n".join([
    "The football match ended late and fans left the stadium quietly after the final whistle blew.",
    "Machine learning models predict protein binding affinity, which speeds up drug discovery pipelines considerably.",
    "Weather forecasts for the coming week show rain in most northern regions with mild temperatures overall.",
    "Drug discovery timelines shrank when machine learning screening replaced manual compound selection steps.",
])


def test_bm25_prefers_relevant_passages():
    ps = split_passages(LONG, target=60, max_len=200)
    ranked = bm25_rank("How does machine learning speed up drug discovery?", ps)
    best = ps[ranked[0][0]]
    assert "machine learning" in best.lower() and "football" not in best


def test_select_passages_drops_irrelevant_and_respects_cap():
    out = select_passages("machine learning drug discovery", LONG, k=2, max_chars=300, target=100)
    assert "football" not in out and "weather" not in out.lower() and len(out) <= 300 + 10


def test_select_passages_empty_when_nothing_matches():
    assert select_passages("quantum chromodynamics", LONG) == ""


def test_split_passages_bounds():
    for p in split_passages("word " * 2000, target=500, max_len=900):
        assert len(p) <= 900
