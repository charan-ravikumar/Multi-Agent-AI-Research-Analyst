import asyncio

import httpx
import pytest

from research_analyst.config import settings
from research_analyst.retrieval import arxiv as ax
from tests.helpers import make_pdf

ABS = "https://arxiv.org/abs/2401.00001v1"


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def pdf_response(data: bytes, ctype="application/pdf"):
    return httpx.Response(200, headers={"content-type": ctype}, content=data)


@pytest.fixture(autouse=True)
def _no_spacing(monkeypatch):
    monkeypatch.setattr(settings, "arxiv_min_interval_seconds", 0.0)
    ax._pdf_cache.clear()


def run(coro):
    return asyncio.run(coro)


def test_abs_to_pdf_url_only_accepts_arxiv_abs_links():
    assert ax.abs_to_pdf_url(ABS) == "https://arxiv.org/pdf/2401.00001v1"
    assert ax.abs_to_pdf_url("https://arxiv.org/abs/quant-ph/0301001v2") == "https://arxiv.org/pdf/quant-ph/0301001v2"
    for bad in ("https://evil.example/abs/1", "http://arxiv.org/abs/1", "https://arxiv.org/pdf/1", "file:///etc/passwd", ""):
        assert ax.abs_to_pdf_url(bad) is None


def test_extracts_text_from_a_real_pdf():
    data = make_pdf(["Surface codes suppress logical errors", "Threshold depends on the noise model"])
    text = run(ax.fetch_pdf_text(ABS, client=client(lambda r: pdf_response(data))))
    assert "Surface codes suppress logical errors" in text and "Threshold depends" in text


def test_only_the_first_pages_are_parsed(monkeypatch):
    monkeypatch.setattr(settings, "arxiv_pdf_max_pages", 2)
    data = make_pdf([f"page number {i} content" for i in range(1, 6)])
    text = run(ax.fetch_pdf_text(ABS, client=client(lambda r: pdf_response(data))))
    assert "page number 2" in text and "page number 3" not in text


def test_oversized_pdf_is_skipped(monkeypatch):
    monkeypatch.setattr(settings, "arxiv_pdf_max_bytes", 300)
    data = make_pdf(["x " * 500])
    assert run(ax.fetch_pdf_text(ABS, client=client(lambda r: pdf_response(data)))) is None


@pytest.mark.parametrize("resp", [
    lambda r: httpx.Response(404),
    lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>not a pdf</html>"),
    lambda r: pdf_response(b"%PDF-1.4 this is garbage and not parseable"),
    lambda r: (_ for _ in ()).throw(httpx.ConnectError("down")),
])
def test_failures_return_none_never_raise(resp):
    assert run(ax.fetch_pdf_text(ABS, client=client(resp))) is None


def test_redirect_off_arxiv_is_refused():
    def handler(req):
        if req.url.host == "arxiv.org":
            return httpx.Response(302, headers={"location": "https://evil.example/paper.pdf"})
        return pdf_response(make_pdf(["secret"]))

    assert run(ax.fetch_pdf_text(ABS, client=client(handler))) is None


def test_non_arxiv_url_makes_no_request():
    hits = []
    assert run(ax.fetch_pdf_text("https://evil.example/abs/1", client=client(lambda r: hits.append(r) or pdf_response(b"")))) is None
    assert hits == []


def test_429_starts_the_shared_cooldown_and_results_are_cached(monkeypatch):
    ax._cooldown_until = 0.0
    n = {"n": 0}

    def handler(req):
        n["n"] += 1
        return httpx.Response(429)

    c = client(handler)
    assert run(ax.fetch_pdf_text(ABS, client=c)) is None
    assert run(ax.fetch_pdf_text(ABS, client=c)) is None
    assert n["n"] == 1                                              # second call skipped during cooldown
    ax._cooldown_until = 0.0
    n["n"] = 0
    good = client(lambda r: (n.__setitem__("n", n["n"] + 1), pdf_response(make_pdf(["cached text"])))[1])
    assert "cached text" in run(ax.fetch_pdf_text(ABS, client=good))
    assert "cached text" in run(ax.fetch_pdf_text(ABS, client=good))
    assert n["n"] == 1


def test_hard_wrapped_pdf_lines_are_reflowed_into_sentences():
    from research_analyst.retrieval.arxiv import _extract_pdf_text
    data = make_pdf(["A self-attention layer connects all positions\nwith a constant number of sequentially exe-\ncuted operations."])
    text = _extract_pdf_text(data, 5)
    assert "\n" not in text and "sequentially executed operations" in text
