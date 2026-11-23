"""
tests/test_web_search_manual.py

Standalone test for tools/web_search.py.
Run from the project root:

    .venv/Scripts/python.exe tests/test_web_search_manual.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.web_search import search_web


async def main() -> None:
    query = "recent advances in AI drug discovery"
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    results = await search_web(query, max_results=5)

    if not results:
        print("No results returned (zero results or search failed).")
        return

    print(f"Returned {len(results)} result(s):\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.title}")
        print(f"     {r.url}")
        print(f"     Snippet: {r.snippet[:120]}..." if len(r.snippet) > 120 else f"     Snippet: {r.snippet}")
        print()

    assert all(isinstance(r.url, str) and r.url.startswith("http") for r in results), \
        "All results must have a valid http URL"
    assert all(isinstance(r.title, str) and r.title for r in results), \
        "All results must have a non-empty title"

    print(f"All assertions passed.")


if __name__ == "__main__":
    asyncio.run(main())
