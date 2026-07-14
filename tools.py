"""
tools.py
========

Standalone tool implementations for the assistant that don't need
per-guild state. Guild-bound tools (music controls) live on the agent in
agent.py and delegate to the session's MusicPlayer.
"""

from __future__ import annotations

import asyncio

from ddgs import DDGS

_SNIPPET_CHARS = 250
_MAX_RESULT_CHARS = 2000  # keep tool results small for local-model context


async def web_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo text search, formatted as a compact result list."""

    def _search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        results = await asyncio.to_thread(_search)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return f"No results found for {query!r}."

    lines = []
    for r in results:
        title = r.get("title", "untitled")
        body = (r.get("body") or "").strip()
        if len(body) > _SNIPPET_CHARS:
            body = body[:_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
        href = r.get("href", "")
        lines.append(f"- {title}: {body} ({href})")

    text = "\n".join(lines)
    return text[:_MAX_RESULT_CHARS]
