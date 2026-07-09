"""Free, async web search tool using DuckDuckGo."""

from __future__ import annotations
import asyncio
from ddgs import DDGS

from .registry import registry


@registry.register(
    name="web_search",
    description="Search the web for real-time information, facts, or current events.",
    parameters={"query": "str (required): The search query string."},
)
async def web_search(query: str, max_results: int = 8) -> str:
    """
    Executes a DuckDuckGo search and returns formatted results.
    """
    try:

        def _sync_search() -> list[dict[str, str]]:
            with DDGS() as ddgs:
                return list(
                    ddgs.text(
                        query,
                        max_results=max_results,
                    )
                )

        # Run blocking DDGS call in a thread pool to avoid blocking the async event loop
        results = await asyncio.to_thread(_sync_search)

        if not results:
            return "No search results found."

        # Format results for the LLM to read easily
        formatted = []

        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. Title: {r.get('title', 'N/A')}\n"
                f"Snippet: {r.get('body', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search failed: {e}"
