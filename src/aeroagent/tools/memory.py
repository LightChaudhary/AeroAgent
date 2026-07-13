"""Memory tools for AeroAgent - save_to_memory and search_memory."""

from __future__ import annotations

import asyncio

from ..memory.memory import memory
from .registry import registry


@registry.register(
    name="search_memory",
    description="Search agent memory for relevant context before calling web_search. Always call this first.",
    parameters={"query": "str (required): The topic or question to search memory for."},
)
async def search_memory(query: str) -> str:
    """
    Search persistent memory for relevant context.
    Returns formatted results ready for LLM consumption.
    """
    result = await asyncio.to_thread(memory.format_recall, query=query, top_k=3)
    return result


@registry.register(
    name="save_to_memory",
    description="Save important information to persistent memory for future recall.",
    parameters={"text": "str (required): The information to save to memory."},
)
async def save_to_memory(text: str) -> str:
    """
    Save a piece of text to persistent memory.
    Returns confirmation with the assigned memory ID.
    """
    if not text or not text.strip():
        return "Nothing saved - empty text provided."

    doc_id = await asyncio.to_thread(memory.remember, text=text)
    return f"Saved to memory (id={doc_id}): {text[:80]}..."
