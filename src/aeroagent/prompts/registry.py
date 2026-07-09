"""Versioned system prompts for the agent loop.

Prompt content is decoupled from agent.py so that:
1. Prompt changes are tracked as discrete, named versions rather than silent edits to a hardcoded
string.
2. Eval runs (Phase 3) can record which prompt_version produced a given AgentState, making it possible
to compare quality across versions.

To introduce a new prompt variant: add a new PROMPT_V {n} constant below, register it in PROMPTS, and
optionally bump DEFAULT_PROMPT_VERSION once it's been validated against the eval suite.
"""

from __future__ import annotations


def _build_prompt_v1(tool_names: str) -> str:
    return (
        "You are a helpful AI assistant with memory. Follow this EXACT workflow:\n\n"
        "STEP 1: ALWAYS call search_memory first to check existing knowledge.\n"
        "STEP 2: If memory returned 'No relevant memory found', call web_search ONCE.\n"
        "        If memory returned useful results, skip to STEP 3.\n"
        "STEP 3: Once you have results (from memory OR web_search), you MUST finalize.\n\n"
        "CRITICAL RULES:\n"
        "- NEVER skip search_memory.\n"
        "- Never call web_search more than once.\n"
        "- Never call save_to_memory - it is handled automatically.\n"
        "- After gathering results, ALWAYS output a finalize action with your answer.\n"
        "- The 'final_answer' field is REQUIRED in the finalize action.\n\n"
        f"Available tools: {tool_names}\n\n"
        "Respond with ONLY valid JSON - no markdown, no explanation, no extra text.\n\n"
        "JSON schemas:\n"
        '{"action": "call_tool", "tool_name": "search_memory", "tool_input": {"query": "your query"}}\n'
        '{"action": "call_tool", "tool_name": "web_search", "tool_input": {"query": "your query"}}\n'
        '{"action": "finalize", "final_answer": "your complete answer here - this field is required"}\n'
    )


# Maps prompt version -> builder function. Each builder takes the comma-joined list of LLM-visible tool names
# and return the full system prompt string.
PROMPT_BUILDERS = {
    "v1": _build_prompt_v1,
}

DEFAULT_PROMPT_VERSION = "v1"


def get_prompt(tool_names: str, version: str = DEFAULT_PROMPT_VERSION) -> str:
    """Build the system prompt for a given version.

    Raises KeyError if the version isn't registered, so a typo'd version
    string fails loudly instead of silently falling back.
    """
    builder = PROMPT_BUILDERS[version]
    return builder(tool_names)
