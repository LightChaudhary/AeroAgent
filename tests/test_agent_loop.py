"""Tests for agent loop control flow: max steps, tool blocking, synthesis fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.aeroagent.agent import Agent
from src.aeroagent.llm import LLMClient


@pytest.fixture
def mock_llm() -> LLMClient:
    client = LLMClient()
    client.chat_completion = AsyncMock()
    client.chat_completion_text = AsyncMock()
    return client


def _json_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _text_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _search_memory_tool(query: str) -> str:
    return "No relevant memory found."


def _web_search_tool(query: str) -> str:
    return f"1. Result for: {query}\nSnippet: test\nURL: http://example.com"


# --- Max steps / synthesis fallback ---


@pytest.mark.asyncio
async def test_agent_completes_after_memory_and_search(mock_llm: LLMClient):
    """Agent should call search_memory, then web_search, then finalize."""
    mock_llm.chat_completion.side_effect = [
        # Step 1: call search_memory
        _json_response(
            '{"action": "call_tool", "tool_name": "search_memory", "tool_input": {"query": "test"}}'
        ),
        # Step 2: call web_search (memory had no hit)
        _json_response(
            '{"action": "call_tool", "tool_name": "web_search", "tool_input": {"query": "test"}}'
        ),
        # Step 3: finalize
        _json_response('{"action": "finalize", "final_answer": "Here is the answer."}'),
    ]

    agent = Agent(
        llm_client=mock_llm,
        tools={"search_memory": _search_memory_tool, "web_search": _web_search_tool},
        max_steps=6,
        max_tool_calls=3,
    )
    state = await agent.run("test prompt")

    assert state.status == "completed"
    assert state.final_answer == "Here is the answer."
    assert any(
        s.tool_name == "search_memory" for s in state.steps if s.action == "call_tool"
    )
    assert any(
        s.tool_name == "web_search" for s in state.steps if s.action == "call_tool"
    )


@pytest.mark.asyncio
async def test_agent_blocks_premature_finalize_without_memory(mock_llm: LLMClient):
    """Agent should force search_memory if LLM tries to finalize without calling it."""
    mock_llm.chat_completion.side_effect = [
        # Step 1: LLM tries to finalize immediately
        _json_response(
            '{"action": "finalize", "final_answer": "Answer without memory."}'
        ),
        # Step 2: after forced memory search, LLM finalizes properly
        _json_response('{"action": "finalize", "final_answer": "Proper answer."}'),
    ]

    agent = Agent(
        llm_client=mock_llm,
        tools={"search_memory": _search_memory_tool, "web_search": _web_search_tool},
        max_steps=6,
    )
    state = await agent.run("test prompt")

    assert state.status == "completed"
    # search_memory should have been called by the forced path
    tool_steps = [s for s in state.steps if s.action == "call_tool"]
    assert any(s.tool_name == "search_memory" for s in tool_steps)


@pytest.mark.asyncio
async def test_agent_blocks_web_search_after_memory_hit(mock_llm: LLMClient):
    """Agent should block web_search when memory returned useful results."""

    def _memory_with_hit(query: str) -> str:
        return "Relevant memory:\n1. (relevance=0.9) Some useful info"

    mock_llm.chat_completion.side_effect = [
        # Step 1: call search_memory
        _json_response(
            '{"action": "call_tool", "tool_name": "search_memory", "tool_input": {"query": "test"}}'
        ),
        # Step 2: LLM tries web_search despite memory hit (should be blocked)
        _json_response(
            '{"action": "call_tool", "tool_name": "web_search", "tool_input": {"query": "test"}}'
        ),
    ]

    agent = Agent(
        llm_client=mock_llm,
        tools={"search_memory": _memory_with_hit, "web_search": _web_search_tool},
        max_steps=6,
    )
    state = await agent.run("test prompt")

    # Should have exited to synthesis since web_search was blocked
    assert state.status == "completed"
    web_search_calls = [
        s
        for s in state.steps
        if s.action == "call_tool" and s.tool_name == "web_search"
    ]
    assert len(web_search_calls) == 0


@pytest.mark.asyncio
async def test_agent_max_tool_calls_forces_synthesis(mock_llm: LLMClient):
    """After max_tool_calls, agent should stop calling tools and synthesize."""
    mock_llm.chat_completion.side_effect = [
        # Step 1: call search_memory
        _json_response(
            '{"action": "call_tool", "tool_name": "search_memory", "tool_input": {"query": "test"}}'
        ),
        # Step 2: call web_search
        _json_response(
            '{"action": "call_tool", "tool_name": "web_search", "tool_input": {"query": "test"}}'
        ),
        # Step 3: tries another tool but should be blocked by max_tool_calls=2
        _json_response(
            '{"action": "call_tool", "tool_name": "web_search", "tool_input": {"query": "test2"}}'
        ),
    ]
    mock_llm.chat_completion_text.return_value = _text_response(
        "Synthesized answer from results."
    )

    agent = Agent(
        llm_client=mock_llm,
        tools={"search_memory": _search_memory_tool, "web_search": _web_search_tool},
        max_steps=6,
        max_tool_calls=2,
    )
    state = await agent.run("test prompt")

    assert state.status == "completed"
    tool_steps = [
        s
        for s in state.steps
        if s.action == "call_tool" and s.tool_name != "save_to_memory"
    ]
    assert len(tool_steps) <= 2


@pytest.mark.asyncio
async def test_agent_invalid_json_retry(mock_llm: LLMClient):
    """Agent should retry once on invalid JSON, then fall back to raw text."""
    mock_llm.chat_completion.side_effect = [
        # Step 1: invalid JSON
        _json_response("I don't understand what you mean."),
        # Step 2: retry also invalid
        _json_response("Still not JSON, here is my answer: 42"),
    ]

    agent = Agent(
        llm_client=mock_llm,
        tools={},
        max_steps=4,
    )
    state = await agent.run("What is 6 * 7?")

    assert state.status == "completed"
    assert "42" in (state.final_answer or "")


@pytest.mark.asyncio
async def test_agent_loop_handles_tool_exception(mock_llm: LLMClient):
    """Agent should record an error step when a tool raises an exception."""

    def _broken_tool(query: str) -> str:
        raise RuntimeError("Tool is broken")

    mock_llm.chat_completion.side_effect = [
        _json_response(
            '{"action": "call_tool", "tool_name": "broken", "tool_input": {"query": "test"}}'
        ),
    ]

    agent = Agent(
        llm_client=mock_llm,
        tools={"broken": _broken_tool},
        max_steps=4,
    )
    state = await agent.run("test")

    error_steps = [s for s in state.steps if s.action == "error"]
    assert len(error_steps) >= 1
    assert "broken" in (error_steps[0].output or "")


@pytest.mark.asyncio
async def test_agent_unknown_tool_records_error(mock_llm: LLMClient):
    """Agent should record an error when LLM calls a tool that doesn't exist."""
    mock_llm.chat_completion.side_effect = [
        _json_response(
            '{"action": "call_tool", "tool_name": "nonexistent", "tool_input": {"query": "test"}}'
        ),
    ]

    agent = Agent(
        llm_client=mock_llm,
        tools={},
        max_steps=4,
    )
    state = await agent.run("test")

    error_steps = [s for s in state.steps if s.action == "error"]
    assert any("not found" in (s.output or "") for s in error_steps)


@pytest.mark.asyncio
async def test_agent_think_loop_breaks(mock_llm: LLMClient):
    """Agent should break out if LLM keeps returning think actions and exhaust max_steps."""
    mock_llm.chat_completion.side_effect = [
        _json_response('{"action": "think"}'),
        _json_response('{"action": "think"}'),
    ]
    mock_llm.chat_completion_text.return_value = _text_response(
        "Forced synthesis answer."
    )

    agent = Agent(
        llm_client=mock_llm,
        tools={},
        max_steps=2,
    )
    state = await agent.run("test")

    assert state.status == "completed"
    assert state.final_answer is not None


@pytest.mark.asyncio
async def test_agent_defensive_parsing_infers_action_from_tool_name(
    mock_llm: LLMClient,
):
    """Agent should infer 'call_tool' action when model omits the action key."""
    mock_llm.chat_completion.side_effect = [
        # Missing "action" but has "tool_name" — should be inferred as call_tool
        _json_response(
            '{"tool_name": "search_memory", "tool_input": {"query": "test"}}'
        ),
        # Then finalize
        _json_response('{"action": "finalize", "final_answer": "Done."}'),
    ]

    agent = Agent(
        llm_client=mock_llm,
        tools={"search_memory": _search_memory_tool},
        max_steps=6,
    )
    state = await agent.run("test")

    assert state.status == "completed"
    assert any(
        s.tool_name == "search_memory" for s in state.steps if s.action == "call_tool"
    )
