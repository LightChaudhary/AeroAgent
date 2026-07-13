"""Automated tests for the async agent loop and tool execution."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.aeroagent.agent import Agent
from src.aeroagent.llm import LLMClient


# --- Fixtures ---
@pytest.fixture
def mock_llm_client() -> LLMClient:
    """Create a mock LLM client that returns predictable JSON."""
    client = LLMClient()
    client.chat_completion = AsyncMock()
    return client


@pytest.fixture
def dummy_tool():
    """A simple sync tool for testing."""

    def _tool(query: str) -> str:
        return f"Mocked result for: {query}"

    return _tool


# --- Tests ---
@pytest.mark.asyncio
async def test_agent_initialization(mock_llm_client: LLMClient, dummy_tool):
    """Test that the agent initializes with correct state and tools."""
    agent = Agent(llm_client=mock_llm_client, tools={"dummy": dummy_tool}, max_steps=2)

    assert agent.llm == mock_llm_client
    assert "dummy" in agent.tools
    assert agent.max_steps == 2


@pytest.mark.asyncio
async def test_agent_tool_execution(mock_llm_client: LLMClient, dummy_tool):
    """Test that the agent correctly routes to and executes a tool."""
    # Force the LLM to return a tool call decision
    mock_llm_client.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"action": "call_tool", "tool_name": "dummy", "tool_input": {"query": "test"}}'
                }
            }
        ]
    }

    agent = Agent(llm_client=mock_llm_client, tools={"dummy": dummy_tool}, max_steps=2)
    state = await agent.run("Run the dummy tool")

    assert state.status == "completed"  # Max steps reached, didn't finalize
    assert len(state.steps) >= 2
    assert any(
        step.action == "call_tool" and step.tool_name == "dummy" for step in state.steps
    )
    assert any("Mocked result for: test" in (step.output or "") for step in state.steps)


@pytest.mark.asyncio
async def test_agent_defensive_parsing_plain_text(mock_llm_client: LLMClient):
    """Test that the agent gracefully handles non-JSON LLM output."""
    # Force the LLM to return plain text (simulating a small model failing JSON schema)
    mock_llm_client.chat_completion.return_value = {
        "choices": [
            {"message": {"content": "I cannot use tools, but the answer is 42."}}
        ]
    }

    agent = Agent(llm_client=mock_llm_client, tools={}, max_steps=2)
    state = await agent.run("What is the answer?")

    assert state.status == "completed"
    assert state.final_answer == "I cannot use tools, but the answer is 42."
    assert any(step.action == "finalize" for step in state.steps)
