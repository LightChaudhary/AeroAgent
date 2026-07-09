"""Offline tests for the eval harness: EvalCase validation and judge scoring.

These tests mock LLMClient entirely, so they run without live Ollama - consistent with the rest of 
the test suite.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock

from src.aeroagent.evals.dataset import EVAL_CASES, EvalCase
from src.aeroagent.evals.judge import judge_case
from src.aeroagent.llm import LLMClient
from src.aeroagent.state import AgentState

# --- Fixtures ---
@pytest.fixture
def mock_llm_client() -> LLMClient:
    client = LLMClient()
    client.chat_completion = AsyncMock()
    return client

@pytest.fixture
def sample_eval_case() -> EvalCase:
    return EvalCase(
        case_id="test_001",
        prompt="What is the capital of France?",
        criteria="The answer must state that Paris is the capital of France.",
        category="factual",
    )

# --- EvalCase / dataset tests ---
def test_eval_requires_core_fields():
    """EvalCase should reject construction without required fields."""
    with pytest.raises(ValidationError):
        EvalCase(case_id="missing_fields")
    
def test_eval_case_defaults_category_to_general():
    case = EvalCase(case_id="x", prompt="hi", criteria="say hi back")
    assert case.category == "general"

def test_starter_dataset_has_unique_case_ids():
    """Guards against accidental duplicate case_ids in the starter dataset."""
    ids = [c.case_id for c in EVAL_CASES]
    assert len(ids) == len(set(ids))

def test_starter_dataset_is_non_empty():
    assert len(EVAL_CASES) > 0

# --- judge_case tests ---
@pytest.mark.asyncio
async def test_judge_case_parses_valid_score(
    mock_llm_client: LLMClient, sample_eval_case: EvalCase, sample_agent_state: AgentState
):
    mock_llm_client.chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": '{"score": 5, "rationale": "Correct and complete."}'
            }
        }]
    }

    result = await judge_case(mock_llm_client, sample_eval_case, sample_agent_state)
    
    assert result.case_id == "test_001"
    assert result.score == 5
    assert result.rationale == "Correct and complete."
    assert result.raw_answer == "Paris is the capital of France."

@pytest.mark.asyncio
async def test_judge_case_clamps_out_of_range_score(
    mock_llm_client: LLMClient, sample_eval_case: EvalCase, sample_agent_state: AgentState
):
    """Judge model hallucinating a score outside 1-5 shouldn't crash JudgeResult."""
    mock_llm_client.chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": '{"score": 9, "rationale": "Way off scale."}'
            }
        }]
    }

    result = await judge_case(mock_llm_client, sample_eval_case, sample_agent_state)

    assert result.score == 5  # clamped to max

@pytest.mark.asyncio
async def test_judge_case_falls_back_on_invalid_json(
    mock_llm_client: LLMClient, sample_eval_case: EvalCase, sample_agent_state: AgentState
):
    """Malformed judge output should degrade to a low score, not raise."""
    mock_llm_client.chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": "I refuse to output JSON today."
            }
        }]
    }

    result = await judge_case(mock_llm_client, sample_eval_case, sample_agent_state)

    assert result.score == 1
    assert "could not be parsed" in result.rationale.lower()


@pytest.mark.asyncio
async def test_judge_case_falls_back_on_llm_exception(
    mock_llm_client: LLMClient, sample_eval_case: EvalCase, sample_agent_state: AgentState
):
    """A raised exception during the judge call shouldn't crash the eval run."""
    mock_llm_client.chat_completion.side_effect = RuntimeError("connection refused")

    result = await judge_case(mock_llm_client, sample_eval_case, sample_agent_state)

    assert result.score == 1
    assert "judge call failed" in result.rationale.lower()

@pytest.mark.asyncio
async def test_judge_case_preserves_raw_answer_even_when_empty(
    mock_llm_client: LLMClient, sample_eval_case: EvalCase
):
    """If the agent produced no final_answer, judge_case shouldn't crash."""
    empty_state = AgentState(prompt="What is the capital of France?", status="error")
    empty_state.final_answer = None

    mock_llm_client.chat_completion.return_value = {
        "choices": [{
            "message": {
                "content": '{"score": 1, "rationale": "No answer provided."}'
            }
        }]
    }

    result = await judge_case(mock_llm_client, sample_eval_case, empty_state)

    assert result.raw_answer == ""
    assert result.score == 1