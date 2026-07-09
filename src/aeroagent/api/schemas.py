"""Request/response models for the FastAPI wrapper.

Kept separate from AgentState (src/aeroagent/state.py) on purpose: the API contract should be able to
evolve independently of the internal state model used by the agent loop.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..prompts.registry import DEFAULT_PROMPT_VERSION


class RunRequest(BaseModel):
    """Incoming request body for POST /run."""

    prompt: str = Field(min_length=1, description="The user's natural-language prompt.")
    prompt_version: str = Field(
        default=DEFAULT_PROMPT_VERSION,
        description="Which registered system prompt version to use.",
    )
    max_steps: int = Field(default=8, ge=1, le=20)
    max_tool_calls: int = Field(default=2, ge=0, le=10)


class StepOut(BaseModel):
    """A single execution step, trimmed for API consumption."""

    step_id: int
    action: str
    tool_name: str | None = None
    output: str | None = None
    latency_ms: float | None = None


class RunResponse(BaseModel):
    """Response body for POST /run."""

    trace_id: str
    status: str
    final_answer: str | None
    prompt_version: str | None
    metrics: dict[str, Any]
    steps: list[StepOut]


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    model: str
