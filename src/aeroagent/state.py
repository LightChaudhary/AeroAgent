"""Pydantic v2 state models for the AeroAgent framework."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, field_validator, Field, ConfigDict

class ToolSchema(BaseModel):
    """Defines a callable tool with strict validation."""
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError("Tool name must be alphanumeric with underscores only")
        return v.lower()

class AgentStep(BaseModel):
    """Represents a single execution step in the agent loop."""
    model_config = ConfigDict(extra="allow")
    step_id: int
    action: Literal["think", "call_tool", "finalize", "error"]
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    output: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float | None = None

class AgentState(BaseModel):
    """State tracking for the async agent loop."""
    prompt: str
    status: Literal["pending", "running", "completed", "error"] = "pending"
    steps: list[AgentStep] = Field(default_factory=list)
    final_answer: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    def add_step(self, action: Literal["think", "call_tool", "finalize", "error"], tool_name: str | None = None, **kwargs: Any) -> None: 
        """Append a new step to the state trace."""
        step = AgentStep(
            step_id = len(self.steps) + 1,
            action = action,
            tool_name = tool_name,
            **kwargs
        )
        self.steps.append(step)
    
    def aggregate_metrics(self) -> dict[str, Any]:
        """Roll up per-step latency into run-level totals and store in `metrics`.
        
        Safe to call even if no steps carry latency_ms (e.g mocked LLM clients in tests) - totals simply come out as 0.
        """
        latencies = [s.latency_ms for s in self.steps if s.latency_ms is not None]
        self.metrics = {
            "total_latency_ms": round(sum(latencies), 2),
            "llm_call_count": len(latencies),
            "tool_call_count": sum(1 for s in self.steps if s.action == "call_tool"),
            "step_count": len(self.steps),
        }
        return self.metrics