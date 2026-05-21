"""Pydantic v2 state models for the AeroAgent framework."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, field_validator, Field

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
    pass

class AgentState(BaseModel):
    pass
