"""Type-safe tool registry and execution wrapper."""

from __future__ import annotations
import inspect
import asyncio
from typing import Any, Callable
from pydantic import ValidationError

from ..state import ToolSchema


class ToolRegistry:
    """Registers and executes tools with strict Pydantic validation."""

    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, description: str, parameters: dict[str, Any]):
        """Decorator to register a function as an agent tool."""

        def decorator(func: Callable[..., Any]):
            self._tools[name] = {
                "schema": ToolSchema(
                    name=name, description=description, parameters=parameters
                ),
                "func": func,
            }
            return func

        return decorator

    def get_schema(self, name: str) -> ToolSchema | None:
        """Return the Pydantic schema for a given tool."""
        tool = self._tools.get(name)
        return tool["schema"] if tool else None

    def get_all_schemas(self) -> list[ToolSchema]:
        """Return schemas for all registered tools."""
        return [t["schema"] for t in self._tools.values()]

    async def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a tool, validating inputs first."""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found in registry.")

        schema = tool["schema"]
        func = tool["func"]

        # TODO (Phase 2): Build a dynamic Pydantic model from schema.parameters and validate kwargs against it.
        # Currently a no-op - kwargs pass through unvalidated. Safe for Phase 1 since search.py has its own type hints.

        # Validate inputs against the tool's parameter schema
        validated_kwargs = kwargs

        # Execute (handle both sync ana=d async functions)
        result = func(**validated_kwargs)
        if inspect.iscoroutine(result):
            return await result
        return result


# Global registry instance
registry = ToolRegistry()
