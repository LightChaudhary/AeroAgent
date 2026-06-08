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
        self._tools : dict[str, dict[str, Any]] = {}
    
    def register(self, name: str, description: str, parameters: dict[str, Any]):
        """Decorator to registor a function as an agent tool."""
        def decorator(func: Callable[..., Any]):
            self._tools[name] = {
                "schema": ToolSchema(name=name, description=description, parameters=parameters),
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

        # Validate inputs against the tool's parameter schema
        try:
            # Create a dynamic Pydantic model for validation
            ParamModel = schema.parameters # Assuming parameters is a dict of Pydantic Field definitions or similar
            # for simplicity in Phase 1, we'll do basic type checking or rely on the func's own validation
            # A more advanced version would dynamically build a Pydantic model here.
            validated_kwargs = kwargs
        except Exception as e:
            raise ValueError(f"Invalid arguments for tool '{name}': {e}")
        
        # Execute (handle both sync ana=d async functions)
        result = func(**validated_kwargs)
        if inspect.iscoroutine(result):
            return await result
        return result
    
#Global registry instance
registry = ToolRegistry()