"""Local JSON trace logger for agent execution debugging and observability."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .state import AgentState


class AgentTracer:
    """Handles writing agent execution traces to local JSON files."""

    def __init__(self, trace_dir: str = "traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def generate_trace_id(self) -> str:
        """Generate a unique ID for single agent run."""
        return f"trace_{uuid.uuid4().hex[:8]}"

    def save_trace(
        self,
        state: AgentState,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Serialize and save the agent state to a JSON file.
        Returns the path to the saved trace file.
        """
        trace_id = trace_id or self.generate_trace_id()
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{trace_id}.json"
        filepath = self.trace_dir / filename

        trace_data = {
            "trace_id": trace_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "status": state.status,
            "prompt": state.prompt,
            "final_answer": state.final_answer,
            "error_message": state.error_message,
            "steps": [step.model_dump(mode="json") for step in state.steps],
            "metadata": metadata or {},
            "prompt_version": state.prompt_version,
            "metrics": state.metrics,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        return str(filepath)


# Global tracer instance
tracer = AgentTracer()
