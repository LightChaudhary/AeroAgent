"""FastAPI app exposing the AeroAgent as a REST service.

Run locally with:
    uvicorn src.aeroagent.api.main:app --reload

Requires Ollama running locally, same as the CLI.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from ..agent import Agent
from ..llm import DEFAULT_MODEL, LLMClient
from ..tools.memory import search_memory
from ..tools.search import web_search
from ..tracer import tracer
from .schemas import HealthResponse, RunRequest, RunResponse, StepOut

# Single shared LLMClient for the app's lifetime, closed cleanly on shutdown.
__llm_client: LLMClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global __llm_client
    __llm_client = LLMClient()
    yield
    if __llm_client is not None:
        await __llm_client.close()
    
app = FastAPI(
    title="AeroAgent API",
    description="REST interface for the AeroAgent async agent framework.",
    version="0.1.0",
    lifespan=lifespan,
)

def _build_tools() -> dict[str, Any]:
    """Tools available to the agent. save_to_memory is auto-invoked, not exposed."""
    return {
        "search_memory": search_memory,
        "web_search": web_search,
    }

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model=DEFAULT_MODEL)

@app.post("/run", response_model=RunResponse)
async def run_agent(request: RunRequest) -> RunResponse:
    if __llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized.")
    
    agent = Agent(
        llm_client=__llm_client,
        tools=_build_tools(),
        max_steps=request.max_steps,
        max_tool_calls=request.max_tool_calls,
        prompt_version=request.prompt_version,
    )

    try:
        state = await agent.run(request.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")
    
    trace_id = tracer.generate_trace_id()
    tracer.save_trace(state, trace_id=trace_id, metadata={"interface": "api"})

    return RunResponse(
        trace_id=trace_id,
        status=state.status,
        final_answer=state.final_answer,
        prompt_version=state.prompt_version,
        metrics=state.metrics,
        steps=[
            StepOut(
                step_id=s.step_id,
                action=s.action,
                tool_name=s.tool_name,
                output=s.output,
                latency_ms=s.latency_ms,
            )
            for s in state.steps
        ],
    )