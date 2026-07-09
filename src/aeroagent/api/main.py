"""FastAPI app exposing the AeroAgent as a REST service.

Run locally with:
    uvicorn src.aeroagent.api.main:app --reload

Requires Ollama running locally, same as the CLI.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from ..agent import Agent
from ..llm import DEFAULT_MODEL, LLMClient
from ..tools.memory import search_memory
from ..tools.search import web_search
from ..tracer import tracer
from .schemas import HealthResponse, RunRequest, RunResponse, StepOut

# Single shared LLMClient for the app's lifetime, closed cleanly on shutdown.
_llm_client: LLMClient | None = None


def get_client_ip(request: Request) -> str:
    """Resolve the real client IP.

    Falls back to X-Forwarded-For when running behind a reverse proxy or load
    balancer (nginx, cloud LB, Docker Compose in front of the app), since
    get_remote_address alone would return the proxy's IP and effectively make
    every client share a single rate-limit bucket.

    NOTE: only trust X-Forwarded-For if the proxy in front of this app is
    known to set/overwrite it (not just append to a client-supplied value).
    Otherwise a client can spoof this header to bypass rate limiting.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    return forwarded.split(",")[0].strip() if forwarded else get_remote_address(request)


# Per-client-IP rate limiting. /run is the expensive endpoint (real LLM calls, tool execution) so it
# gets a tighter limit than the default.
limiter = Limiter(key_func=get_client_ip)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _llm_client
    _llm_client = LLMClient()
    yield
    if _llm_client is not None:
        await _llm_client.close()


app = FastAPI(
    title="AeroAgent API",
    description="REST interface for the AeroAgent async agent framework.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Ensures Retry-After / X-RateLimit-* headers are set correctly on 429s.
app.add_middleware(SlowAPIMiddleware)


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
@limiter.limit("10/minute")
async def run_agent(request: Request, body: RunRequest) -> RunResponse:
    if _llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized.")

    agent = Agent(
        llm_client=_llm_client,
        tools=_build_tools(),
        max_steps=body.max_steps,
        max_tool_calls=body.max_tool_calls,
        prompt_version=body.prompt_version,
    )

    try:
        state = await agent.run(body.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}") from e

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
