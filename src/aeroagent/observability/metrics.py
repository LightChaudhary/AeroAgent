"""Latency and token/cost metrics utilities for LLM calls.

Local Ollama models are free to run, so estimated_cost_usd will be 0.0 for the default setup. The 
pricing table exists so this module keeps working unchanged if LLMClient is ever pointed at a 
hosted provider.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

# USD per 1M tokens, as (prompt_price, completion_price). Unknown models (including all local Ollama models)
# default to (0.0, 0.0).
MODEL_PRICING: dict[str, tuple[float, float]] = {}

@dataclass
class CallMetrics:
    """Metrics captured for a single LLM call."""

    latency_ms: float 
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }
    
def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a call. Returns 0.0 for local/unpriced models."""
    prompt_price, completion_price = MODEL_PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens / 1_000_000) * prompt_price + (completion_tokens / 1_000_000) * completion_price

@contextmanager
def timed() -> Iterator[Any]:
    """Context manager yielding a callable that returns elapsed ms so far.
    
    Usage:
    with timed() as elapsed:
        ...do work...
    ms = elapsed()
    """
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000