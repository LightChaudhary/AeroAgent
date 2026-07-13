"""Async Ollama client with OpenAI-compatible API and structured JSON handling."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

from .observability.metrics import CallMetrics, estimate_cost, timed

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("AEROAGENT_MODEL", "llama3.2:3b")


class LLMClient:
    """Async HTTP client for local Ollama instances."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Execute chat completion in JSON mode — for agent loop decisions only."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        return await self._post(payload)

    async def chat_completion_text(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Execute chat completion in plain-text mode — for synthesis / summarisation.

        Does NOT set response_format so the model can reply in natural language
        without being forced to wrap everything in a JSON object.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # No response_format — plain text output
        }
        return await self._post(payload)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Shared HTTP POST with retry logic.

        Attaches a `_metrics` key (latency, token countsm estimated cost) to the returned response dict. This
        is additive: existing callers that only read `choices` are unaffected, and callers that don't care
        about metrics can ignore the key entirely.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                with timed() as elapsed:
                    resp = await self._client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                latency_ms = elapsed()

                usage = data.get("usage", {}) or {}
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get(
                    "total_tokens", prompt_tokens + completion_tokens
                )

                metrics = CallMetrics(
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimate_cost(
                        self.model, prompt_tokens, completion_tokens
                    ),
                )
                data["_metrics"] = metrics.to_dict()
                return data
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"LLM request failed after {self.max_retries} attempts: {e}"
                    ) from e
                await asyncio.sleep(1.5 * attempt)
            except httpx.HTTPStatusError as e:
                if attempt == self.max_retries or e.response.status_code < 500:
                    raise RuntimeError(
                        f"LLM returned {e.response.status_code}: {e.response.text}"
                    ) from e
                await asyncio.sleep(1.5 * attempt)

    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        """Extract JSON from model output."""
        text = text.strip()

        # Remove markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extract first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None
