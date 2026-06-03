"""Async Ollama client with OpenAI-compatible API and structured JSON handling."""
from __future__ import annotations
import json
import asyncio
from typing import Any
import httpx

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:1.5b"

class LLMClient:
    """Async HTTP client for local Ollama instances."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL, timeout: float = 30.0,
                 max_retries: int = 2,):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(timeout=self.timeout)
    
    async def close(self) -> None:
        await self._client.aclose()
    
    async def chat_completion(self, messages: list[dict[str, str]], temperature: float =0.0, max_tokens: int=512,) -> dict[str, Any]:
       """Execute chat completion with retry & timeout boundaries."""
       payload = {
            "model": self.model,
           "messages": messages,
           "temperature": temperature,
           "max_tokens": max_tokens,
       }

       for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._client.post(f"{self.base_url}/chat/completions", json=payload,)
                resp.raise_for_status()
                return resp.json()
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {e}") from e
                await asyncio.sleep(1.5 * attempt)
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"LLM returned {e.response.status_code}: {e.response.text}") from e
            
    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        """Safely parse JSON from LLM output, handling markdown wrappers."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```") and lines[-1].startswith("```"):
               text = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
               return None