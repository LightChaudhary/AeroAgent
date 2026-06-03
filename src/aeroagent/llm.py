"""Async Ollama client with OpenAI-compatible API and structured JSON handling."""
from __future__ import annotations
import json
import asyncio
from typing import Any
import httpx

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:1.5b"

class LLMCLient:
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