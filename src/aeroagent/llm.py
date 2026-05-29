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
    pass