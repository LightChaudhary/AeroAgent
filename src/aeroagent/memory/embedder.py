"""Sentence-transformer embedding wrapper for AeroAgent memory."""
from __future__ import annotations
import torch
from sentence_transformers import SentenceTransformer

# Auto-detect best available device
# MPS = Apple Silicon GPU, falls back to CPU
def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384 # fixed output size for all-MiniLM-L6-v2

class Embedder:
    """Wraps sentence-transformers to produce fixed-size embedding vectors."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None,):
        self.device = device or _get_device()
        self.model_name = model_name
        # Load at import time - first call will be fast
        print(f"[Embedder] Loading '{model_name}' on device='{self.device}'...")
        self._model = SentenceTransformer(model_name, device=self.device)
        print(f"[Embedder] Ready. Embedding dim ={EMBEDDING_DIM}")
    
    def encode(self, text:str) -> list[float]:
        """Encode a single string into a fixed-size embedding vector."""
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()
    
    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode mutliple strings in one forward pass (more efficient)."""
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return vectors.tolist()

# Global embedder instance - loaded once at startup
embedder = Embedder()