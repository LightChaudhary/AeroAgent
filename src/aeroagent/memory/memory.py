"""MemoryManager : high-level interface for AeroAgent memory."""
from __future__ import annotations
from typing import Any

from .embedder import embedder
from .store import store

class MemoryManager:
    """
    High-level memory interface for the agent. Ties the embedder and vector store together.
    All agent code should use this - never talk to embedder/store directly.
    """

    def __init__(self):
        self._embedder = embedder
        self._store = store
    
    # ------------------------------------------------------------------
    # Write 
    # ------------------------------------------------------------------

    def remember(self, 
                 text: str, 
                 metadata: dict[str, Any] | None = None,
            ) -> str:
        """
        Embed and persist a piece of text to memory.
        Returns the doc_id of the stored memory.

        Usage:
            memory.remember("Python 3.13 introduced free-threaded mode.")
        """
        if not text or not text.strip():
            return "empty"
        
        vector = self._embedder.encode(text)
        doc_id = self._store.add(text=text, vector=vector, metadata=metadata or {})
        print(f"[Memory] Saved -> id='{doc_id}' | {text[:60]}... ")
        return doc_id
    
    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recall(self, 
               query: str, 
               top_k: int = 3, 
               min_relevance: float = 0.4,
            ) -> list[dict[str, Any]]:
        """
        Search memory for context relevant to the query.
        Returns top-k results above the relevance threshold.

        Usage:
            results = memory.recall("Python 3.13 features")
            # [{"text": "...", "score": 0.87, "metadata": {...}}, ...]
        """
        if not query or not query.strip():
            return []
        
        vector = self._embedder.encode(query)
        results = self._store.query(vector=vector, top_k=top_k, min_relevance=min_relevance)
        print(f"[Memory] Recall -> query='{query[:50]}' | hits={len(results)}")
        return results
    
    def format_recall(self, query: str, top_k: int = 3) -> str:
        """
        Recall and format results as a string for the LLM context.
        Returns 'No relevant memory found.' if nothing matches.

        Usage: 
            context = memory.format_recall("Python 3.13 features")
            # Ready to inject into the LLM prompt directly
        """
        results = self.recall(query=query, top_k=top_k)

        if not results:
            return "No relevant memory found."
        
        lines = ["Relevant memory:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. (relevance={r['score']}) {r['text']}")
        
        return "\n".join(lines)
    
    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of stored memories."""
        return self._store.count()
    
    def clear(self) -> None:
        """Wipe all memories. Useful for testing."""
        self._store.clear()

# Global memory instance
memory = MemoryManager()