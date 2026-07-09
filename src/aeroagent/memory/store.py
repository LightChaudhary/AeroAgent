"""ChromaDB persistent vector store for AeroAgent memory."""

from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings

from .embedder import EMBEDDING_DIM

CHROMA_PERSIST_DIR = "data/memory"
COLLECTION_NAME = "aeroagent_memory"


class MemoryStore:
    """Persistent local vector store backed by ChromaDB."""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[MemoryStore] Ready. Collection= '{collection_name}' "
            f"| Documents={self._collection.count()} "
            f"| Path= '{persist_dir}'"
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Store a text + its embedding vector.

        Uses a content hash as the ID so saving identical text twice is a
        no-op — ChromaDB will just overwrite with the same data rather than
        creating a duplicate entry.

        Returns the doc_id.
        """
        # Deterministic ID from content — prevents duplicates across runs
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        doc_id = doc_id or f"mem_{content_hash}"

        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        # ChromaDB `upsert` overwrites if the ID exists — safe for dedup
        self._collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[meta],
        )
        return doc_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        vector: list[float],
        top_k: int = 3,
        min_relevance: float = 0.4,
    ) -> list[dict[str, Any]]:
        """Find the top-k most similar documents above min_relevance."""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 = identical, 0 = opposite
            score = 1 - (dist / 2)
            if score >= min_relevance:
                hits.append({"text": doc, "score": round(score, 4), "metadata": meta})

        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of stored memories."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all memories (useful for testing)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print("[MemoryStore] Cleared all memories.")


# Global store instance
store = MemoryStore()
