"""Unit tests for AeroAgent memory subsystem.

Covers:
- Embedder: encode, encode_batch, output shape
- MemoryStore: add, upsert dedup, query, count, clear
- MemoryManager: remember, recall, format_recall
"""

from __future__ import annotations

import pytest

from src.aeroagent.memory.embedder import EMBEDDING_DIM, Embedder
from src.aeroagent.memory.store import MemoryStore
from src.aeroagent.memory.memory import MemoryManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Shared embedder instance — loaded once for the whole module."""
    return Embedder()


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    """Fresh in-process MemoryStore backed by a temp directory."""
    s = MemoryStore(
        persist_dir=str(tmp_path / "chroma"), collection_name="test_collection"
    )
    yield s
    s.clear()
    # Explicitly close ChromaDB client to release file handles
    s._client.clear_system_cache()


@pytest.fixture()
def manager(tmp_path) -> MemoryManager:
    """Fresh MemoryManager with its own isolated store."""
    from src.aeroagent.memory.embedder import Embedder
    from src.aeroagent.memory.store import MemoryStore

    _embedder = Embedder()
    _store = MemoryStore(
        persist_dir=str(tmp_path / "chroma"), collection_name="test_manager"
    )

    m = MemoryManager()
    m._embedder = _embedder
    m._store = _store
    yield m
    m.clear()
    # Explicitly close ChromaDB client to release file handles
    m._store._client.clear_system_cache()


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------


class TestEmbedder:
    def test_encode_returns_list(self, embedder):
        vector = embedder.encode("hello world")
        assert isinstance(vector, list)

    def test_encode_correct_dim(self, embedder):
        vector = embedder.encode("test sentence")
        assert len(vector) == EMBEDDING_DIM

    def test_encode_floats(self, embedder):
        vector = embedder.encode("test")
        assert all(isinstance(v, float) for v in vector)

    def test_encode_batch_shape(self, embedder):
        texts = ["first sentence", "second sentence", "third sentence"]
        vectors = embedder.encode_batch(texts)
        assert len(vectors) == 3
        assert all(len(v) == EMBEDDING_DIM for v in vectors)

    def test_encode_batch_single(self, embedder):
        vectors = embedder.encode_batch(["only one"])
        assert len(vectors) == 1
        assert len(vectors[0]) == EMBEDDING_DIM

    def test_same_text_same_vector(self, embedder):
        v1 = embedder.encode("deterministic embedding")
        v2 = embedder.encode("deterministic embedding")
        assert v1 == v2

    def test_different_text_different_vector(self, embedder):
        v1 = embedder.encode("cat")
        v2 = embedder.encode("quantum mechanics")
        assert v1 != v2


# ---------------------------------------------------------------------------
# MemoryStore tests
# ---------------------------------------------------------------------------


class TestMemoryStore:
    def _make_vector(self) -> list[float]:
        return [0.0] * EMBEDDING_DIM

    def test_initial_count_zero(self, store):
        assert store.count() == 0

    def test_add_increments_count(self, store):
        store.add(text="hello", vector=self._make_vector())
        assert store.count() == 1

    def test_add_returns_doc_id(self, store):
        doc_id = store.add(text="some text", vector=self._make_vector())
        assert doc_id.startswith("mem_")

    def test_add_same_text_no_duplicate(self, store):
        text = "identical content"
        store.add(text=text, vector=self._make_vector())
        store.add(text=text, vector=self._make_vector())
        assert store.count() == 1

    def test_add_different_text_two_docs(self, store):
        store.add(text="first doc", vector=self._make_vector())
        store.add(text="second doc", vector=self._make_vector())
        assert store.count() == 2

    def test_content_hash_id_deterministic(self, store):
        text = "hash me"
        id1 = store.add(text=text, vector=self._make_vector())
        id2 = store.add(text=text, vector=self._make_vector())
        assert id1 == id2

    def test_query_empty_store_returns_empty(self, store):
        results = store.query(vector=self._make_vector())
        assert results == []

    def test_query_returns_hits(self, store, embedder):
        text = "Python is a programming language"
        vector = embedder.encode(text)
        store.add(text=text, vector=vector)

        query_vector = embedder.encode("Python programming")
        hits = store.query(vector=query_vector, min_relevance=0.0)
        assert len(hits) >= 1

    def test_query_hit_has_required_keys(self, store, embedder):
        text = "machine learning"
        vector = embedder.encode(text)
        store.add(text=text, vector=vector)

        hits = store.query(
            vector=embedder.encode("machine learning"), min_relevance=0.0
        )
        assert hits
        assert "text" in hits[0]
        assert "score" in hits[0]
        assert "metadata" in hits[0]

    def test_query_score_between_0_and_1(self, store, embedder):
        text = "neural networks"
        vector = embedder.encode(text)
        store.add(text=text, vector=vector)

        hits = store.query(vector=embedder.encode("neural networks"), min_relevance=0.0)
        for hit in hits:
            assert 0.0 <= hit["score"] <= 1.0

    def test_query_min_relevance_filters(self, store, embedder):
        store.add(text="apple fruit", vector=embedder.encode("apple fruit"))
        hits = store.query(
            vector=embedder.encode("quantum physics"), min_relevance=0.99
        )
        assert hits == []

    def test_query_sorted_by_score_descending(self, store, embedder):
        store.add(
            text="Python 3.13 release", vector=embedder.encode("Python 3.13 release")
        )
        store.add(
            text="banana smoothie recipe",
            vector=embedder.encode("banana smoothie recipe"),
        )

        hits = store.query(
            vector=embedder.encode("Python release"), min_relevance=0.0, top_k=2
        )
        if len(hits) > 1:
            assert hits[0]["score"] >= hits[1]["score"]

    def test_clear_resets_count(self, store):
        store.add(text="will be cleared", vector=self._make_vector())
        assert store.count() == 1
        store.clear()
        assert store.count() == 0

    def test_metadata_timestamp_stored(self, store):
        store.add(text="timestamped doc", vector=self._make_vector())
        hits = store.query(vector=self._make_vector(), min_relevance=0.0)
        assert hits
        assert "timestamp" in hits[0]["metadata"]


# ---------------------------------------------------------------------------
# MemoryManager tests
# ---------------------------------------------------------------------------


class TestMemoryManager:
    def test_remember_returns_id(self, manager):
        doc_id = manager.remember("Python 3.13 was released in October 2024")
        assert doc_id.startswith("mem_")

    def test_remember_empty_text_returns_early(self, manager):
        result = manager.remember("")
        assert result == "empty"

    def test_remember_whitespace_returns_early(self, manager):
        result = manager.remember("   ")
        assert result == "empty"

    def test_remember_increments_store(self, manager):
        manager.remember("first memory")
        manager.remember("second memory")
        assert manager.count() == 2

    def test_remember_duplicate_no_increment(self, manager):
        manager.remember("duplicate text")
        manager.remember("duplicate text")
        assert manager.count() == 1

    def test_recall_empty_query_returns_empty(self, manager):
        results = manager.recall("")
        assert results == []

    def test_recall_no_memory_returns_empty(self, manager):
        results = manager.recall("something", min_relevance=0.9)
        assert results == []

    def test_recall_finds_relevant_memory(self, manager):
        manager.remember("Python 3.13 introduced free-threaded mode")
        results = manager.recall("Python 3.13 features", min_relevance=0.5)
        assert len(results) >= 1

    def test_recall_result_structure(self, manager):
        manager.remember("ChromaDB is a vector database")
        results = manager.recall("vector database", min_relevance=0.0)
        assert results
        assert "text" in results[0]
        assert "score" in results[0]

    def test_format_recall_no_results(self, manager):
        output = manager.format_recall("xyzzy obscure query 999")
        assert output == "No relevant memory found."

    def test_format_recall_with_results(self, manager):
        manager.remember("AeroAgent is an async AI agent framework")
        output = manager.format_recall("AeroAgent framework")
        assert "Relevant memory:" in output
        assert "relevance=" in output

    def test_format_recall_numbered(self, manager):
        manager.remember("first fact about Python")
        output = manager.format_recall("Python", top_k=3)
        if "Relevant memory:" in output:
            assert "1." in output

    def test_clear_wipes_all(self, manager):
        manager.remember("to be wiped")
        manager.clear()
        assert manager.count() == 0
        output = manager.format_recall("wiped")
        assert output == "No relevant memory found."
