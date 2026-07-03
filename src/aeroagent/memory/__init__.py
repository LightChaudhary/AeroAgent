"""AeroAgent memory subsystem.

Exposes the global instances used throughout the agent.
Import from here rather than from submodules directly:

    from src.aeroagent.memory import memory
    from src.aeroagent.memory import embedder, store  # if needed
"""

from .embedder import Embedder, embedder
from .memory import MemoryManager, memory
from .store import MemoryStore, store

__all__ = [
    "Embedder", "embedder",
    "MemoryManager", "memory",
    "MemoryStore", "store",
]