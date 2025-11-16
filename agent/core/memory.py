# agent/core/memory.py
"""
Memory component for the coding agent with optional Redis backend.

Behavior:
- If a Redis server is reachable at the configured URL (default: redis://localhost:6379/0)
  AND the `redis` Python package is installed, the Memory facade will use a Redis-backed
  store (RedisMemory).
- Otherwise the implementation falls back to an in-process in-memory store (InMemory).
- The API of the facade matches the in-memory version you already had:
    store(), append(), recall(), recall_all(), snapshot()/to_dict(), save(), load(), clear()

Notes:
- RedisMemory implemented here is a pragmatic simple wrapper that stores each memory record
  as JSON in a Redis list (newest items at head). It supports simple recall by scanning the
  list client-side (good for low-to-medium scale). For production-scale search consider
  embedding-based vector indices or RediSearch.
- Redis usage is optional; no external services are required for the fallback in-memory mode.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Sequence, Union

# Prefer project logger if available
try:
    from agent.utils.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Optional redis library
try:
    import redis  # type: ignore

    _HAS_REDIS = True
except Exception:
    redis = None  # type: ignore
    _HAS_REDIS = False


# -------------------------
# Config dataclasses
# -------------------------
@dataclass
class MemoryConfig:
    """
    Configuration for the Memory facade (decides which backend to use).
    - redis_url: if provided and reachable, Redis backend will be used.
                Default: environment GROQ etc? here we use REDIS_URL env or 'redis://localhost:6379/0'
    - namespace: key prefix for Redis-backed memory
    - enable_persistence: for InMemory backend, whether to persist to file after writes
    - persist_path: path for JSON persistence for InMemory backend
    - max_items: cap on items to keep (applies to both backends; Redis uses LTRIM, InMemory truncates)
    """
    redis_url: Optional[str] = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    namespace: str = "agent:memory"
    enable_persistence: bool = False
    persist_path: Optional[str] = None
    max_items: Optional[int] = None


@dataclass
class _Record:
    """Internal representation for storing items consistently across backends."""
    id: int
    timestamp: float
    content: Any
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "timestamp": self.timestamp, "content": self.content, "summary": self.summary}


# -------------------------
# Redis-backed memory
# -------------------------
class RedisMemory:
    """
    Simple Redis-backed memory store.

    Implementation details:
    - Uses LPUSH on a Redis list key (namespace + ':items') to push newest-first.
    - Optionally trims list with LTRIM when max_items is configured.
    - Stores JSON-serialized records. Each record is a dict: {id, timestamp, content, summary}.
    - recall() retrieves the whole list into the client and performs simple text-match ranking.
      This is acceptable for small/medium workloads; replace with RediSearch or vector DB for scale.
    """

    def __init__(self, cfg: MemoryConfig):
        if not _HAS_REDIS:
            raise RuntimeError("redis package not installed; cannot use RedisMemory")
        self.cfg = cfg
        try:
            # Create client
            self._r = redis.from_url(cfg.redis_url)
            # test connectivity
            self._r.ping()
        except Exception as e:
            # Propagate a clear error for the caller to handle
            raise RuntimeError(f"Unable to connect to Redis at {cfg.redis_url}: {e}") from e

        self._list_key = f"{self.cfg.namespace}:items"
        # Keep a simple running id counter in Redis for consistent ids across processes
        self._id_key = f"{self.cfg.namespace}:next_id"

    def _next_id(self) -> int:
        try:
            return int(self._r.incr(self._id_key))
        except Exception:
            # if incr fails, fallback to timestamp-based id
            return int(time.time() * 1000)

    def store(self, item: Any, *, summary: Optional[str] = None) -> Dict[str, Any]:
        rec = {
            "id": self._next_id(),
            "timestamp": time.time(),
            "content": item,
            "summary": summary if summary is not None else self._auto_summary(item),
        }
        payload = json.dumps(rec, ensure_ascii=False)
        # push newest-first
        self._r.lpush(self._list_key, payload)
        if self.cfg.max_items:
            # keep only newest max_items
            self._r.ltrim(self._list_key, 0, self.cfg.max_items - 1)
        return rec

    append = store  # alias

    def recall_all(self) -> List[Dict[str, Any]]:
        raw = self._r.lrange(self._list_key, 0, -1)  # newest-first
        out: List[Dict[str, Any]] = []
        for b in raw:
            try:
                if isinstance(b, bytes):
                    decoded = b.decode("utf-8")
                else:
                    decoded = str(b)
                out.append(json.loads(decoded))
            except Exception:
                out.append({"content": b})
        return out

    def recall(self, query: Union[str, Any], *, k: int = 5, mode: str = "keyword") -> List[Dict[str, Any]]:
        q = self._stringify(query).lower()
        if not q:
            return []

        candidates = self.recall_all()  # newest-first

        if mode == "exact":
            results = [rec for rec in candidates if (self._stringify(rec.get("summary") or rec.get("content") or "").lower() == q)]
            return results[:k]

        if mode == "fuzzy":
            texts = [self._stringify(rec.get("summary") or rec.get("content") or "") for rec in candidates]
            close = get_close_matches(q, texts, n=min(len(texts), k), cutoff=0.3)
            results = []
            # preserve newest-first
            for rec in candidates:
                txt = self._stringify(rec.get("summary") or rec.get("content") or "")
                if txt in close:
                    results.append(rec)
                    if len(results) >= k:
                        break
            return results

        # keyword scoring
        q_tokens = [t for t in q.split() if t.strip()]
        scored = []
        for rec in candidates:
            txt = self._stringify(rec.get("summary") or rec.get("content") or "").lower()
            score = sum(txt.count(tok) for tok in q_tokens) if q_tokens else 0
            scored.append((score, rec.get("id", 0), rec))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        results = [rec for score, _id, rec in scored if score > 0]
        if not results:
            # fallback most recent
            return candidates[:k]
        return results[:k]

    def snapshot(self) -> Dict[str, Any]:
        items = self.recall_all()
        # next_id read
        try:
            next_id = int(self._r.get(self._id_key) or 0)
        except Exception:
            next_id = 0
        return {"items": items, "next_id": next_id}

    to_dict = snapshot

    # In Redis backend, save/load as operations are not needed (data is already persisted),
    # but we provide compatibility methods.
    def save(self, path: Optional[str] = None) -> None:
        """Dump Redis-backed memory snapshot to local JSON file (one-off export)."""
        p = path or self.cfg.persist_path
        if not p:
            raise ValueError("No persist path provided for save().")
        tmp = f"{p}.tmp"
        data = self.snapshot()
        os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, p)
        logger.debug("Redis-backed memory exported to %s", p)

    def load(self, path: Optional[str] = None) -> None:
        """
        Load items from a JSON file and push them into Redis.
        Existing Redis list is kept; loaded items are pushed to head in order of file listing.
        """
        p = path or self.cfg.persist_path
        if not p or not os.path.exists(p):
            raise ValueError(f"No persisted memory found at: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        items = data.get("items", [])
        if not isinstance(items, list):
            raise ValueError("Persisted memory has invalid format.")
        # push items (oldest last so that newest ends up at head)
        for rec in reversed(items):
            payload = json.dumps(rec, ensure_ascii=False)
            self._r.lpush(self._list_key, payload)
        if self.cfg.max_items:
            self._r.ltrim(self._list_key, 0, self.cfg.max_items - 1)
        logger.debug("Loaded %d items into Redis memory from %s", len(items), p)

    def clear(self) -> None:
        self._r.delete(self._list_key)
        # do not reset id counter to preserve uniqueness across processes
        logger.debug("Cleared Redis-backed memory list %s", self._list_key)

    # Helpers
    @staticmethod
    def _stringify(obj: Any) -> str:
        try:
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            try:
                return str(obj)
            except Exception:
                return ""

    @staticmethod
    def _auto_summary(item: Any) -> str:
        s = RedisMemory._stringify(item)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                for k in ("summary", "text", "content"):
                    if k in parsed and isinstance(parsed[k], str):
                        return parsed[k][:400]
                return json.dumps(parsed)[:400]
        except Exception:
            pass
        return s[:400]


# -------------------------
# In-memory implementation
# -------------------------
class InMemory:
    """
    In-process memory store (previous simple implementation).
    """

    def __init__(self, cfg: MemoryConfig):
        self.cfg = cfg
        self._items: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._next_id = 1
        # Try to load persisted memory on creation if persist_path exists
        if self.cfg.persist_path and os.path.exists(self.cfg.persist_path):
            try:
                self.load(self.cfg.persist_path)
            except Exception as e:
                logger.warning("Failed to load persisted memory from %s: %s", self.cfg.persist_path, e)

    def store(self, item: Any, *, summary: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            rec = {
                "id": self._next_id,
                "timestamp": time.time(),
                "content": item,
                "summary": summary if summary is not None else self._auto_summary(item),
            }
            self._next_id += 1
            self._items.append(rec)
            if self.cfg.max_items and len(self._items) > self.cfg.max_items:
                self._items = self._items[-self.cfg.max_items :]
            if self.cfg.enable_persistence and self.cfg.persist_path:
                try:
                    self.save(self.cfg.persist_path)
                except Exception as e:
                    logger.warning("Failed to persist memory after store: %s", e)
            return rec

    append = store

    def recall_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(reversed(self._items))

    def recall(self, query: Union[str, Any], *, k: int = 5, mode: str = "keyword") -> List[Dict[str, Any]]:
        q = self._stringify(query).lower()
        if not q:
            return []
        with self._lock:
            candidates = [(rec, self._stringify(rec.get("summary") or rec.get("content") or "")) for rec in self._items]

        if mode == "exact":
            matches = [rec for rec, txt in candidates if txt.lower() == q]
            return list(reversed(matches))[:k]

        if mode == "fuzzy":
            texts = [txt for _, txt in candidates]
            close = get_close_matches(q, texts, n=min(len(texts), k), cutoff=0.3)
            results = []
            for rec, txt in reversed(candidates):
                if txt in close and rec not in results:
                    results.append(rec)
                    if len(results) >= k:
                        break
            return results

        q_tokens = [t for t in q.split() if t.strip()]
        scored: List[tuple] = []
        for rec, txt in candidates:
            txt_l = txt.lower()
            score = sum(txt_l.count(tok) for tok in q_tokens) if q_tokens else 0
            scored.append((score, rec["id"], rec))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        results = [rec for score, _id, rec in scored if score > 0]
        if not results:
            with self._lock:
                return list(reversed(self._items))[:k]
        return results[:k]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {"items": list(self._items), "next_id": self._next_id}

    to_dict = snapshot

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.cfg.persist_path
        if not p:
            raise ValueError("No persist path provided.")
        tmp = f"{p}.tmp"
        with self._lock:
            data = self.snapshot()
            os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            os.replace(tmp, p)
        logger.debug("Memory saved to %s", p)

    def load(self, path: Optional[str] = None) -> None:
        p = path or self.cfg.persist_path
        if not p or not os.path.exists(p):
            raise ValueError(f"No persisted memory found at: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        with self._lock:
            items = data.get("items", [])
            if not isinstance(items, list):
                raise ValueError("Persisted memory has invalid format.")
            self._items = items
            self._next_id = int(data.get("next_id", (max((it.get("id", 0) for it in items), default=0) + 1)))
        logger.debug("Memory loaded from %s (%d items)", p, len(self._items))

    def clear(self) -> None:
        with self._lock:
            self._items = []
            self._next_id = 1
        if self.cfg.enable_persistence and self.cfg.persist_path:
            try:
                self.save(self.cfg.persist_path)
            except Exception as e:
                logger.warning("Failed to persist memory after clear: %s", e)

    # Helpers
    @staticmethod
    def _stringify(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            try:
                return str(obj)
            except Exception:
                return ""

    @staticmethod
    def _auto_summary(item: Any) -> str:
        s = InMemory._stringify(item)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                for k in ("summary", "text", "content", "message"):
                    if k in parsed and isinstance(parsed[k], str):
                        return parsed[k][:400]
                return json.dumps(parsed, ensure_ascii=False)[:400]
        except Exception:
            pass
        return s[:400]


# -------------------------
# Facade that chooses backend
# -------------------------
class Memory:
    """
    Facade that chooses RedisMemory if Redis is available and reachable; otherwise InMemory.

    Use the same API as the underlying implementation:
      - store(item, summary=None)
      - append(item, summary=None)
      - recall(query, k=5, mode="keyword")
      - recall_all()
      - snapshot() / to_dict()
      - save(path=None) / load(path=None)
      - clear()
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self._backend: Union[RedisMemory, InMemory]
        # Decision: use Redis only if:
        #  - redis package installed, and
        #  - redis_url is provided and reachable
        if _HAS_REDIS and self.config.redis_url:
            try:
                logger.debug("Attempting to use RedisMemory at %s", self.config.redis_url)
                self._backend = RedisMemory(self.config)
                logger.info("Using Redis-backed Memory at %s", self.config.redis_url)
            except Exception as e:
                logger.warning("RedisMemory unavailable (%s). Falling back to in-memory. Reason: %s", self.config.redis_url, e)
                self._backend = InMemory(self.config)
        else:
            if not _HAS_REDIS and self.config.redis_url:
                logger.info("redis package not installed; falling back to in-memory memory.")
            else:
                logger.debug("No redis_url configured; using in-memory memory.")
            self._backend = InMemory(self.config)

    # Proxy methods
    def store(self, item: Any, *, summary: Optional[str] = None) -> Dict[str, Any]:
        return self._backend.store(item, summary=summary)

    def append(self, item: Any, *, summary: Optional[str] = None) -> Dict[str, Any]:
        return self._backend.append(item, summary=summary)

    def recall(self, query: Union[str, Any], *, k: int = 5, mode: str = "keyword") -> List[Dict[str, Any]]:
        return self._backend.recall(query, k=k, mode=mode)

    def recall_all(self) -> List[Dict[str, Any]]:
        return self._backend.recall_all()

    def snapshot(self) -> Dict[str, Any]:
        return self._backend.snapshot()

    def to_dict(self) -> Dict[str, Any]:
        return self._backend.to_dict()

    def save(self, path: Optional[str] = None) -> None:
        return self._backend.save(path)

    def load(self, path: Optional[str] = None) -> None:
        return self._backend.load(path)

    def clear(self) -> None:
        return self._backend.clear()

    # Expose backend type for diagnostic use
    def backend_name(self) -> str:
        return "redis" if isinstance(self._backend, RedisMemory) else "inmemory"


# -------------------------
# Manual test / demo
# -------------------------
if __name__ == "__main__":
    cfg = MemoryConfig(enable_persistence=False, max_items=100)
    mem = Memory(cfg)
    print("Selected backend:", mem.backend_name())
    mem.store({"event": "opened_file", "path": "README.md"}, summary="Opened README.md")
    mem.store("Ran unit tests", summary="unit tests")
    mem.store("Fixed bug in memory recall", summary="bugfix memory recall")
    print("All:", mem.recall_all())
    print("Recall keyword 'tests':", mem.recall("tests"))
    print("Recall fuzzy 'memroy':", mem.recall("memroy", mode="fuzzy"))
    print("Snapshot:", mem.snapshot())
