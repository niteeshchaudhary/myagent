# agent/rag/config.py
"""
RAG configuration dataclasses and helpers.

This module centralizes configuration for the RAG pipeline:
- indexing (chunk size, overlap, file filters)
- embedding (model selection, openai vs local)
- vectorstore options (faiss/chroma/redis)
- storage paths for index and metadata

The config can be constructed programmatically, loaded from a YAML/JSON file,
or populated from environment variables (useful when running in CI / containers).
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# prefer project logger if available
try:
    from agent.utils.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)


@dataclass
class IndexingConfig:
    """
    Settings for chunking and indexing code files.
    - chunk_size_tokens: approximate target size of each chunk (tokens). If using a simple
                         whitespace tokenizer this is interpreted as approx chars/tokens.
    - chunk_overlap_tokens: overlap between neighboring chunks (helps retrieval continuity).
    - file_globs: list of glob patterns to include (empty -> include common code files).
    - exclude_globs: list of glob patterns to exclude (e.g. tests, node_modules).
    - language_priority: optional list of languages to prioritize when chunking by function.
    """
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 80
    file_globs: List[str] = field(default_factory=lambda: [
        "**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.go", "**/*.rs", "**/*.c", "**/*.cpp",
        "**/*.html", "**/*.css", "**/*.json", "**/*.md"
    ])
    exclude_globs: List[str] = field(default_factory=lambda: [
        "**/node_modules/**", "**/.git/**", "**/.venv/**", "**/venv/**", "**/__pycache__/**"
    ])
    language_priority: List[str] = field(default_factory=lambda: ["python", "javascript", "java", "go"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingConfig:
    """
    Embedding configuration:
    - prefer_openai: if True and OPENAI_API_KEY present, use OpenAI embeddings by default.
    - openai_model: OpenAI embedding model (if using OpenAI). Example: 'text-embedding-3-small'
    - local_model: sentence-transformers model id to use as local fallback.
    - batch_size: how many texts to embed per API call/batch.
    """
    prefer_openai: bool = False
    openai_model: str = "text-embedding-3-small"  # set to a reasonable default
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VectorStoreConfig:
    """
    Configuration for the backing vector store.
    - store_type: one of 'faiss', 'chroma', 'redis', 'inmemory'
    - index_path: path on disk where the vector index (and metadata) will be saved (for disk-backed stores)
    - redis_url: if using redis vector, connection string
    - metric: distance metric (e.g., 'cosine', 'l2')
    """
    store_type: str = "faiss"
    index_path: str = ".agent_index"
    redis_url: Optional[str] = None
    metric: str = "cosine"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RagConfig:
    """
    Top-level RAG configuration combining indexing, embedding, and vectorstore settings.
    Provides helpers to load from environment and to persist/load config files.
    """
    repo_root: str = "."
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    top_k: int = 8
    max_chunk_chars: int = 3000  # safety cutoff when sending a chunk to LLM
    # whether to prefer exact (ripgrep/tags) results before semantic retrieval
    use_hybrid_retrieval: bool = True
    # whether to include `@file` expansions automatically when forming prompts
    expand_file_refs: bool = True
    # commit hash or index version marker (optional, set by indexer)
    index_version: Optional[str] = None

    def __post_init__(self):
        # normalize repo_root and ensure index path exists
        self.repo_root = os.path.abspath(self.repo_root or ".")
        idx_path = os.path.abspath(self.vectorstore.index_path)
        try:
            os.makedirs(idx_path, exist_ok=True)
        except Exception as e:
            logger.debug("Could not create index path %s: %s", idx_path, e)

    # -------------------------
    # Helpers to create from env or file
    # -------------------------
    @classmethod
    def from_env(cls, repo_root: Optional[str] = None) -> "RagConfig":
        """
        Build RagConfig from environment variables. Useful for simple deployments.
        Recognized env vars:
          - RAG_INDEX_PATH, RAG_STORE_TYPE, RAG_TOP_K
          - OPENAI_API_KEY (influences prefer_openai)
          - RAG_OPENAI_MODEL, RAG_LOCAL_EMBED_MODEL
        """
        repo_root = repo_root or os.environ.get("RAG_REPO_ROOT", ".")
        cfg = cls(repo_root=repo_root)

        # vectorstore
        store_type = os.environ.get("RAG_STORE_TYPE") or os.environ.get("VECTORSTORE_TYPE")
        if store_type:
            cfg.vectorstore.store_type = store_type

        index_path = os.environ.get("RAG_INDEX_PATH") or os.environ.get("VECTOR_INDEX_PATH")
        if index_path:
            cfg.vectorstore.index_path = index_path

        if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY"):
            cfg.embedding.prefer_openai = True

        oa_model = os.environ.get("RAG_OPENAI_MODEL")
        if oa_model:
            cfg.embedding.openai_model = oa_model

        local_model = os.environ.get("RAG_LOCAL_EMBED_MODEL")
        if local_model:
            cfg.embedding.local_model = local_model

        top_k = os.environ.get("RAG_TOP_K")
        if top_k:
            try:
                cfg.top_k = int(top_k)
            except Exception:
                logger.debug("Invalid RAG_TOP_K value: %s", top_k)

        # chunk sizes (optional overrides)
        cs = os.environ.get("RAG_CHUNK_SIZE_TOKENS")
        if cs:
            try:
                cfg.indexing.chunk_size_tokens = int(cs)
            except Exception:
                logger.debug("Invalid RAG_CHUNK_SIZE_TOKENS: %s", cs)
        co = os.environ.get("RAG_CHUNK_OVERLAP")
        if co:
            try:
                cfg.indexing.chunk_overlap_tokens = int(co)
            except Exception:
                logger.debug("Invalid RAG_CHUNK_OVERLAP: %s", co)

        # redis url
        rurl = os.environ.get("REDIS_URL")
        if rurl:
            cfg.vectorstore.redis_url = rurl

        return cfg

    @classmethod
    def load_from_file(cls, path: str) -> "RagConfig":
        """
        Load configuration from a JSON or simple dict file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            logger.exception("Failed to read config file %s: %s", path, e)
            raise

        # Map dict to dataclasses carefully
        def _map_indexing(d: Dict) -> IndexingConfig:
            return IndexingConfig(**{k: v for k, v in d.items() if k in IndexingConfig.__dataclass_fields__})

        def _map_embedding(d: Dict) -> EmbeddingConfig:
            return EmbeddingConfig(**{k: v for k, v in d.items() if k in EmbeddingConfig.__dataclass_fields__})

        def _map_vectorstore(d: Dict) -> VectorStoreConfig:
            return VectorStoreConfig(**{k: v for k, v in d.items() if k in VectorStoreConfig.__dataclass_fields__})

        repo_root = data.get("repo_root", ".")
        indexing = _map_indexing(data.get("indexing", {}))
        embedding = _map_embedding(data.get("embedding", {}))
        vectorstore = _map_vectorstore(data.get("vectorstore", {}))
        top_k = int(data.get("top_k", 8))
        max_chunk_chars = int(data.get("max_chunk_chars", 3000))
        use_hybrid = bool(data.get("use_hybrid_retrieval", True))
        expand_refs = bool(data.get("expand_file_refs", True))
        idx_version = data.get("index_version")

        return cls(
            repo_root=repo_root,
            indexing=indexing,
            embedding=embedding,
            vectorstore=vectorstore,
            top_k=top_k,
            max_chunk_chars=max_chunk_chars,
            use_hybrid_retrieval=use_hybrid,
            expand_file_refs=expand_refs,
            index_version=idx_version,
        )

    def save_to_file(self, path: str) -> None:
        """
        Save the current config to a JSON file.
        """
        data = {
            "repo_root": self.repo_root,
            "indexing": self.indexing.to_dict(),
            "embedding": self.embedding.to_dict(),
            "vectorstore": self.vectorstore.to_dict(),
            "top_k": self.top_k,
            "max_chunk_chars": self.max_chunk_chars,
            "use_hybrid_retrieval": self.use_hybrid_retrieval,
            "expand_file_refs": self.expand_file_refs,
            "index_version": self.index_version,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("RAG config saved to %s", path)

    # -------------------------
    # Utility helpers
    # -------------------------
    def choose_embedding_backend(self) -> str:
        """
        Return 'openai' or 'local' depending on prefer_openai and environment.
        """
        if self.embedding.prefer_openai and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
            return "openai"
        return "local"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "indexing": self.indexing.to_dict(),
            "embedding": self.embedding.to_dict(),
            "vectorstore": self.vectorstore.to_dict(),
            "top_k": self.top_k,
            "max_chunk_chars": self.max_chunk_chars,
            "use_hybrid_retrieval": self.use_hybrid_retrieval,
            "expand_file_refs": self.expand_file_refs,
            "index_version": self.index_version,
        }

    def __repr__(self) -> str:
        return f"RagConfig(repo_root={self.repo_root}, store={self.vectorstore.store_type}, embed_backend={self.choose_embedding_backend()}, top_k={self.top_k})"
