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
        "**/node_modules/**", "**/.git/**", "**/.venv/**", "**/venv/**", "**/__pycache__/**",
        "**/dist/**", "**/build/**", "**/.next/**", "**/target/**", "**/out/**",
        "**/.cache/**", "**/.pytest_cache/**", "**/.mypy_cache/**", "**/coverage/**",
        "**/.coverage", "**/.nyc_output/**", "**/logs/**", "**/*.log", "**/.log/**"
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
    - groq_model: Groq model name to use for embeddings (e.g., 'qwen/qwen3-32b')
    - batch_size: how many texts to embed per API call/batch.
    """
    prefer_openai: bool = False
    openai_model: str = "text-embedding-3-small"  # set to a reasonable default
    local_model: str = "BAAI/bge-base-en-v1.5"  # Updated to use BAAI/bge-base-en-v1.5
    groq_model: Optional[str] = None
    batch_size: int = 64


@dataclass
class RerankerConfig:
    """
    Reranker configuration for improving retrieval quality:
    - enabled: whether to use reranking
    - model: sentence-transformers reranker model id
    - top_k: number of results to rerank (retrieve more, rerank, then return top_k)
    """
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Cross-encoder reranker model (can be local path or HF model ID)
    top_k: int = 20  # Retrieve this many, then rerank and return top_k

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
    Top-level RAG configuration combining indexing, embedding, vectorstore, and reranker settings.
    Provides helpers to load from environment and to persist/load config files.
    """
    repo_root: str = "."
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
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
        # Make index path relative to repo_root if it's a relative path
        if not os.path.isabs(self.vectorstore.index_path):
            idx_path = os.path.join(self.repo_root, self.vectorstore.index_path)
        else:
            idx_path = self.vectorstore.index_path
        idx_path = os.path.abspath(idx_path)
        # Update the config with the resolved path
        self.vectorstore.index_path = idx_path
        try:
            os.makedirs(idx_path, exist_ok=True)
        except Exception as e:
            logger.debug("Could not create index path %s: %s", idx_path, e)

    # -------------------------
    # Helpers to create from env or file
    # -------------------------
    @classmethod
    def from_env(cls, repo_root: Optional[str] = None, config_file: Optional[str] = None) -> "RagConfig":
        """
        Build RagConfig from environment variables and/or YAML config file.
        Useful for simple deployments.
        
        Recognized env vars:
          - RAG_INDEX_PATH, RAG_STORE_TYPE, RAG_TOP_K
          - OPENAI_API_KEY (influences prefer_openai)
          - RAG_OPENAI_MODEL, RAG_LOCAL_EMBED_MODEL
        
        Args:
            repo_root: Repository root path
            config_file: Optional path to rag.yaml file. If not provided, tries to find configs/rag.yaml
        """
        repo_root = repo_root or os.environ.get("RAG_REPO_ROOT", ".")
        cfg = cls(repo_root=repo_root)
        
        # Try to load from YAML file first
        if config_file is None:
            # Try to find configs/rag.yaml relative to project root
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            config_file = os.path.join(project_root, "configs", "rag.yaml")
        
        if config_file and os.path.exists(config_file):
            try:
                yaml_cfg = cls.load_from_file(config_file)
                # Merge YAML config (YAML takes precedence)
                cfg = yaml_cfg
                logger.info("Loaded RAG config from %s", config_file)
            except Exception as e:
                logger.debug("Failed to load RAG config from file %s: %s", config_file, e)

        # vectorstore
        store_type = os.environ.get("RAG_STORE_TYPE") or os.environ.get("VECTORSTORE_TYPE")
        if store_type:
            cfg.vectorstore.store_type = store_type

        index_path = os.environ.get("RAG_INDEX_PATH") or os.environ.get("VECTOR_INDEX_PATH")
        if index_path:
            cfg.vectorstore.index_path = index_path

        # Only set prefer_openai from env if YAML didn't explicitly set embed_backend to "local" or "groq"
        # Check if we loaded from YAML and if embed_backend was explicitly set
        yaml_explicitly_set = False
        if config_file and os.path.exists(config_file):
            try:
                import yaml
                with open(config_file, "r", encoding="utf-8") as fh:
                    yaml_data = yaml.safe_load(fh) or {}
                    if "rag" in yaml_data:
                        embed_backend = yaml_data["rag"].get("embed_backend", "auto")
                        # If explicitly set to local, groq, or openai, don't override with env
                        if embed_backend in ("local", "groq", "openai"):
                            yaml_explicitly_set = True
            except Exception:
                pass  # If we can't check, proceed with env override
        
        # Only override with env if YAML didn't explicitly set embed_backend
        if not yaml_explicitly_set and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
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
        Load configuration from a JSON or YAML file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        
        # Try YAML first, then JSON
        try:
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except ImportError:
                # Fall back to JSON if YAML not available
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
        except Exception as e:
            logger.exception("Failed to read config file %s: %s", path, e)
            raise

        # Handle YAML format (rag.yaml) vs JSON format
        # YAML format has top-level keys like: enabled, index_path, store_type, etc.
        # JSON format has nested structure: indexing, embedding, vectorstore
        
        if "rag" in data:
            # YAML format with 'rag' wrapper
            rag_data = data["rag"]
        elif "indexing" in data or "embedding" in data:
            # JSON format (nested structure)
            rag_data = data
        else:
            # Assume YAML format (flat structure)
            rag_data = data
        
        # Map dict to dataclasses carefully
        def _map_indexing(d: Dict) -> IndexingConfig:
            return IndexingConfig(**{k: v for k, v in d.items() if k in IndexingConfig.__dataclass_fields__})

        def _map_embedding(d: Dict) -> EmbeddingConfig:
            return EmbeddingConfig(**{k: v for k, v in d.items() if k in EmbeddingConfig.__dataclass_fields__})

        def _map_vectorstore(d: Dict) -> VectorStoreConfig:
            return VectorStoreConfig(**{k: v for k, v in d.items() if k in VectorStoreConfig.__dataclass_fields__})

        def _map_reranker(d: Dict) -> RerankerConfig:
            return RerankerConfig(**{k: v for k, v in d.items() if k in RerankerConfig.__dataclass_fields__})

        # Handle YAML flat format (from rag.yaml) vs JSON nested format
        if "indexing" not in rag_data and "embedding" not in rag_data:
            # YAML flat format - map to nested structure
            # Check if RAG is enabled
            enabled = rag_data.get("enabled", True)
            if not enabled:
                # Return a minimal disabled config
                return cls(
                    repo_root=rag_data.get("repo_root", "."),
                    top_k=0,
                    use_hybrid_retrieval=False
                )
            
            indexing_data = {
                "chunk_size_tokens": rag_data.get("chunk_size_tokens", 500),
                "chunk_overlap_tokens": rag_data.get("chunk_overlap_tokens", 80),
            }
            embed_backend = rag_data.get("embed_backend", "auto")
            # Handle embed_backend options: "openai", "local", "groq", or "auto"
            if embed_backend == "openai":
                prefer_openai = True
                groq_model = None  # Don't use Groq when OpenAI is explicitly set
            elif embed_backend == "groq":
                # Use Groq LLM models for embeddings (workaround since Groq doesn't have native embeddings)
                prefer_openai = False
                groq_model = rag_data.get("groq_model")  # Use Groq model if specified
            elif embed_backend == "local":
                # Use local sentence-transformers embeddings
                prefer_openai = False
                groq_model = None  # Explicitly disable Groq when local is set
            else:  # "auto"
                # Auto-detect: prefer OpenAI if API key is present, otherwise use local
                prefer_openai = (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")) is not None
                groq_model = rag_data.get("groq_model") if not prefer_openai else None
            
            embedding_data = {
                "prefer_openai": prefer_openai,
                "openai_model": rag_data.get("openai_model", "text-embedding-3-small"),
                "local_model": rag_data.get("local_model", "BAAI/bge-base-en-v1.5"),
                "groq_model": groq_model,  # Use the determined value (None for local/openai, or actual value for groq)
            }
            vectorstore_data = {
                "store_type": rag_data.get("store_type", "faiss"),
                "index_path": rag_data.get("index_path", ".agent_index"),
                "redis_url": rag_data.get("redis_url"),
                "metric": rag_data.get("metric", "cosine"),
            }
            reranker_data = {
                "enabled": rag_data.get("reranker_enabled", True),
                "model": rag_data.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
                "top_k": rag_data.get("reranker_top_k", 20),
            }
            repo_root = rag_data.get("repo_root", ".")
            top_k = int(rag_data.get("top_k", 8))
            max_chunk_chars = int(rag_data.get("max_chunk_chars", 3000))
            use_hybrid = bool(rag_data.get("use_hybrid_retrieval", True))
            expand_refs = bool(rag_data.get("expand_file_refs", True))
            idx_version = rag_data.get("index_version")
            
            # Create config objects from YAML flat format
            indexing = _map_indexing(indexing_data)
            embedding = _map_embedding(embedding_data)
            vectorstore = _map_vectorstore(vectorstore_data)
            reranker = _map_reranker(reranker_data)
        else:
            # JSON nested format
            repo_root = rag_data.get("repo_root", ".")
            indexing = _map_indexing(rag_data.get("indexing", {}))
            embedding = _map_embedding(rag_data.get("embedding", {}))
            vectorstore = _map_vectorstore(rag_data.get("vectorstore", {}))
            reranker = _map_reranker(rag_data.get("reranker", {}))
            top_k = int(rag_data.get("top_k", 8))
            max_chunk_chars = int(rag_data.get("max_chunk_chars", 3000))
            use_hybrid = bool(rag_data.get("use_hybrid_retrieval", True))
            expand_refs = bool(rag_data.get("expand_file_refs", True))
            idx_version = rag_data.get("index_version")
        
        return cls(
            repo_root=repo_root,
            indexing=indexing,
            embedding=embedding,
            vectorstore=vectorstore,
            reranker=reranker,
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
            "reranker": self.reranker.to_dict(),
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
        Return 'openai', 'local', or 'groq' depending on prefer_openai and environment.
        Note: 'groq' backend uses local embeddings since Groq doesn't provide native embedding models.
        """
        if self.embedding.prefer_openai and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
            return "openai"
        # Check if groq backend was requested (stored in a way we can detect)
        # For now, groq uses local embeddings, so return "local" or "groq" based on context
        return "local"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "indexing": self.indexing.to_dict(),
            "embedding": self.embedding.to_dict(),
            "vectorstore": self.vectorstore.to_dict(),
            "reranker": self.reranker.to_dict(),
            "top_k": self.top_k,
            "max_chunk_chars": self.max_chunk_chars,
            "use_hybrid_retrieval": self.use_hybrid_retrieval,
            "expand_file_refs": self.expand_file_refs,
            "index_version": self.index_version,
        }

    def __repr__(self) -> str:
        return f"RagConfig(repo_root={self.repo_root}, store={self.vectorstore.store_type}, embed_backend={self.choose_embedding_backend()}, top_k={self.top_k})"
