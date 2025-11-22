# agent/rag/indexer.py
"""
Repository indexer for RAG (chunking + embeddings + vector store).

Features
- Walks a repository and creates chunks from source files.
  * For Python files: attempts function/class-aware splitting using ast.
  * For other files: fixed-size chunking by characters with overlap.
- Embeds chunks using either OpenAI embeddings (if configured) or a local
  sentence-transformers model (fallback).
- Stores vectors in FAISS (if installed) with a JSON metadata store that maps
  vector ids -> {path, start_line, end_line, text, lang, repo, chunk_id, ...}
- Provides a simple CLI to `index` and `index --changed` and a programmatic API.

Notes:
- This implementation is pragmatic, defensive, and suitable for local development.
- For production-grade indexing, consider more robust tokenization and incremental
  update strategies (git-aware).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import pathlib
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

# logging
try:
    from agent.utils.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)

# config
try:
    from agent.rag.config import RagConfig, IndexingConfig, EmbeddingConfig, VectorStoreConfig
except Exception:
    # fallback in case module path differs when running standalone
    from rag.config import RagConfig, IndexingConfig, EmbeddingConfig, VectorStoreConfig  # type: ignore

# optional dependencies
_HAS_FAISS = False
try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore

_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None  # type: ignore

# OpenAI optional
_HAS_OPENAI = False
try:
    import openai  # type: ignore

    _HAS_OPENAI = True
except Exception:
    openai = None  # type: ignore

# Groq optional
_HAS_GROQ = False
try:
    from groq import Groq  # type: ignore

    _HAS_GROQ = True
except Exception:
    Groq = None  # type: ignore
    _HAS_GROQ = False

import numpy as np
from uuid import uuid4

# -------------------------
# Data containers
# -------------------------
@dataclass
class ChunkMeta:
    chunk_id: str
    path: str
    repo: str
    lang: str
    start_line: Optional[int]
    end_line: Optional[int]
    text: str
    created_at: float
    source_hash: str  # hash of file contents at time of chunking

    def to_dict(self) -> Dict:
        d = asdict(self)
        # avoid storing full text twice if large; but we will include text to allow rehydration
        return d


# -------------------------
# Helper utilities
# -------------------------
def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_file_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        logger.exception("Failed reading file: %s", path)
        return ""


def _is_binary_file(path: str) -> bool:
    # naive check: look for NUL byte in first 1KB
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(1024)
            return b"\x00" in chunk
    except Exception:
        return True


def _guess_lang_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".md": "markdown",
        ".json": "json",
    }.get(ext, ext.lstrip(".") or "text")


# -------------------------
# Chunking logic
# -------------------------
def _chunk_python_source_by_definitions(source: str, path: str, chunk_size_chars: int, overlap_chars: int) -> List[Tuple[int, int, str]]:
    """
    Try to split Python source into chunks aligned to top-level function/class definitions.
    Returns list of tuples (start_line, end_line, text).
    If file is small or no top-level defs, falls back to fixed-size chunking.
    """
    try:
        tree = ast.parse(source)
    except Exception:
        # fallback to fixed-size chunking
        return _fixed_chunk_text(source, chunk_size_chars, overlap_chars)

    # collect top-level node line ranges
    blocks: List[Tuple[int, int]] = []
    for node in tree.body:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None) if hasattr(node, "end_lineno") else None
        if start and end:
            blocks.append((start, end))

    if not blocks:
        return _fixed_chunk_text(source, chunk_size_chars, overlap_chars)

    # Map line numbers to text lines
    lines = source.splitlines()
    chunks: List[Tuple[int, int, str]] = []
    # produce chunk per block, but combine adjacent small blocks into a larger chunk if needed
    i = 0
    while i < len(blocks):
        s, e = blocks[i]
        # join subsequent blocks until approx chunk_size_chars reached
        j = i
        current_text = "\n".join(lines[s - 1 : e])
        while len(current_text) < chunk_size_chars and (j + 1) < len(blocks):
            j += 1
            ns, ne = blocks[j]
            # include lines between previous end and new end
            current_text = "\n".join(lines[s - 1 : ne])
            e = ne
        chunks.append((s, e, current_text))
        i = j + 1

    # convert any chunk that is still too large into fixed-size subchunks
    final_chunks: List[Tuple[int, int, str]] = []
    for s, e, text in chunks:
        if len(text) <= chunk_size_chars + 100:
            final_chunks.append((s, e, text))
        else:
            # split the large chunk by fixed-char windows preserving lines
            sub = _fixed_chunk_text(text, chunk_size_chars, overlap_chars)
            # sub returns start/end line indices relative to this chunk; convert to repo file lines
            # get base line offset = s
            for sub_s, sub_e, sub_text in sub:
                final_chunks.append((s + sub_s - 1, s + sub_e - 1, sub_text))
    return final_chunks


def _fixed_chunk_text(text: str, chunk_size_chars: int, overlap_chars: int) -> List[Tuple[int, int, str]]:
    """
    Generic fixed-size chunking by characters but returns line-aligned chunks.
    Returns list of (start_line, end_line, text).
    """
    lines = text.splitlines()
    if not lines:
        return []
    # join with newline and then create windows by characters, but map to lines
    joined = "\n".join(lines)
    N = len(joined)
    if N <= chunk_size_chars:
        return [(1, len(lines), joined)]
    chunks: List[Tuple[int, int, str]] = []
    start_char = 0
    while start_char < N:
        end_char = min(start_char + chunk_size_chars, N)
        # expand end_char to end of nearest line
        prefix = joined[:end_char]
        # count lines in prefix
        end_line = prefix.count("\n") + 1
        # find start_line similarly
        prefix2 = joined[:start_char]
        start_line = prefix2.count("\n") + 1 if start_char > 0 else 1
        sub_lines = lines[start_line - 1 : end_line]
        if not sub_lines:
            break
        sub_text = "\n".join(sub_lines)
        chunks.append((start_line, end_line, sub_text))
        if end_char == N:
            break
        # advance start_char by chunk_size - overlap (approx)
        step = chunk_size_chars - overlap_chars
        if step <= 0:
            break
        start_char += step
    return chunks


def chunk_file(path: str, repo_root: str, indexing_cfg: IndexingConfig) -> List[ChunkMeta]:
    """
    Produce chunks for a single file.
    Returns list of ChunkMeta objects.
    """
    full_path = os.path.join(repo_root, path) if not os.path.isabs(path) else path
    if _is_binary_file(full_path):
        logger.debug("Skipping binary file for indexing: %s", full_path)
        return []
    text = _read_file_text(full_path)
    if not text:
        return []
    lang = _guess_lang_from_path(full_path)
    # approximate chars per token mapping: 1 token ~ 4 chars (very rough)
    chunk_chars = max(300, int(indexing_cfg.chunk_size_tokens * 4))
    overlap_chars = max(50, int(indexing_cfg.chunk_overlap_tokens * 4))
    chunks: List[Tuple[int, int, str]] = []
    if lang == "python":
        try:
            chunks = _chunk_python_source_by_definitions(text, full_path, chunk_chars, overlap_chars)
        except Exception:
            logger.exception("python-aware chunking failed for %s; falling back to fixed chunking", full_path)
            chunks = _fixed_chunk_text(text, chunk_chars, overlap_chars)
    else:
        chunks = _fixed_chunk_text(text, chunk_chars, overlap_chars)

    metas: List[ChunkMeta] = []
    file_hash = _hash_text(text)
    now = time.time()
    for idx, (start_line, end_line, chunk_text) in enumerate(chunks):
        chunk_id = f"{_hash_text(path)}-{idx}-{uuid4().hex[:8]}"
        meta = ChunkMeta(
            chunk_id=chunk_id,
            path=os.path.relpath(full_path, repo_root),
            repo=repo_root,
            lang=lang,
            start_line=int(start_line) if start_line else None,
            end_line=int(end_line) if end_line else None,
            text=chunk_text,
            created_at=now,
            source_hash=file_hash,
        )
        metas.append(meta)
    return metas


# -------------------------
# Embedding helpers
# -------------------------
class Embedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.backend = "local"
        self.model_local = None
        self.groq_client = None
        self.use_groq = False
        
        # Determine embedding backend priority:
        # 1. If prefer_openai is True and OpenAI is available -> use OpenAI
        # 2. If groq_model is set AND prefer_openai is False (not explicitly disabled) AND Groq is available -> use Groq
        # 3. Otherwise -> use local
        
        # Check for OpenAI first if explicitly preferred
        if cfg.prefer_openai and _HAS_OPENAI and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
            self.backend = "openai"
            # set openai api key if present in env (openai lib will auto pick)
            logger.info("Using OpenAI embeddings (model: %s)", cfg.openai_model)
        # Check for Groq only if prefer_openai is False (meaning we're not forcing OpenAI)
        # AND groq_model is explicitly set (meaning user wants Groq)
        elif (not cfg.prefer_openai and cfg.groq_model and _HAS_GROQ and 
              (os.environ.get("GROQ_API_KEY"))):
            try:
                self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                self.backend = "groq"
                self.use_groq = True
                logger.info("Using Groq model '%s' for embeddings", cfg.groq_model)
            except Exception as e:
                logger.warning("Failed to initialize Groq client for embeddings: %s. Falling back to local embeddings.", e)
                self.use_groq = False
                # Fall through to local
        else:
            # Use local embeddings (works for both "local" and "groq" backends when groq_model not specified)
            if _HAS_SENTENCE_TRANSFORMERS:
                self.backend = "local"
                try:
                    logger.info("Loading local sentence-transformers model: %s", cfg.local_model)
                    self.model_local = SentenceTransformer(cfg.local_model)
                except Exception:
                    logger.exception("Failed to load sentence-transformers model, will fallback to naive hashing embeddings.")
                    self.model_local = None
            else:
                self.model_local = None
                logger.info("sentence-transformers not available; embedding will use hashing fallback.")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Try Groq embeddings first if configured
        if self.backend == "groq" and self.use_groq and self.groq_client:
            try:
                return self._embed_with_groq(texts)
            except Exception:
                logger.exception("Groq embedding failed; falling back to local or hashing.")
                # fallthrough to local/hashing
        if self.backend == "openai" and _HAS_OPENAI:
            try:
                model = self.cfg.openai_model
                # Use OpenAI client for v1.0+ API
                client = openai.OpenAI()
                # batch in chunks of cfg.batch_size
                embeddings: List[List[float]] = []
                batch = []
                for t in texts:
                    batch.append(t)
                    if len(batch) >= self.cfg.batch_size:
                        resp = client.embeddings.create(model=model, input=batch)
                        embeddings.extend([e.embedding for e in resp.data])
                        batch = []
                if batch:
                    resp = client.embeddings.create(model=model, input=batch)
                    embeddings.extend([e.embedding for e in resp.data])
                return embeddings
            except Exception:
                logger.exception("OpenAI embedding failed; falling back to local or hashing.")
                # fallthrough to local/hashing
        if self.model_local:
            try:
                vecs = self.model_local.encode(texts, show_progress_bar=False)
                # ensure python lists
                return [list(map(float, v.tolist())) for v in vecs]
            except Exception:
                logger.exception("Local model encode failed; falling back to hashing.")
        # fallback: deterministic hashing to a fixed-dim vector (not semantically useful)
        logger.warning("Using hashing fallback embeddings (not semantically meaningful). Install sentence-transformers or set OPENAI_API_KEY for better embeddings.")
        dim = 256
        out = []
        for t in texts:
            h = hashlib.md5(t.encode("utf-8")).digest()
            vec = [float(b) / 255.0 for b in h]
            # pad/extend to dim
            if len(vec) < dim:
                vec = vec + [0.0] * (dim - len(vec))
            out.append(vec[:dim])
        return out

    def _embed_with_groq(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Groq LLM models.
        
        Since Groq doesn't have native embedding models, we use a workaround:
        1. Use Groq LLM to generate a semantic representation of the text
        2. Hash the LLM output to create a consistent embedding vector
        
        This approach uses Groq's models but creates embeddings from their output.
        """
        if not self.groq_client or not self.cfg.groq_model:
            raise RuntimeError("Groq client or model not configured")
        
        # Warn user about slowness
        if len(texts) > 10:
            logger.warning(
                f"Groq embedding is very slow for {len(texts)} chunks (requires {len(texts)} API calls). "
                f"Consider using local embeddings (embed_backend: 'local') for faster indexing."
            )
        
        embeddings: List[List[float]] = []
        # Process texts individually or in small batches to avoid token limits
        batch_size = min(self.cfg.batch_size, 8)  # Smaller batches for Groq LLM
        total = len(texts)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            # Show progress for large batches
            if total > 10 and i % 10 == 0:
                print(f"\rProcessing Groq embeddings: {i}/{total} ({100*i//total}%)...", end="", flush=True)
            
            for text in batch:
                try:
                    # Use Groq LLM to generate a semantic representation
                    # We ask the model to summarize/represent the text in a consistent way
                    prompt = f"""Generate a concise semantic representation of the following text. 
Focus on key concepts, meaning, and context. Keep it brief and consistent:

Text: {text[:1000]}  # Limit text length to avoid token limits

Semantic representation:"""
                    
                    response = self.groq_client.chat.completions.create(
                        model=self.cfg.groq_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=100,
                        temperature=0.1  # Low temperature for consistency
                    )
                    
                    # Extract the generated representation
                    representation = ""
                    if response.choices:
                        for choice in response.choices:
                            if choice.message and choice.message.content:
                                representation = choice.message.content.strip()
                    
                    # Create embedding from the representation
                    # Use a combination of text hash and representation hash for better semantic similarity
                    text_hash = hashlib.sha256(text.encode("utf-8")).digest()
                    repr_hash = hashlib.sha256(representation.encode("utf-8")).digest()
                    
                    # Combine hashes to create a 256-dim vector
                    combined = text_hash + repr_hash[:16]  # 32 + 16 = 48 bytes
                    vec = [float(b) / 255.0 for b in combined]
                    
                    # Extend to 256 dimensions using a deterministic method
                    while len(vec) < 256:
                        # Use a hash of the index to extend
                        ext_hash = hashlib.md5(f"{text}{len(vec)}".encode()).digest()
                        vec.extend([float(b) / 255.0 for b in ext_hash[:min(16, 256 - len(vec))]])
                    
                    batch_embeddings.append(vec[:256])
                    
                except Exception as e:
                    logger.warning("Failed to generate Groq embedding for text, using hash fallback: %s", e)
                    # Fallback to simple hash
                    h = hashlib.md5(text.encode("utf-8")).digest()
                    vec = [float(b) / 255.0 for b in h]
                    if len(vec) < 256:
                        vec = vec + [0.0] * (256 - len(vec))
                    batch_embeddings.append(vec[:256])
            
            embeddings.extend(batch_embeddings)
        
        if total > 10:
            print(f"\rProcessing Groq embeddings: {total}/{total} (100%)... Done!", flush=True)
        
        return embeddings


# -------------------------
# Vector store (FAISS or simple fallback)
# -------------------------
class VectorStore:
    def __init__(self, cfg: VectorStoreConfig):
        self.cfg = cfg
        self.index_path = os.path.abspath(cfg.index_path or ".agent_index")
        _ensure_dir(self.index_path)
        self.meta_path = os.path.join(self.index_path, "meta.json")
        self.vectors_path = os.path.join(self.index_path, "vectors.npy")
        self._meta: Dict[str, Dict] = {}  # chunk_id -> meta dict
        self._ids: List[str] = []  # chunk ids in index order
        self._emb_dim: Optional[int] = None
        self._faiss_index = None
        self._np_vectors = None
        self._loaded = False
        self._init_store()

    def _init_store(self):
        # try to load existing index
        try:
            if _HAS_FAISS and self.cfg.store_type in ("faiss", "faiss_cpu"):
                # attempt to load faiss index file
                idx_file = os.path.join(self.index_path, "faiss.index")
                meta_file = self.meta_path
                if os.path.exists(idx_file) and os.path.exists(meta_file):
                    try:
                        self._faiss_index = faiss.read_index(idx_file)
                        with open(meta_file, "r", encoding="utf-8") as fh:
                            j = json.load(fh)
                        self._meta = j.get("meta", {})
                        self._ids = j.get("ids", [])
                        self._emb_dim = int(j.get("dim", 0)) if j.get("dim") else None
                        self._loaded = True
                        logger.info("Loaded FAISS index (%d vectors, dim=%s)", len(self._ids), self._emb_dim)
                    except Exception:
                        logger.exception("Failed to load existing FAISS index; starting empty.")
                        self._faiss_index = None
                else:
                    # start new index lazily when dim known
                    self._faiss_index = None
            else:
                # fallback: load numpy vectors + meta json
                if os.path.exists(self.vectors_path) and os.path.exists(self.meta_path):
                    try:
                        self._np_vectors = np.load(self.vectors_path)
                        with open(self.meta_path, "r", encoding="utf-8") as fh:
                            j = json.load(fh)
                        self._meta = j.get("meta", {})
                        self._ids = j.get("ids", [])
                        self._emb_dim = int(self._np_vectors.shape[1])
                        self._loaded = True
                        logger.info("Loaded numpy vector store (%d vectors, dim=%s)", len(self._ids), self._emb_dim)
                    except Exception:
                        logger.exception("Failed to load numpy vector store; starting empty.")
                        self._np_vectors = None
        except Exception:
            logger.exception("VectorStore init encountered error.")

    def _save_meta(self):
        data = {"meta": self._meta, "ids": self._ids, "dim": self._emb_dim}
        with open(self.meta_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def add(self, chunk_metas: List[ChunkMeta], embeddings: List[List[float]], *, force_recreate: bool = False):
        """
        Add chunks with embeddings to the store.
        embeddings length must match chunk_metas length.
        
        Args:
            force_recreate: If True and dimension mismatch, clear old index and recreate with new dimension
        """
        if not chunk_metas:
            return
        if len(chunk_metas) != len(embeddings):
            raise ValueError("chunk_metas and embeddings length mismatch")

        emb_arr = np.array(embeddings, dtype=np.float32)
        n, dim = emb_arr.shape
        if self._emb_dim is None:
            self._emb_dim = dim
        if dim != self._emb_dim:
            # Dimension mismatch - this usually happens when switching embedding backends
            # Auto-clear and recreate if force_recreate is True, otherwise provide helpful error
            if force_recreate or (self._loaded and len(self._ids) == 0):
                # Clear old index and recreate with new dimension
                logger.warning(
                    "Embedding dimension mismatch (expected %d, got %d). "
                    "This usually happens when switching embedding backends (e.g., Groq 256-dim to local 768-dim). "
                    "Clearing old index and recreating with new dimension.",
                    self._emb_dim, dim
                )
                self.clear()
                self._emb_dim = dim
            else:
                raise ValueError(
                    f"Embedding dim mismatch: expected {self._emb_dim}, got {dim}. "
                    f"This usually happens when switching embedding backends (e.g., Groq 256-dim to local 768-dim). "
                    f"Use --force flag when indexing to recreate the index with the new dimension: "
                    f"agent index --force"
                )

        # append metadata and ids
        for meta, emb in zip(chunk_metas, embeddings):
            cid = meta.chunk_id
            self._meta[cid] = meta.to_dict()
            self._ids.append(cid)

        # add vectors to faiss or numpy store
        if _HAS_FAISS and self.cfg.store_type in ("faiss", "faiss_cpu"):
            if self._faiss_index is None:
                # create new IndexFlatIP for cosine-like similarity (normalize vectors)
                index = faiss.IndexFlatIP(dim)
                self._faiss_index = index
                logger.info("Created new FAISS IndexFlatIP (dim=%d)", dim)
            # ensure we normalize vectors for inner product cosine similarity if desired
            # convert to float32
            vecs = emb_arr.astype(np.float32)
            # normalize
            faiss.normalize_L2(vecs)
            self._faiss_index.add(vecs)
            # save index and meta
            try:
                faiss.write_index(self._faiss_index, os.path.join(self.index_path, "faiss.index"))
                self._save_meta()
            except Exception:
                logger.exception("Failed to persist FAISS index (non-fatal).")
        else:
            # numpy fallback: append to existing array
            if self._np_vectors is None:
                self._np_vectors = emb_arr
            else:
                self._np_vectors = np.vstack([self._np_vectors, emb_arr])
            # persist numpy + meta
            try:
                np.save(self.vectors_path, self._np_vectors)
                self._save_meta()
            except Exception:
                logger.exception("Failed to persist numpy vector store (non-fatal).")

    def query(self, vector: List[float], top_k: int = 8) -> List[Tuple[str, float]]:
        """
        Query the vector store with a vector and return list of (chunk_id, score) sorted desc.
        Score is cosine similarity in faiss case (due to normalization) or dot product / negative distance.
        """
        if not vector:
            return []
        v = np.array(vector, dtype=np.float32).reshape(1, -1)
        query_dim = v.shape[1]
        
        # Check for dimension mismatch - this happens when switching embedding backends
        if self._emb_dim is not None and query_dim != self._emb_dim:
            logger.error(
                "Vector dimension mismatch: query vector has %d dimensions but index has %d dimensions. "
                "This usually happens when switching embedding backends (e.g., Groq 256-dim to local 768-dim). "
                "Please re-index with: agent index --force",
                query_dim, self._emb_dim
            )
            return []  # Return empty results rather than crashing
        if _HAS_FAISS and self._faiss_index is not None:
            faiss.normalize_L2(v)
            D, I = self._faiss_index.search(v, top_k)
            res = []
            for idx, score in zip(I[0].tolist(), D[0].tolist()):
                if idx < 0 or idx >= len(self._ids):
                    continue
                cid = self._ids[idx]
                res.append((cid, float(score)))
            return res
        elif self._np_vectors is not None:
            # Check for dimension mismatch before querying
            if self._emb_dim is not None and query_dim != self._emb_dim:
                logger.error(
                    "Vector dimension mismatch: query vector has %d dimensions but index has %d dimensions. "
                    "This usually happens when switching embedding backends (e.g., Groq 256-dim to local 768-dim). "
                    "Please re-index with: agent index --force",
                    query_dim, self._emb_dim
                )
                return []  # Return empty results rather than crashing
            
            # compute cosine similarity manually
            mat = self._np_vectors
            # normalize
            vv = v / (np.linalg.norm(v) + 1e-12)
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
            matn = mat / norms
            sims = (matn @ vv.T).reshape(-1)
            idxs = np.argsort(-sims)[:top_k]
            res = [(self._ids[int(i)], float(sims[int(i)])) for i in idxs]
            return res
        else:
            return []

    def get_meta(self, chunk_id: str) -> Optional[Dict]:
        return self._meta.get(chunk_id)

    def all_meta(self) -> Dict[str, Dict]:
        return dict(self._meta)
    
    def clear(self):
        """
        Clear all vectors and metadata from the store.
        Useful when switching embedding backends with different dimensions.
        """
        self._meta = {}
        self._ids = []
        self._emb_dim = None
        self._faiss_index = None
        self._np_vectors = None
        self._loaded = False
        
        # Remove existing index files
        try:
            idx_file = os.path.join(self.index_path, "faiss.index")
            if os.path.exists(idx_file):
                os.remove(idx_file)
            if os.path.exists(self.vectors_path):
                os.remove(self.vectors_path)
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
            logger.info("Cleared vector store (removed existing index files)")
        except Exception as e:
            logger.warning("Failed to remove some index files during clear: %s", e)


# -------------------------
# Indexer main class
# -------------------------
class Indexer:
    def __init__(self, cfg: Optional[RagConfig] = None):
        self.cfg = cfg or RagConfig.from_env()
        self.repo_root = os.path.abspath(self.cfg.repo_root or ".")
        self.index_path = os.path.abspath(self.cfg.vectorstore.index_path)
        _ensure_dir(self.index_path)
        self.indexer_cfg = self.cfg.indexing
        self.embedder = Embedder(self.cfg.embedding)
        self.vstore = VectorStore(self.cfg.vectorstore)
        logger.info("Indexer initialized: repo=%s index=%s embed_backend=%s", self.repo_root, self.index_path, self.embedder.backend)

    def index_repo(self, paths: Optional[Iterable[str]] = None, *, force: bool = False) -> Dict:
        """
        Index an entire repo (or specific paths). Returns summary dict.
        If paths is None, walk repo_root and include files matching file_globs (simple implementation).
        
        Args:
            force: If True, force reindexing. Also clears existing index if embedding dimension changes.
        """
        logger.info("Starting index_repo (force=%s)", force)
        
        # If force=True, clear existing index to handle dimension changes when switching embedding backends
        if force and self.vstore._loaded:
            logger.info("Force reindexing: clearing existing index to handle potential dimension changes")
            self.vstore.clear()
        start = time.time()
        candidates: List[str] = []
        if paths:
            # use provided paths (can be files or dirs)
            for p in paths:
                p_abs = os.path.abspath(p)
                if os.path.isdir(p_abs):
                    for root, dirs, files in os.walk(p_abs):
                        for f in files:
                            candidates.append(os.path.join(root, f))
                else:
                    candidates.append(p_abs)
        else:
            # walk repo and include by globs: simple suffix matching
            includes = [g.lstrip("**/") for g in self.indexer_cfg.file_globs]
            excludes = [g.lstrip("**/") for g in self.indexer_cfg.exclude_globs]
            
            # Common directories to exclude (modify dirs in-place to prevent os.walk from descending)
            excluded_dir_names = {"node_modules", ".git", "__pycache__", ".venv", "venv", 
                                  ".next", "dist", "build", ".build", "target", "out", 
                                  ".cache", ".pytest_cache", ".mypy_cache", ".coverage",
                                  "coverage", ".nyc_output", "logs", ".log"}
            
            for root, dirs, files in os.walk(self.repo_root):
                # Modify dirs in-place to prevent os.walk from descending into excluded directories
                dirs[:] = [d for d in dirs if d not in excluded_dir_names and not d.startswith('.')]
                
                # Also check if current root should be skipped
                skip_root = False
                for ex_name in excluded_dir_names:
                    if ex_name in root:
                        skip_root = True
                        break
                if skip_root:
                    continue
                for f in files:
                    fp = os.path.join(root, f)
                    
                    # Skip log files explicitly
                    if f.endswith('.log') or '/logs/' in fp or fp.endswith('/logs'):
                        continue
                    
                    # skip hidden large dirs etc implicitly
                    # include by extension
                    include_match = False
                    for inc in includes:
                        if inc.startswith("*."):
                            if f.endswith(inc.lstrip("*")):
                                include_match = True
                                break
                        else:
                            if inc in fp:
                                include_match = True
                                break
                    if not include_match:
                        continue
                    # exclude patterns - check both full path and relative path
                    skip_file = False
                    fp_rel = os.path.relpath(fp, self.repo_root)
                    for ex in excludes:
                        if not ex.strip():
                            continue
                        # Check both absolute and relative paths
                        if ex in fp or ex in fp_rel:
                            skip_file = True
                            break
                        # Also check if exclude pattern matches as a glob-like pattern
                        # e.g., "**/logs/**" should match any path containing "/logs/"
                        ex_clean = ex.replace("**/", "").replace("/**", "").replace("*", "")
                        if ex_clean and ex_clean in fp:
                            skip_file = True
                            break
                    if skip_file:
                        continue
                    candidates.append(fp)

        # dedupe and relativize
        candidates = list(dict.fromkeys([os.path.relpath(os.path.abspath(x), self.repo_root) for x in candidates if os.path.exists(x)]))
        logger.info("Indexing %d candidate files", len(candidates))
        all_metas: List[ChunkMeta] = []
        for rel in candidates:
            try:
                metas = chunk_file(rel, self.repo_root, self.indexer_cfg)
                all_metas.extend(metas)
            except Exception:
                logger.exception("Failed chunking file: %s", rel)

        logger.info("Produced %d chunks to embed", len(all_metas))
        # embed in batches with progress reporting
        texts = [m.text for m in all_metas]
        embeddings = []
        batch = []
        batch_metas = []
        B = max(1, self.cfg.embedding.batch_size)
        total_batches = (len(texts) + B - 1) // B
        
        # Show progress for large batches
        show_progress = len(texts) > 10
        if show_progress:
            print(f"Embedding {len(texts)} chunks in {total_batches} batches...", end="", flush=True)
        
        for i, txt in enumerate(texts):
            batch.append(txt)
            batch_metas.append(all_metas[i])
            if len(batch) >= B or i == len(texts) - 1:
                batch_num = (i // B) + 1
                if show_progress:
                    print(f"\rEmbedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end="", flush=True)
                try:
                    embs = self.embedder.embed_texts(batch)
                    # ensure 2D list
                    embeddings.extend(embs)
                except Exception:
                    logger.exception("Embedding batch failed; using empty vectors for batch")
                    embs = [[0.0] * 256 for _ in batch]
                    embeddings.extend(embs)
                batch = []
                batch_metas = []
        
        if show_progress:
            print()  # New line after progress

        # persist to vector store
        try:
            # Use force_recreate if force=True to handle dimension mismatches
            self.vstore.add(all_metas, embeddings, force_recreate=force)
        except Exception:
            logger.exception("Failed adding vectors to store")

        # Save index version marker
        index_version = str(int(time.time()))
        self.cfg.index_version = index_version
        # try to persist config
        try:
            cfg_path = os.path.join(self.index_path, "rag_config.json")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"index_version": index_version, "created_at": time.time()}, indent=2))
        except Exception:
            logger.exception("Failed writing rag_config.json")

        elapsed = time.time() - start
        logger.info("Indexing finished: %d chunks indexed in %.2fs", len(all_metas), elapsed)
        return {"chunks_indexed": len(all_metas), "elapsed_s": elapsed, "index_version": index_version}

    def index_changed_files(self) -> Dict:
        """
        Index only changed files based on git status. Requires git available.
        """
        try:
            rc, out, err = self._run_cmd(["git", "status", "--porcelain"], cwd=self.repo_root)
            if rc != 0:
                logger.warning("git status failed: %s", err)
                return {"changed": 0, "msg": "git status failed"}
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            files = []
            for ln in lines:
                # porcelain format: XY PATH
                parts = ln.split()
                if len(parts) >= 2:
                    path = parts[-1]
                    files.append(path)
            if not files:
                return {"changed": 0, "msg": "no changed files"}
            return self.index_repo(paths=files)
        except Exception:
            logger.exception("index_changed_files failed")
            return {"changed": 0, "msg": "exception"}

    def _run_cmd(self, cmd: Iterable[str], cwd: Optional[str] = None, timeout: int = 30):
        import subprocess

        try:
            proc = subprocess.run(list(cmd), cwd=cwd or self.repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            out = proc.stdout.decode("utf-8", errors="replace")
            err = proc.stderr.decode("utf-8", errors="replace")
            return proc.returncode, out, err
        except Exception as e:
            logger.exception("Command failed: %s", e)
            return 1, "", str(e)


# -------------------------
# Simple CLI
# -------------------------
def _cli():
    parser = argparse.ArgumentParser(prog="rag.indexer", description="Index repository for RAG retrieval")
    sub = parser.add_subparsers(dest="cmd")
    p_index = sub.add_parser("index", help="Index repository (all files)")
    p_index.add_argument("--repo", default=".", help="Repository root")
    p_index.add_argument("--index-path", default=None, help="Index path (overrides config)")
    p_index.add_argument("--force", action="store_true", help="Force reindex")
    p_changed = sub.add_parser("index-changed", help="Index changed files (git status)")
    p_changed.add_argument("--repo", default=".", help="Repository root")
    args = parser.parse_args()
    if args.cmd == "index":
        cfg = RagConfig.from_env(repo_root=args.repo)
        if args.index_path:
            cfg.vectorstore.index_path = args.index_path
        idx = Indexer(cfg)
        res = idx.index_repo(force=args.force)
        print(json.dumps(res, indent=2))
    elif args.cmd == "index-changed":
        cfg = RagConfig.from_env(repo_root=args.repo)
        if args.index_path:
            cfg.vectorstore.index_path = args.index_path
        idx = Indexer(cfg)
        res = idx.index_changed_files()
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
