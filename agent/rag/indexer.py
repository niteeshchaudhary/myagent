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
        if cfg.prefer_openai and _HAS_OPENAI and (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")):
            self.backend = "openai"
            # set openai api key if present in env (openai lib will auto pick)
        else:
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
        if self.backend == "openai" and _HAS_OPENAI:
            try:
                model = self.cfg.openai_model
                # batch in chunks of cfg.batch_size
                embeddings: List[List[float]] = []
                batch = []
                for t in texts:
                    batch.append(t)
                    if len(batch) >= self.cfg.batch_size:
                        resp = openai.Embedding.create(model=model, input=batch)
                        embeddings.extend([e["embedding"] for e in resp["data"]])
                        batch = []
                if batch:
                    resp = openai.Embedding.create(model=model, input=batch)
                    embeddings.extend([e["embedding"] for e in resp["data"]])
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

    def add(self, chunk_metas: List[ChunkMeta], embeddings: List[List[float]]):
        """
        Add chunks with embeddings to the store.
        embeddings length must match chunk_metas length.
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
            raise ValueError(f"Embedding dim mismatch: expected {self._emb_dim}, got {dim}")

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
        """
        logger.info("Starting index_repo (force=%s)", force)
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
            for root, dirs, files in os.walk(self.repo_root):
                # skip excluded directories
                skip = False
                for ex in ("node_modules", ".git", "__pycache__", ".venv", "venv"):
                    if ex in root:
                        skip = True
                        break
                if skip:
                    continue
                for f in files:
                    fp = os.path.join(root, f)
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
                    # exclude patterns
                    skip_file = False
                    for ex in excludes:
                        if ex.strip() and ex in fp:
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
        # embed in batches
        texts = [m.text for m in all_metas]
        embeddings = []
        batch = []
        batch_metas = []
        B = max(1, self.cfg.embedding.batch_size)
        for i, txt in enumerate(texts):
            batch.append(txt)
            batch_metas.append(all_metas[i])
            if len(batch) >= B or i == len(texts) - 1:
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

        # persist to vector store
        try:
            self.vstore.add(all_metas, embeddings)
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
