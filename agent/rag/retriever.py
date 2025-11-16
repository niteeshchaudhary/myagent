# agent/rag/retriever.py
"""
Hybrid retriever for RAG: combines fast exact search (ripgrep / tags) with
semantic retrieval (vector store) and produces grounded context for LLM prompts.

Features:
- Heuristic routing to decide whether a query is an "exact/symbol" query or a
  natural-language conceptual query.
- If exact/symbol-like: run ripgrep (agent.search.rg_search) and tags_client
  to get precise file locations and return them as priority sources.
- Otherwise: embed query and query the vector store (FAISS / numpy fallback).
- Rerank semantic results by a simple lexical overlap score blended with vector score.
- Compose a grounded prompt (with file paths and line ranges) suitable for sending
  to the LLM; includes "EXACT" sources first, then "RETRIEVED" semantic chunks.
- Defensive: if index is missing or empty, falls back to exact search or file
  expansion using @ references.

Public API:
- Retriever(cfg: RagConfig)
    .retrieve(query: str, k: int = None, use_hybrid: bool = True) -> List[dict]
    .compose_grounded_prompt(question: str, chunks: List[dict]) -> str

Returned chunk dict shape:
{
  "source_type": "exact" | "semantic",
  "path": "src/foo.py",
  "start_line": 12,        # optional
  "end_line": 30,          # optional
  "text": "def foo(...): ...",  # excerpt text
  "chunk_id": "abc123" or None,
  "score": 0.92,           # normalized score
  "meta": { ... }          # original metadata from vectorstore if semantic
}
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

# project imports
try:
    from agent.rag.config import RagConfig
    from agent.rag.indexer import Embedder, VectorStore, ChunkMeta
    from agent.search.rg_search import rg_search, find_symbol
    from agent.search.tags_client import TagsClient
    from agent.utils.file_ref import expand_file_refs
except Exception:
    # allow running module directly for debugging if package paths differ
    from rag.config import RagConfig  # type: ignore
    from rag.indexer import Embedder, VectorStore, ChunkMeta  # type: ignore
    from ..search.rg_search import rg_search, find_symbol  # type: ignore
    from ..search.tags_client import TagsClient  # type: ignore
    from ..utils.file_ref import expand_file_refs  # type: ignore

# prefer existing project logger
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


# -------------------------
# Helpers / heuristics
# -------------------------
_SYMBOL_LIKE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.]*$")  # simple identifier or dotted name
_CODE_TOKENS_RE = re.compile(r"[\w_]+\(|def\s+|class\s+|=>|->|::|#\s*type", re.I)


def _is_symbol_query(q: str) -> bool:
    """
    Heuristic: return True if query looks like a request about a symbol/identifier:
    - Short (<= 4 words) and contains identifier-like tokens or explicit phrases
      like "where is X", "find X", "who calls X".
    """
    if not q or not q.strip():
        return False
    q = q.strip()
    ql = q.lower()
    # explicit trigger words
    if any(kw in ql for kw in ("where is ", "find ", "who calls", "definition of", "defined in", "references to")):
        return True
    # if query is a single token that looks like an identifier
    if len(q.split()) == 1 and _SYMBOL_LIKE_RE.match(q):
        return True
    # if query contains obvious code tokens
    if _CODE_TOKENS_RE.search(q):
        return True
    # otherwise treat as conceptual / natural-language
    return False


def _token_overlap_score(a: str, b: str) -> float:
    """
    Simple lexical overlap score: number of common word tokens / sqrt(len(a_tokens)+len(b_tokens)).
    Normalized to [0,1] roughly.
    """
    if not a or not b:
        return 0.0
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    inter = ta.intersection(tb)
    score = len(inter) / math.sqrt(len(ta) + len(tb))
    return float(score)


# -------------------------
# Retriever implementation
# -------------------------
class Retriever:
    def __init__(self, cfg: Optional[RagConfig] = None):
        self.cfg = cfg or RagConfig.from_env()
        # instantiate embedder & vectorstore lazily
        try:
            self.embedder = Embedder(self.cfg.embedding)
            self.vstore = VectorStore(self.cfg.vectorstore)
        except Exception:
            # defensive fallback
            self.embedder = None
            self.vstore = None
        # tags client for definitions
        try:
            self.tags = TagsClient(repo_root=self.cfg.repo_root)
            # attempt to lazy-build tags on demand; do not build in constructor
        except Exception:
            self.tags = None

    # -------------------------
    # High-level retrieve API
    # -------------------------
    def retrieve(self, query: str, k: Optional[int] = None, use_hybrid: Optional[bool] = None) -> List[Dict]:
        """
        Retrieve top-k relevant snippets for query.

        Returns list of dicts with standardized fields:
          - source_type: "exact" or "semantic"
          - path, start_line, end_line, text, chunk_id, score, meta
        """
        k = k or self.cfg.top_k
        use_hybrid = self.cfg.use_hybrid_retrieval if use_hybrid is None else use_hybrid
        q = (query or "").strip()
        if not q:
            return []

        logger.debug("Retriever.retrieve: query=%s k=%s hybrid=%s", q[:160], k, use_hybrid)

        # If query mentions explicit @file references, expand and skip retrieval
        if "@" in q and self.cfg.expand_file_refs:
            logger.debug("Query contains '@' file refs; expanding files and returning expanded content.")
            expanded = expand_file_refs(q, repo_root=self.cfg.repo_root, max_chars=self.cfg.max_chunk_chars)
            # wrap as single pseudo-chunk for LLM
            return [
                {
                    "source_type": "file_expansion",
                    "path": None,
                    "start_line": None,
                    "end_line": None,
                    "text": expanded,
                    "chunk_id": None,
                    "score": 1.0,
                    "meta": {"note": "expanded @file references"},
                }
            ]

        # if symbol-like, run exact searches first
        if _is_symbol_query(q):
            logger.debug("Query classified as symbol/exact type.")
            exact_hits = self._exact_search(q, max_results=k)
            if exact_hits:
                # normalize and return top exact hits
                # exact hits can be enriched by reading file lines around matched line for context
                normalized = [self._normalize_exact_hit(h) for h in exact_hits[:k]]
                # if hybrid is enabled, also get semantic candidates to supplement
                if use_hybrid:
                    sem = self._semantic_search(q, k=k)
                    merged = self._merge_and_rerank(normalized, sem, q, k)
                    return merged
                return normalized

            # if no exact hits, fall back to semantic
            sem = self._semantic_search(q, k=k)
            return sem

        # otherwise conceptual query — primarily semantic retrieval
        sem = self._semantic_search(q, k=k)
        # Optionally enrich with exact symbol hits if hybrid
        if use_hybrid:
            # run a light exact search for keywords from top semantic results (helpful for citations)
            try:
                keywords = self._extract_keywords(q, limit=3)
                extra_exact = []
                for kw in keywords:
                    extra_exact.extend(self._exact_search(kw, max_results=2))
                exact_norm = [self._normalize_exact_hit(h) for h in extra_exact]
                merged = self._merge_and_rerank(exact_norm, sem, q, k)
                return merged
            except Exception:
                logger.exception("Hybrid enrichment failed; returning semantic results")
                return sem
        return sem

    # -------------------------
    # Exact search helpers
    # -------------------------
    def _exact_search(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        Use ripgrep and tags to find exact symbol definitions/usages.
        Returns list of rg_hit dicts (raw output from rg_search) and tag entries.
        """
        hits: List[Dict] = []
        # First try symbol definitions via tags client
        try:
            if self.tags:
                # ensure tags are built if empty
                if not self.tags._all_entries:
                    try:
                        self.tags.build_tags()
                    except Exception:
                        pass
                defs = self.tags.find_definitions(query, max_results=max_results)
                if defs:
                    # convert tags entries to rg-like dicts
                    for d in defs:
                        # tag entry fields: name, path, line, pattern, kind
                        hits.append({"path": os.path.relpath(d.get("path") or "", self.cfg.repo_root), "line": d.get("line"), "text": d.get("pattern") or "", "match_text": d.get("name"), "type": "tag"})
            # If tags did not find anything or to supplement, use ripgrep
            rg_hits = rg_search(query, repo_root=self.cfg.repo_root, max_results=max_results)
            if rg_hits:
                hits.extend(rg_hits)
        except Exception:
            logger.exception("Exact search failed; falling back to rg_search only.")
            try:
                hits = rg_search(query, repo_root=self.cfg.repo_root, max_results=max_results)
            except Exception:
                logger.exception("rg_search fallback also failed.")
                hits = []
        # dedupe by path+line
        seen = set()
        out = []
        for h in hits:
            key = (h.get("path"), h.get("line"))
            if key in seen:
                continue
            seen.add(key)
            out.append(h)
            if len(out) >= max_results:
                break
        return out

    def _normalize_exact_hit(self, hit: Dict) -> Dict:
        """
        Convert rg/tags hit into normalized chunk dict expected by downstream code.
        Attempt to read a snippet of file around the matched line for context.
        """
        path = hit.get("path") or hit.get("file") or ""
        # prefer relative path
        try:
            path_rel = os.path.relpath(path, self.cfg.repo_root) if os.path.isabs(path) else path
        except Exception:
            path_rel = path
        line = hit.get("line") or hit.get("line_number") or hit.get("ln") or None
        snippet = ""
        start_line = None
        end_line = None
        try:
            abs_path = os.path.join(self.cfg.repo_root, path_rel)
            if os.path.exists(abs_path) and line:
                ln = int(line)
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    all_lines = fh.read().splitlines()
                # choose window +-8 lines
                s = max(1, ln - 8)
                e = min(len(all_lines), ln + 8)
                snippet = "\n".join(all_lines[s - 1 : e])
                start_line = s
                end_line = e
            else:
                # fallback: try to include line text if provided in hit
                snippet = hit.get("text") or ""
        except Exception:
            logger.debug("Failed to read snippet for exact hit %s:%s", path_rel, line)
            snippet = hit.get("text") or ""
        return {
            "source_type": "exact",
            "path": path_rel,
            "start_line": start_line,
            "end_line": end_line,
            "text": snippet,
            "chunk_id": None,
            "score": 1.0,
            "meta": {"raw_hit": hit},
        }

    # -------------------------
    # Semantic retrieval helpers
    # -------------------------
    def _semantic_search(self, query: str, k: int = 8) -> List[Dict]:
        """
        Embed the query and query the vector store.
        Returns list of normalized chunk dicts with meta from vectorstore.
        """
        if not self.vstore or not self.embedder:
            logger.warning("Vector store or embedder not initialized; returning empty semantic results.")
            return []

        # embed the query
        try:
            vecs = self.embedder.embed_texts([query])
            if not vecs:
                return []
            vec = vecs[0]
        except Exception:
            logger.exception("Embedding query failed; falling back to simple empty list.")
            return []

        # query vector store
        try:
            hits = self.vstore.query(vec, top_k=k * 2)  # request a margin for reranking
        except Exception:
            logger.exception("Vector store query failed.")
            hits = []

        results: List[Dict] = []
        # convert chunk ids to metas
        for cid, score in hits:
            meta = self.vstore.get_meta(cid) or {}
            text = meta.get("text") or ""
            path = meta.get("path") or meta.get("file") or ""
            start_line = meta.get("start_line")
            end_line = meta.get("end_line")
            results.append(
                {
                    "source_type": "semantic",
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "text": text,
                    "chunk_id": cid,
                    "score": float(score),
                    "meta": meta,
                }
            )
        # rerank by blending vector score and lexical overlap
        reranked = self._rerank_semantic_results(results, query, top_k=k)
        return reranked

    def _rerank_semantic_results(self, results: List[Dict], query: str, top_k: int = 8) -> List[Dict]:
        """
        Blend vector score and lexical overlap. Vector scores may be cosine similarity in [0,1]
        (FAISS normalized) or dot-product-like values.
        We compute a blended score = alpha * normalized_vector + (1-alpha) * overlap_score.
        Normalize vector scores to [0,1] by min/max in results.
        """
        if not results:
            return []
        vec_scores = [r.get("score", 0.0) for r in results]
        min_s = min(vec_scores)
        max_s = max(vec_scores)
        range_s = max_s - min_s if max_s - min_s > 1e-12 else 1.0
        normalized = [(s - min_s) / range_s for s in vec_scores]
        alpha = 0.75  # weight to give vector similarity
        scored: List[Tuple[float, Dict]] = []
        for r, ns in zip(results, normalized):
            overlap = _token_overlap_score(r.get("text", ""), query)
            blended = alpha * ns + (1.0 - alpha) * overlap
            r["score"] = float(blended)
            scored.append((blended, r))
        # sort desc by blended score
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for _, r in scored[:top_k]]
        return top

    # -------------------------
    # Merge exact + semantic results
    # -------------------------
    def _merge_and_rerank(self, exact: List[Dict], semantic: List[Dict], query: str, top_k: int) -> List[Dict]:
        """
        Merge two lists (exact first) and rerank with a bias toward exact hits.
        Exact hits get a small boost to ensure they're prioritized when relevant.
        """
        merged = []
        seen = set()
        # boost exact hits
        for e in exact:
            key = (e.get("path"), e.get("start_line"), e.get("end_line"))
            if key in seen:
                continue
            seen.add(key)
            e_copy = dict(e)
            e_copy["score"] = float(e_copy.get("score", 1.0) + 0.2)  # small boost
            merged.append(e_copy)
        # add semantic hits
        for s in semantic:
            key = (s.get("path"), s.get("start_line"), s.get("end_line"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(s)
        # final rerank using token overlap + existing score
        for m in merged:
            overlap = _token_overlap_score(m.get("text", ""), query)
            m["score"] = float(0.7 * m.get("score", 0.0) + 0.3 * overlap)
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged[:top_k]

    # -------------------------
    # Utilities
    # -------------------------
    def _extract_keywords(self, text: str, limit: int = 3) -> List[str]:
        """
        Very simple keyword extractor: return top N longest alphanumeric tokens excluding stopwords.
        """
        tokens = re.findall(r"\w+", text)
        tokens = [t for t in tokens if len(t) > 2]
        # remove common stopwords (tiny set)
        stop = {"the", "and", "for", "with", "that", "this", "from", "into", "to", "a", "an", "in", "is"}
        tokens = [t for t in tokens if t.lower() not in stop]
        tokens = sorted(set(tokens), key=lambda x: (-len(x), x))[:limit]
        return tokens

    def compose_grounded_prompt(self, question: str, chunks: List[Dict], max_chars_per_chunk: Optional[int] = None) -> str:
        """
        Compose a grounding prompt for the LLM. Format includes:
         - short instruction
         - EXACT sources (if any)
         - RETRIEVED semantic sources
         - user question
         - instruction to answer only using sources and list sources at end

        max_chars_per_chunk: truncate each chunk's text to this many chars if set.
        """
        max_chars_per_chunk = max_chars_per_chunk or self.cfg.max_chunk_chars
        lines = []
        lines.append("You are an assistant that answers questions about the repository.")
        lines.append("Use ONLY the provided sources to answer. If the answer is not contained in the sources, say you don't know and suggest which files to inspect.")
        lines.append("")
        exacts = [c for c in chunks if c.get("source_type") == "exact"]
        sems = [c for c in chunks if c.get("source_type") == "semantic"]
        # add exact sources first
        if exacts:
            lines.append("EXACT SOURCES:")
            for i, c in enumerate(exacts, 1):
                header = f"{i}) <FILE: {c.get('path')}"
                if c.get("start_line") or c.get("end_line"):
                    header += f" lines {c.get('start_line') or '?'}-{c.get('end_line') or '?'}"
                header += ">"
                lines.append(header)
                text = c.get("text") or ""
                if max_chars_per_chunk and len(text) > max_chars_per_chunk:
                    text = text[: max_chars_per_chunk - 100] + "\n\n...[TRUNCATED]"
                lines.append("```file")
                lines.append(text)
                lines.append("```")
                lines.append("")
        # semantic sources
        if sems:
            lines.append("RETRIEVED SOURCES:")
            for i, c in enumerate(sems, 1):
                header = f"{i}) <FILE: {c.get('path')}"
                if c.get("start_line") or c.get("end_line"):
                    header += f" lines {c.get('start_line') or '?'}-{c.get('end_line') or '?'}"
                header += f"> (score={c.get('score', 0):.3f})"
                lines.append(header)
                text = c.get("text") or ""
                if max_chars_per_chunk and len(text) > max_chars_per_chunk:
                    text = text[: max_chars_per_chunk - 100] + "\n\n...[TRUNCATED]"
                lines.append("```file")
                lines.append(text)
                lines.append("```")
                lines.append("")

        # final question and constraints
        lines.append("QUESTION:")
        lines.append(question)
        lines.append("")
        lines.append("INSTRUCTIONS:")
        lines.append("- Answer concisely and correctly using only the sources above.")
        lines.append("- At the end, include a SOURCES: list with file paths and line ranges you used.")
        prompt = "\n".join(lines)
        return prompt

    def get_chunks_by_ids(self, ids: Iterable[str]) -> List[Dict]:
        """
        Retrieve chunk metas from vectorstore given chunk ids.
        """
        out = []
        if not self.vstore:
            return out
        for cid in ids:
            m = self.vstore.get_meta(cid)
            if not m:
                continue
            out.append(
                {
                    "source_type": "semantic",
                    "path": m.get("path"),
                    "start_line": m.get("start_line"),
                    "end_line": m.get("end_line"),
                    "text": m.get("text"),
                    "chunk_id": cid,
                    "score": 1.0,
                    "meta": m,
                }
            )
        return out


# -------------------------
# Convenience CLI
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="rag.retriever", description="Hybrid retriever for codebase RAG")
    parser.add_argument("--repo", default=".", help="repo root")
    parser.add_argument("--query", required=True, help="query text")
    parser.add_argument("--k", type=int, default=6, help="top k results")
    args = parser.parse_args()
    cfg = RagConfig.from_env(repo_root=args.repo)
    r = Retriever(cfg)
    res = r.retrieve(args.query, k=args.k)
    print(json.dumps(res, indent=2))
