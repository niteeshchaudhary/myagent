# agent/llm/rag_query.py
"""
High-level RAG query wrapper that glues together:
- Retriever (hybrid exact + semantic)
- Prompt templates (grounding)
- LLM adapter calls (generate / stream)

Public API:
- RAGQuery(cfg=None, retriever=None, llm=None)
    .answer(question, k=None, prefer_exact=False, use_llm=True, timeout=None) -> dict

Returned dict:
{
  "answer": str,               # final LLM answer (empty if use_llm=False)
  "prompt": str,               # the grounding prompt sent to the LLM (or composed prompt)
  "chunks": [...],             # list of chunks used (normalized)
  "sources": [...],            # short list of sources (path + lines)
  "llm_raw": {...} or None,    # raw LLM adapter return when available
  "used_llm": bool,
  "elapsed_s": float
}
"""

from __future__ import annotations

import time
import traceback
from typing import Any, Dict, Iterable, List, Optional

# project imports
try:
    from agent.rag.retriever import Retriever
    from agent.rag.templates import grounding_prompt, format_sources_list_for_output
    from agent.rag.config import RagConfig
except Exception:
    # fallback if module path resolution differs when running standalone
    from rag.retriever import Retriever  # type: ignore
    from rag.templates import grounding_prompt, format_sources_list_for_output  # type: ignore
    from rag.config import RagConfig  # type: ignore

# logger
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


class RAGQuery:
    """
    Orchestrates retrieval + grounding prompt composition + LLM invocation.

    Parameters:
        cfg: RagConfig (optional). If not provided, a default from env will be used.
        retriever: optional Retriever instance (will be created if omitted).
        llm: optional LLM adapter. Expected to expose `.generate(prompt, model=None, timeout=None)` and/or `.stream(prompt, model=None)`.
             If llm is None and use_llm=True, the method will return the prompt and retrieved chunks for offline use.
    """

    def __init__(self, cfg: Optional[RagConfig] = None, retriever: Optional[Retriever] = None, llm: Optional[Any] = None):
        self.cfg = cfg or RagConfig.from_env()
        self.retriever = retriever or Retriever(self.cfg)
        self.llm = llm

    def answer(
        self,
        question: str,
        *,
        k: Optional[int] = None,
        prefer_exact: bool = False,
        use_llm: bool = True,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Run retrieval + (optionally) ask LLM to answer using the grounded prompt.

        Args:
          question: user question
          k: number of retrieval results to use (defaults to cfg.top_k)
          prefer_exact: if True, bias to exact search first (useful for symbol queries)
          use_llm: if False, do not call LLM; return the composed prompt and chunks only
          model: optional model name/identifier (passed to llm.generate)
          timeout: optional timeout for LLM calls
          stream: if True and llm has .stream, stream chunks (not implemented for CLI here; returns concatenated text)

        Returns:
          dict with keys: answer, prompt, chunks, sources, llm_raw, used_llm, elapsed_s
        """
        start = time.time()
        try:
            # 1) retrieve relevant chunks
            chunks = self.retriever.retrieve(question, k=k, use_hybrid=not prefer_exact)
            # 2) compose grounded prompt
            prompt = grounding_prompt(question, chunks, max_chars_per_chunk=self.cfg.max_chunk_chars)
            # quick sources list
            sources_list = format_sources_list_for_output(chunks)

            # If LLM not requested, return prompt + chunks so caller can inspect
            if not use_llm or self.llm is None:
                elapsed = time.time() - start
                return {
                    "answer": "",
                    "prompt": prompt,
                    "chunks": chunks,
                    "sources": sources_list,
                    "llm_raw": None,
                    "used_llm": False,
                    "elapsed_s": elapsed,
                }

            # 3) call LLM
            llm_raw = None
            answer_text = ""
            # prefer streaming if requested and available
            if stream and hasattr(self.llm, "stream"):
                try:
                    # collect streamed chunks into a final string
                    for chunk in self.llm.stream(prompt, model=model):
                        # chunk may be str
                        if isinstance(chunk, (str, bytes)):
                            piece = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                            answer_text += piece
                        else:
                            # unexpected chunk type; stringify
                            answer_text += str(chunk)
                    llm_raw = {"streamed": True, "text": answer_text}
                except Exception:
                    logger.exception("LLM streaming failed; falling back to non-streamed generate.")
                    # fallback to generate
                    try:
                        gen = self._call_llm_generate(prompt, model=model, timeout=timeout)
                        answer_text = gen.get("text") or str(gen)
                        llm_raw = gen
                    except Exception:
                        logger.exception("LLM generate also failed after streaming failed.")
                        answer_text = ""
                        llm_raw = None
            else:
                # non-stream path
                gen = self._call_llm_generate(prompt, model=model, timeout=timeout)
                # normalize gen to dict with 'text'
                if isinstance(gen, dict):
                    llm_raw = gen
                    answer_text = gen.get("text") or gen.get("output") or ""
                else:
                    # could be raw string
                    llm_raw = {"text": str(gen)}
                    answer_text = str(gen)

            elapsed = time.time() - start
            return {
                "answer": answer_text,
                "prompt": prompt,
                "chunks": chunks,
                "sources": sources_list,
                "llm_raw": llm_raw,
                "used_llm": True,
                "elapsed_s": elapsed,
            }
        except Exception as e:
            logger.exception("RAGQuery.answer failed: %s", e)
            elapsed = time.time() - start
            return {
                "answer": "",
                "prompt": "",
                "chunks": [],
                "sources": "",
                "llm_raw": {"error": str(e), "trace": traceback.format_exc()},
                "used_llm": False,
                "elapsed_s": elapsed,
            }

    def _call_llm_generate(self, prompt: str, model: Optional[str] = None, timeout: Optional[int] = None) -> Any:
        """
        Call llm.generate(prompt, model=..., timeout=...). Normalize a few adapter shapes.
        """
        if not self.llm:
            raise RuntimeError("No LLM adapter configured for RAGQuery.")
        # If adapter exposes generate(), call it
        if hasattr(self.llm, "generate"):
            try:
                return self.llm.generate(prompt, model=model, timeout=timeout)
            except TypeError:
                # some adapters may have different signature
                return self.llm.generate(prompt)
        # if adapter is callable, call directly
        if callable(self.llm):
            out = self.llm(prompt)
            if isinstance(out, dict):
                return out
            return {"text": str(out)}
        raise RuntimeError("LLM adapter does not support generate() and is not callable.")


# -------------------------
# Small convenience function
# -------------------------
def ask_rag_once(question: str, cfg: Optional[RagConfig] = None, llm: Optional[Any] = None, k: Optional[int] = None, use_llm: bool = True) -> Dict[str, Any]:
    """
    Quick helper: create a RAGQuery from cfg and llm and run answer().
    """
    cfg = cfg or RagConfig.from_env()
    rq = RAGQuery(cfg=cfg, retriever=Retriever(cfg), llm=llm)
    return rq.answer(question, k=k, use_llm=use_llm)
