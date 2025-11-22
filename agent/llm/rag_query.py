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
        
        Automatically adjusts for Groq free tier limits (6,000 TPM) by reducing chunks and truncating.
        """
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
            # Check if using Groq and adjust retrieval parameters
            is_groq = False
            max_tokens = None
            if self.llm:
                # Check if LLM is Groq - be precise to avoid false positives with Ollama
                llm_class_name = self.llm.__class__.__name__ if hasattr(self.llm, "__class__") else ""
                provider_attr = getattr(self.llm, "provider", None)
                
                # Check provider attribute first (most reliable)
                if provider_attr:
                    provider_str = str(provider_attr).lower()
                    if "groq" in provider_str and "ollama" not in provider_str:
                        is_groq = True
                
                # Check class name (GroqLLM is specific)
                if "GroqLLM" in llm_class_name:
                    is_groq = True
                
                # Check config - but be careful not to match Ollama models
                if hasattr(self.llm, "config"):
                    config = self.llm.config
                    # Check if provider in config is explicitly "groq"
                    if hasattr(config, "provider") and config.provider and "groq" in str(config.provider).lower():
                        is_groq = True
                    # Don't use model name alone - too many false positives with Ollama
                
                if is_groq:
                    # Groq free tier: 6,000 TPM limit, be conservative
                    max_tokens = 5000  # Leave some headroom
                    # Reduce k for Groq
                    if k is None:
                        k = min(self.cfg.top_k, 3)  # Max 3 chunks for Groq free tier
                    else:
                        k = min(k, 3)
                    logger.info("Detected Groq LLM - reducing chunks to %d and limiting prompt size for free tier compatibility", k)
            
            # 1) retrieve relevant chunks
            chunks = self.retriever.retrieve(question, k=k, use_hybrid=not prefer_exact)
            
            # 2) compose grounded prompt with token limits if using Groq
            max_chunk_chars = self.cfg.max_chunk_chars
            if is_groq:
                # Reduce chunk size for Groq
                max_chunk_chars = min(max_chunk_chars, 1500)  # Smaller chunks for Groq
            
            prompt = grounding_prompt(
                question, 
                chunks, 
                max_chars_per_chunk=max_chunk_chars,
                max_total_tokens=max_tokens
            )
            # quick sources list (filtered and deduplicated)
            sources_list_raw = format_sources_list_for_output(chunks)
            # Convert to string if it's a list, or keep as-is if already a string
            if isinstance(sources_list_raw, list):
                sources_list = sources_list_raw
            else:
                sources_list = sources_list_raw

            # If LLM not requested, return prompt + chunks so caller can inspect
            if not use_llm or self.llm is None:
                elapsed = time.time() - start
                # Ensure sources_list is in the right format
                if isinstance(sources_list, list):
                    sources_output = sources_list
                else:
                    sources_output = sources_list.split("\n") if sources_list else []
                return {
                    "answer": "",
                    "prompt": prompt,
                    "chunks": chunks,
                    "sources": sources_output,
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
                    # If answer_text is still empty or looks like raw JSON, try to parse it
                    if not answer_text or (answer_text.strip().startswith("{") and answer_text.strip().startswith("{")):
                        # Check if it's raw JSON that wasn't parsed
                        try:
                            import json
                            parsed = json.loads(answer_text.strip())
                            if isinstance(parsed, dict):
                                answer_text = parsed.get("response") or parsed.get("text") or parsed.get("content") or ""
                        except (json.JSONDecodeError, ValueError, AttributeError):
                            pass
                else:
                    # could be raw string - check if it's JSON
                    raw_str = str(gen)
                    try:
                        import json
                        parsed = json.loads(raw_str.strip())
                        if isinstance(parsed, dict):
                            answer_text = parsed.get("response") or parsed.get("text") or parsed.get("content") or ""
                            llm_raw = {"text": answer_text, "raw": raw_str}
                        else:
                            llm_raw = {"text": raw_str}
                            answer_text = raw_str
                    except (json.JSONDecodeError, ValueError):
                        llm_raw = {"text": raw_str}
                        answer_text = raw_str

            # Final cleanup: ensure answer_text doesn't contain raw JSON
            # If it looks like JSON, try to extract text from it
            if answer_text and answer_text.strip().startswith("{"):
                # Check if it's a single JSON object or multiple JSON lines
                lines = answer_text.strip().split("\n")
                cleaned_text = ""
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{"):
                        try:
                            import json
                            parsed = json.loads(line)
                            if isinstance(parsed, dict):
                                # Extract text from JSON
                                text = parsed.get("response") or parsed.get("text") or parsed.get("content") or ""
                                if text:
                                    cleaned_text += text
                        except (json.JSONDecodeError, ValueError):
                            # Not valid JSON, keep as-is
                            cleaned_text += line + "\n"
                    else:
                        cleaned_text += line + "\n"
                if cleaned_text.strip():
                    answer_text = cleaned_text.strip()
            
            # Remove SOURCES section from answer if LLM included it (we'll show it separately)
            if answer_text:
                # Look for SOURCES: or === SOURCES === patterns and remove them
                import re
                # Remove SOURCES section that might be in the answer
                answer_text = re.sub(r'\n\s*===?\s*SOURCES?\s*===?\s*\n.*', '', answer_text, flags=re.DOTALL | re.IGNORECASE)
                answer_text = re.sub(r'\n\s*SOURCES?:\s*\n.*', '', answer_text, flags=re.DOTALL | re.IGNORECASE)
                answer_text = answer_text.strip()
            
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
