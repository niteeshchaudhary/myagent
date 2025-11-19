# agent/llm/openai_llm.py
"""
OpenAI adapter for your agent.

Provides:
- OpenAI_LLM: simple adapter with generate() and stream() APIs.
- Uses the `openai` python package when available. Falls back to HTTP via requests if not.
- Streaming uses the openai library streaming interface when present; otherwise falls back to chunked HTTP (best-effort).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, Optional, Union

# Try to use project's logger
try:
    from agent.utils.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Try to import openai package
try:
    import openai  # type: ignore

    _HAS_OPENAI = True
except Exception:
    openai = None  # type: ignore
    _HAS_OPENAI = False

# Fallback to requests if openai package not available
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


@dataclass
class OpenAIConfig:
    """
    Configuration for OpenAI adapter.
    - api_key: override OPENAI_API_KEY env var if desired.
    - model: default model to use (e.g., "gpt-4o" or "gpt-4").
    - timeout: network timeout in seconds.
    - max_tokens: optional tokens limit.
    - temperature: optional sampling temperature.
    - retry_attempts / retry_backoff: basic retry strategy for network calls.
    """
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    retry_attempts: int = 2
    retry_backoff: float = 1.0


class OpenAI_LLM:
    """
    Adapter class for OpenAI.

    Example:
        cfg = OpenAIConfig(model="gpt-4o")
        llm = OpenAI_LLM(cfg)
        res = llm.generate("Write a haiku about code.")
        print(res["text"])
    """

    def __init__(self, config: Optional[OpenAIConfig] = None, **kwargs):
        # allow passing model, api_key, timeout directly via kwargs for convenience
        self.config = config or OpenAIConfig()
        # override config fields with kwargs if provided
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        # read api key from config or env
        self.api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. OpenAI requests will likely fail until it's provided.")

        # configure openai package if present
        self.client = None
        if _HAS_OPENAI and openai is not None:
            try:
                # Use new OpenAI v1.0.0+ client API
                api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
                base_url = os.environ.get("OPENAI_API_BASE", None)
                client_kwargs = {"api_key": api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url
                self.client = openai.OpenAI(**client_kwargs)
            except Exception as e:
                logger.debug("Failed to create OpenAI client: %s", e)

    # Public API
    def generate(self, prompt: str, *, model: Optional[str] = None, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, timeout: Optional[int] = None) -> Dict:
        """
        Synchronous generation returning full text result.
        Returns a dict: {"text": <str>, "provider": "openai", "raw": <provider response>}
        """
        model = model or self.config.model
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        timeout = timeout or self.config.timeout

        if _HAS_OPENAI and openai is not None and self.client is not None:
            return self._generate_via_openai_pkg(prompt, model=model, max_tokens=max_tokens,
                                                 temperature=temperature, timeout=timeout)
        else:
            # fallback to HTTP via requests
            return self._generate_via_http(prompt, model=model, max_tokens=max_tokens,
                                           temperature=temperature, timeout=timeout)

    def stream(self, prompt: str, *, model: Optional[str] = None, max_tokens: Optional[int] = None,
               temperature: Optional[float] = None, timeout: Optional[int] = None) -> Iterable[str]:
        """
        Generator that yields chunks of text as they arrive.
        If openai package is available, uses streaming interface; else falls back to single-shot generate.
        """
        model = model or self.config.model
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        timeout = timeout or self.config.timeout

        if _HAS_OPENAI and openai is not None and self.client is not None:
            yield from self._stream_via_openai_pkg(prompt, model=model, max_tokens=max_tokens,
                                                   temperature=temperature, timeout=timeout)
        else:
            # fallback to single-shot
            res = self.generate(prompt, model=model, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
            yield res.get("text", "")

    # ---- Implementations using openai package ----
    def _generate_via_openai_pkg(self, prompt: str, *, model: Optional[str], max_tokens: Optional[int],
                                 temperature: Optional[float], timeout: int) -> Dict:
        """
        Use OpenAI v1.0.0+ client API (client.chat.completions.create or client.completions.create).
        """
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Check API key and package installation.")

        # Build messages for chat-based models
        is_chat_model = True  # assume chat style; user may set model accordingly
        if model and model.lower().startswith("text-"):
            # legacy completion model
            is_chat_model = False

        # Build call kwargs
        call_kwargs = {
            "model": model,
            "timeout": timeout,
        }
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        attempt = 0
        while True:
            attempt += 1
            try:
                if is_chat_model:
                    # Use new chat completions API
                    messages = [{"role": "user", "content": prompt}]
                    resp = self.client.chat.completions.create(messages=messages, **call_kwargs)
                    # Parse response: join all choices
                    text = ""
                    if resp.choices:
                        for choice in resp.choices:
                            if choice.message and choice.message.content:
                                text += choice.message.content
                    return {"text": text, "provider": "openai", "raw": resp}
                else:
                    # Use legacy completions API
                    resp = self.client.completions.create(prompt=prompt, **call_kwargs)
                    text = ""
                    if resp.choices:
                        for choice in resp.choices:
                            if choice.text:
                                text += choice.text
                    return {"text": text, "provider": "openai", "raw": resp}
            except Exception as exc:
                logger.warning("OpenAI request attempt %s failed: %s", attempt, exc)
                if attempt > self.config.retry_attempts:
                    logger.exception("OpenAI request failed after %s attempts", attempt)
                    raise
                time.sleep(self.config.retry_backoff * attempt)

    def _stream_via_openai_pkg(self, prompt: str, *, model: Optional[str], max_tokens: Optional[int],
                               temperature: Optional[float], timeout: int) -> Generator[str, None, None]:
        """
        Use the OpenAI v1.0.0+ client streaming interface.
        Yields incrementally as 'delta' chunks arrive.
        """
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Check API key and package installation.")

        is_chat_model = True
        if model and model.lower().startswith("text-"):
            is_chat_model = False

        call_kwargs = {
            "model": model,
            "stream": True,
        }
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        try:
            if is_chat_model:
                # streaming chat using new API
                messages = [{"role": "user", "content": prompt}]
                stream = self.client.chat.completions.create(messages=messages, **call_kwargs)
                for chunk in stream:
                    if chunk.choices:
                        for choice in chunk.choices:
                            if choice.delta and choice.delta.content:
                                yield choice.delta.content
            else:
                # legacy streaming completions
                stream = self.client.completions.create(prompt=prompt, **call_kwargs)
                for chunk in stream:
                    if chunk.choices:
                        for choice in chunk.choices:
                            if choice.text:
                                yield choice.text
        except Exception as exc:
            logger.exception("OpenAI streaming failed: %s", exc)
            # on streaming errors, attempt to yield nothing further and re-raise
            raise

    # ---- HTTP fallback ----
    def _generate_via_http(self, prompt: str, *, model: Optional[str], max_tokens: Optional[int],
                           temperature: Optional[float], timeout: int) -> Dict:
        """
        Very small HTTP fallback using requests to call OpenAI-like endpoints.
        Best-effort; user should install 'openai' package for production use.
        """
        if requests is None:
            raise RuntimeError("Neither 'openai' package nor 'requests' is installed; cannot call OpenAI API.")

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (and not provided in config)")

        # Choose endpoint depending on whether it's a chat model name or text
        is_chat_model = True
        if model and model.lower().startswith("text-"):
            is_chat_model = False

        base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {}
        if is_chat_model:
            endpoint = f"{base.rstrip('/')}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            endpoint = f"{base.rstrip('/')}/v1/completions"
            payload = {"model": model, "prompt": prompt}

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        attempt = 0
        last_exc = None
        while attempt <= self.config.retry_attempts:
            attempt += 1
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
                if not resp.ok:
                    last_exc = RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
                    logger.warning("OpenAI HTTP returned %s: %s", resp.status_code, resp.text[:200])
                    time.sleep(self.config.retry_backoff * attempt)
                    continue
                j = resp.json()
                # parse a few shapes
                text = ""
                if is_chat_model:
                    if "choices" in j:
                        for c in j["choices"]:
                            # choice may have message.content
                            if isinstance(c, dict):
                                msg = c.get("message") or {}
                                text += msg.get("content", "") or c.get("text", "")
                            else:
                                text += str(c)
                    elif "message" in j:
                        text = j["message"].get("content", "")
                    else:
                        text = str(j)
                else:
                    if "choices" in j:
                        for c in j["choices"]:
                            text += c.get("text", "")
                    else:
                        text = str(j)
                return {"text": text, "provider": "openai_http", "raw": j}
            except Exception as exc:
                logger.warning("OpenAI HTTP attempt %s failed: %s", attempt, exc)
                last_exc = exc
                time.sleep(self.config.retry_backoff * attempt)
        raise RuntimeError(f"OpenAI HTTP requests failed after {attempt} attempts. Last error: {last_exc}")

    # ---- Probe ----
    def available_providers(self) -> Dict[str, bool]:
        """
        Quick probe returning whether OpenAI API key is present and whether the openai package is installed.
        """
        return {
            "openai_pkg_installed": _HAS_OPENAI,
            "openai_api_key_set": bool(self.api_key),
        }


# Quick manual test when executed directly (not when imported)
if __name__ == "__main__":
    cfg = OpenAIConfig(model=os.environ.get("DEFAULT_MODEL", "gpt-4o"))
    llm = OpenAI_LLM(cfg)
    print("Probe:", llm.available_providers())
    try:
        print("Generating (single-shot)...")
        out = llm.generate("Say hello in one friendly sentence.")
        print("->", out.get("text"))
        print("\nStreaming demo:")
        for chunk in llm.stream("Write a short friendly greeting in multiple parts."):
            print(chunk, end="", flush=True)
        print("\nDone.")
    except Exception as e:
        logger.exception("Demo failed: %s", e)
