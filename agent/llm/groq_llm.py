# agent/llm/groq_llm.py
"""
Groq adapter for your agent using the official Groq Python SDK.

Provides:
- GroqLLM: adapter with generate() and stream() APIs using the Groq SDK.
- Uses GROQ_API_KEY by default, or accepts it via GroqConfig.
- Supports streaming via the SDK's streaming interface.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, Optional

# Use your project's logger if available
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

# Try to import Groq SDK
try:
    from groq import Groq  # type: ignore

    _HAS_GROQ = True
except Exception:
    Groq = None  # type: ignore
    _HAS_GROQ = False


@dataclass
class GroqConfig:
    """
    Config for Groq adapter.
    - api_key: API key for Groq (defaults to GROQ_API_KEY env var)
    - model: optional model name
    - timeout: request timeout seconds
    - max_tokens, temperature: forwarded to the API if supported
    - retry_attempts/backoff: retry strategy for transient errors
    """
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    retry_attempts: int = 2
    retry_backoff: float = 1.0


class GroqLLM:
    """
    Groq LLM adapter using the official Groq Python SDK.

    Example:
        cfg = GroqConfig(api_key=os.environ.get("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
        llm = GroqLLM(cfg)
        out = llm.generate("Hello")
        for chunk in llm.stream("Hello"): print(chunk)
    """

    def __init__(self, config: Optional[GroqConfig] = None, **kwargs):
        self.config = config or GroqConfig()
        # override config fields with kwargs if provided
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        # env fallback
        if not self.config.api_key:
            self.config.api_key = os.environ.get("GROQ_API_KEY")

        # Create Groq client
        self.client = None
        if _HAS_GROQ and Groq is not None:
            try:
                self.client = Groq(api_key=self.config.api_key)
            except Exception as e:
                logger.debug("Failed to create Groq client: %s", e)
        else:
            logger.warning("Groq SDK not installed. Install with: pip install groq")

    # Public API
    def generate(self, prompt: str, *, model: Optional[str] = None, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, timeout: Optional[int] = None) -> Dict:
        """
        Synchronous single-shot generation. Returns dict: {"text", "provider", "raw", ...}
        """
        if self.client is None:
            raise RuntimeError("Groq client not initialized. Check API key and install groq package: pip install groq")

        model = model or self.config.model
        if not model:
            raise RuntimeError("Model not specified. Set GroqConfig.model or pass model parameter.")
        
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        timeout = timeout or self.config.timeout

        messages = [{"role": "user", "content": prompt}]
        
        call_kwargs = {
            "model": model,
            "messages": messages,
        }
        if max_tokens is not None:
            call_kwargs["max_completion_tokens"] = max_tokens
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self.client.chat.completions.create(**call_kwargs)
                # Parse response
                text = ""
                if resp.choices:
                    for choice in resp.choices:
                        if choice.message and choice.message.content:
                            text += choice.message.content
                return {"text": text, "provider": "groq", "raw": resp}
            except Exception as exc:
                # Check for 413 (Request too large) or rate limit errors
                error_msg = str(exc)
                error_dict = {}
                try:
                    # Try to extract error details from Groq exception
                    if hasattr(exc, 'response') and hasattr(exc.response, 'json'):
                        error_dict = exc.response.json()
                    elif hasattr(exc, 'body'):
                        import json
                        error_dict = json.loads(exc.body) if isinstance(exc.body, str) else exc.body
                except Exception:
                    pass
                
                is_413_error = "413" in error_msg or "Request too large" in error_msg or "tokens per minute" in error_msg
                is_429_error = "429" in error_msg or "Rate limit" in error_msg or "rate_limit" in error_msg
                
                if is_413_error:
                    # Estimate prompt tokens
                    prompt_tokens = len(messages[0].get("content", "")) // 4 if messages else 0
                    logger.error(
                        "Groq API error 413: Request too large. "
                        "Estimated prompt tokens: ~%d. "
                        "Groq on_demand tier has a 6,000 TPM limit. "
                        "The prompt has been truncated, but may still be too large. "
                        "Consider: 1) Reducing memory/context size, 2) Using a smaller model, "
                        "3) Upgrading to Groq Dev Tier, or 4) Using a different LLM provider.",
                        prompt_tokens
                    )
                    # Don't retry 413 errors - they won't succeed
                    raise RuntimeError(
                        f"Groq API error: Request too large (413). "
                        f"Estimated prompt tokens: ~{prompt_tokens}. "
                        f"Groq on_demand tier limit: 6,000 TPM. "
                        f"Please reduce prompt size or upgrade your Groq tier."
                    ) from exc
                
                if is_429_error:
                    # Extract error details
                    error_info = error_dict.get("error", {}) if error_dict else {}
                    error_message = error_info.get("message", error_msg)
                    error_type = error_info.get("type", "")
                    error_code = error_info.get("code", "")
                    
                    # Check if it's a daily limit (TPD) vs per-minute limit (TPM)
                    is_daily_limit = "tokens per day" in error_message.lower() or "TPD" in error_message
                    is_minute_limit = "tokens per minute" in error_message.lower() or "TPM" in error_message
                    
                    # Extract wait time if available
                    wait_seconds = None
                    import re
                    wait_match = re.search(r'Please try again in ([\d.]+)s', error_message)
                    if wait_match:
                        try:
                            wait_seconds = float(wait_match.group(1))
                        except Exception:
                            pass
                    
                    if is_daily_limit:
                        # Daily limit hit - don't retry, provide clear message
                        logger.error(
                            "Groq API error 429: Daily token limit reached. %s. "
                            "Groq free tier has a 500,000 tokens per day limit. "
                            "You've exhausted your daily quota. Please: "
                            "1) Wait until the limit resets (usually at midnight UTC), "
                            "2) Upgrade to Groq Dev Tier for higher limits, or "
                            "3) Use a different LLM provider.",
                            error_message
                        )
                        raise RuntimeError(
                            f"Groq API error: Daily token limit reached (429). "
                            f"{error_message}. "
                            f"Groq free tier limit: 500,000 tokens per day. "
                            f"Please wait for the limit to reset or upgrade your tier."
                        ) from exc
                    elif is_minute_limit and wait_seconds:
                        # Per-minute limit - wait and retry
                        logger.warning(
                            "Groq API error 429: Rate limit (per-minute). Waiting %d seconds before retry...",
                            int(wait_seconds)
                        )
                        if attempt <= self.config.retry_attempts:
                            time.sleep(wait_seconds + 1)  # Add 1 second buffer
                            continue  # Retry
                        else:
                            raise RuntimeError(
                                f"Groq API error: Rate limit exceeded (429). "
                                f"{error_message}. "
                                f"Please wait and try again later."
                            ) from exc
                    else:
                        # Generic 429 - use exponential backoff
                        wait_time = self.config.retry_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            "Groq API error 429: Rate limit. Waiting %d seconds before retry %d/%d...",
                            int(wait_time), attempt, self.config.retry_attempts
                        )
                        if attempt <= self.config.retry_attempts:
                            time.sleep(wait_time)
                            continue  # Retry
                        else:
                            raise RuntimeError(
                                f"Groq API error: Rate limit exceeded (429) after {attempt} attempts. "
                                f"{error_message}"
                            ) from exc
                
                logger.warning("Groq request attempt %s failed: %s", attempt, exc)
                if attempt > self.config.retry_attempts:
                    logger.exception("Groq request failed after %s attempts", attempt)
                    raise
                time.sleep(self.config.retry_backoff * attempt)

    def stream(self, prompt: str, *, model: Optional[str] = None, max_tokens: Optional[int] = None,
               temperature: Optional[float] = None, timeout: Optional[int] = None) -> Iterable[str]:
        """
        Generator that yields chunks as they arrive using Groq SDK streaming.
        """
        if self.client is None:
            raise RuntimeError("Groq client not initialized. Check API key and install groq package: pip install groq")

        model = model or self.config.model
        if not model:
            raise RuntimeError("Model not specified. Set GroqConfig.model or pass model parameter.")
        
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        timeout = timeout or self.config.timeout

        messages = [{"role": "user", "content": prompt}]
        
        call_kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if max_tokens is not None:
            call_kwargs["max_completion_tokens"] = max_tokens
        if temperature is not None:
            call_kwargs["temperature"] = temperature

        try:
            stream = self.client.chat.completions.create(**call_kwargs)
            for chunk in stream:
                if chunk.choices:
                    for choice in chunk.choices:
                        if choice.delta and choice.delta.content:
                            yield choice.delta.content
        except Exception as exc:
            # Check for 413 (Request too large) or rate limit errors
            error_msg = str(exc)
            error_dict = {}
            try:
                if hasattr(exc, 'response') and hasattr(exc.response, 'json'):
                    error_dict = exc.response.json()
                elif hasattr(exc, 'body'):
                    import json
                    error_dict = json.loads(exc.body) if isinstance(exc.body, str) else exc.body
            except Exception:
                pass
            
            is_413_error = "413" in error_msg or "Request too large" in error_msg or "tokens per minute" in error_msg
            is_429_error = "429" in error_msg or "Rate limit" in error_msg or "rate_limit" in error_msg
            
            if is_413_error:
                # Estimate prompt tokens
                prompt_tokens = len(messages[0].get("content", "")) // 4 if messages else 0
                logger.error(
                    "Groq API error 413: Request too large during streaming. "
                    "Estimated prompt tokens: ~%d. "
                    "Groq on_demand tier has a 6,000 TPM limit. "
                    "Please reduce prompt size or upgrade your Groq tier.",
                    prompt_tokens
                )
                raise RuntimeError(
                    f"Groq API error: Request too large (413). "
                    f"Estimated prompt tokens: ~{prompt_tokens}. "
                    f"Groq on_demand tier limit: 6,000 TPM. "
                    f"Please reduce prompt size or upgrade your Groq tier."
                ) from exc
            
            if is_429_error:
                error_info = error_dict.get("error", {}) if error_dict else {}
                error_message = error_info.get("message", error_msg)
                is_daily_limit = "tokens per day" in error_message.lower() or "TPD" in error_message
                
                if is_daily_limit:
                    logger.error(
                        "Groq API error 429: Daily token limit reached during streaming. %s. "
                        "Groq free tier limit: 500,000 tokens per day. "
                        "Please wait for the limit to reset or upgrade your tier.",
                        error_message
                    )
                    raise RuntimeError(
                        f"Groq API error: Daily token limit reached (429). "
                        f"{error_message}. "
                        f"Please wait for the limit to reset or upgrade your tier."
                    ) from exc
                else:
                    logger.error(
                        "Groq API error 429: Rate limit during streaming. %s",
                        error_message
                    )
                    raise RuntimeError(
                        f"Groq API error: Rate limit exceeded (429). {error_message}"
                    ) from exc
            
            logger.exception("Groq streaming failed: %s", exc)
            raise

    def available_providers(self) -> Dict[str, bool]:
        """
        Quick probe whether Groq SDK is installed and API key is set.
        """
        return {
            "groq_sdk_installed": _HAS_GROQ,
            "groq_api_key_set": bool(self.config.api_key),
        }
    
    def check_rate_limit_status(self) -> Dict[str, Any]:
        """
        Check current rate limit status by making a minimal test request.
        Returns info about current usage and limits.
        Note: This makes an API call, so use sparingly.
        """
        if not self.client:
            return {"error": "Groq client not initialized"}
        
        try:
            # Make a minimal test request to check limits
            resp = self.client.chat.completions.create(
                model=self.config.model or "llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=1
            )
            return {
                "status": "ok",
                "message": "Rate limits appear available"
            }
        except Exception as e:
            error_msg = str(e)
            # Try to extract rate limit info from error
            if "429" in error_msg or "Rate limit" in error_msg:
                import re
                # Extract usage info if available
                used_match = re.search(r'Used (\d+)', error_msg)
                limit_match = re.search(r'Limit (\d+)', error_msg)
                wait_match = re.search(r'Please try again in ([\d.]+)s', error_msg)
                
                result = {
                    "status": "rate_limited",
                    "error": error_msg
                }
                if used_match:
                    result["used"] = int(used_match.group(1))
                if limit_match:
                    result["limit"] = int(limit_match.group(1))
                if wait_match:
                    result["wait_seconds"] = float(wait_match.group(1))
                
                # Check if it's daily vs per-minute limit
                if "tokens per day" in error_msg or "TPD" in error_msg:
                    result["limit_type"] = "daily"
                    result["message"] = "Daily token limit reached. Wait until midnight UTC for reset."
                elif "tokens per minute" in error_msg or "TPM" in error_msg:
                    result["limit_type"] = "per_minute"
                    result["message"] = "Per-minute rate limit. Wait and retry."
                else:
                    result["limit_type"] = "unknown"
                    result["message"] = "Rate limit reached. Check Groq console for details."
                
                return result
            
            return {
                "status": "error",
                "error": error_msg
            }


# quick manual test when module run directly
if __name__ == "__main__":
    cfg = GroqConfig(api_key=os.environ.get("GROQ_API_KEY"), model=os.environ.get("DEFAULT_MODEL"))
    llm = GroqLLM(cfg)
    print("Probe:", llm.available_providers())
    try:
        print("Generate demo:")
        out = llm.generate("Write a 1-line greeting.")
        print(out.get("text"))
        print("\nStreaming demo:")
        for chunk in llm.stream("Write a short poem, one phrase per chunk."):
            print(chunk, end="", flush=True)
        print("\nDone.")
    except Exception as e:
        logger.exception("Demo failed: %s", e)
