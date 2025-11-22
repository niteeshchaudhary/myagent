# agent/llm/local_llm.py
"""
Local LLM adapter with Ollama and Groq support.

Provides:
- LocalLLM: unified interface for local LLM providers (ollama, groq).
- generate(prompt, **kwargs) -> dict with 'text' and provider metadata.
- stream(prompt, **kwargs) -> generator yielding partial chunks (best-effort).

Environment variables:
- OLLAMA_CLI: optional path to ollama binary (default: "ollama")
- OLLAMA_API_URL: optional HTTP URL for Ollama server (e.g. "http://localhost:11434")
- GROQ_API_URL: required for Groq HTTP usage (if selecting groq provider)
- GROQ_API_KEY: API key for Groq
- DEFAULT_LOCAL_LLM: default provider to use ("ollama" or "groq")

Notes:
- Ollama: this implementation supports calling the CLI (preferred for local use)
  using subprocess. If you run Ollama as an HTTP server, set OLLAMA_API_URL to
  call it instead (HTTP path attempts JSON API).
- Groq: uses HTTP request with Authorization: Bearer <GROQ_API_KEY>.
- Streaming support is best-effort: Ollama CLI stream returns partial stdout; Groq
  streaming depends on their API (we implement non-streaming with retries).
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, Optional, Union

import requests

# Attempt to import your project's logger if available; otherwise simple fallback
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


@dataclass
class LocalLLMConfig:
    provider: str = "ollama"  # 'ollama' or 'groq'
    model: Optional[str] = None
    timeout: int = 300
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    # low-level controls
    ollama_cli_path: str = os.environ.get("OLLAMA_CLI", "ollama")
    ollama_api_url: Optional[str] = os.environ.get("OLLAMA_API_URL")
    groq_api_url: Optional[str] = os.environ.get("GROQ_API_URL")
    groq_api_key: Optional[str] = os.environ.get("GROQ_API_KEY")
    retry_attempts: int = 2
    retry_backoff: float = 1.0


class LocalLLM:
    """
    Adapter for local LLM providers (Ollama CLI/HTTP and Groq HTTP).
    Usage:
        llm = LocalLLM(LocalLLMConfig(provider='ollama', model='llama2'))
        out = llm.generate("Hello world")
        print(out['text'])
    """

    def __init__(self, config: Optional[LocalLLMConfig] = None):
        self.config = config or LocalLLMConfig(provider=os.environ.get("DEFAULT_LOCAL_LLM", "ollama"))
        self._validate_config()

    def _validate_config(self):
        provider = (self.config.provider or "").lower()
        if provider not in ("ollama", "groq"):
            raise ValueError(f"Unsupported provider '{self.config.provider}'. Supported: 'ollama', 'groq'.")

        if provider == "groq" and not self.config.groq_api_url:
            raise ValueError("GROQ_API_URL environment variable (or config.groq_api_url) is required for groq provider.")
        if provider == "groq" and not self.config.groq_api_key:
            logger.warning("GROQ_API_KEY not set. Groq calls may fail if the API requires an API key.")

    # Public API
    def generate(self, prompt: str, *, model: Optional[str] = None, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, timeout: Optional[int] = None) -> Dict:
        """
        Synchronous generation returning the full text result.
        """
        provider = self.config.provider.lower()
        model = model or self.config.model
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        timeout = timeout or self.config.timeout

        logger.debug("LocalLLM.generate(provider=%s, model=%s, max_tokens=%s, temp=%s)",
                     provider, model, max_tokens, temperature)

        if provider == "ollama":
            return self._generate_ollama(prompt, model=model, max_tokens=max_tokens,
                                         temperature=temperature, timeout=timeout)
        elif provider == "groq":
            return self._generate_groq(prompt, model=model, max_tokens=max_tokens,
                                       temperature=temperature, timeout=timeout)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def stream(self, prompt: str, *, model: Optional[str] = None, **kwargs) -> Iterable[str]:
        """
        Yield partial chunks as they arrive. Streaming support is best-effort:
        - Ollama CLI streaming: yields lines from stdout.
        - Ollama HTTP / Groq HTTP: fallback to non-streaming (yields the final text once).
        """
        provider = self.config.provider.lower()
        if provider == "ollama":
            yield from self._stream_ollama(prompt, model=model, **kwargs)
        elif provider == "groq":
            # Groq streaming is not implemented generically here - fallback to full result
            result = self._generate_groq(prompt, model=model, **kwargs)
            yield result.get("text", "")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ---- Ollama implementations ----
    def _generate_ollama(self, prompt: str, *, model: Optional[str], max_tokens: Optional[int],
                         temperature: Optional[float], timeout: int) -> Dict:
        """
        Primary approach: try HTTP endpoint if provided; otherwise use the Ollama CLI.
        CLI approach uses: `ollama run <model> --prompt <prompt> --json`  (best-effort)
        Adjust flags as needed for your installed ollama version.
        """
        if self.config.ollama_api_url:
            logger.debug("Using Ollama HTTP API at %s", self.config.ollama_api_url)
            return self._ollama_http_generate(prompt, model=model, max_tokens=max_tokens,
                                              temperature=temperature, timeout=timeout)

        # prefer CLI path
        ollama_bin = shutil.which(self.config.ollama_cli_path) or self.config.ollama_cli_path
        if not shutil.which(ollama_bin):
            raise FileNotFoundError(
                f"Ollama CLI not found at '{ollama_bin}'. Install ollama or set OLLAMA_API_URL to use HTTP mode."
            )

        args = [ollama_bin, "run"]
        if model:
            args.append(model)
        # For newer ollama versions, you might use: ["ollama", "generate", model, "--prompt", prompt]
        # We do a "best-effort" set of args; users may need to adapt to their installed CLI.
        # We'll pass the prompt via stdin to avoid shell escaping issues.
        # Try to add flags for temperature / tokens if provided (best-effort)
        if temperature is not None:
            args.extend(["--temperature", str(temperature)])
        if max_tokens is not None:
            args.extend(["--max-tokens", str(max_tokens)])
        # If 'ollama run <model>' supports '--json' or '--pretty' you can add it here.
        # We'll attempt to request JSON if supported, but fallback to reading raw text.
        # Use subprocess to pass prompt via stdin for safety.
        try:
            logger.debug("Running subprocess: %s", " ".join(shlex.quote(a) for a in args))
            proc = subprocess.run(
                args,
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.exception("Ollama CLI timed out")
            raise

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            logger.error("Ollama CLI failed: rc=%s stderr=%s", proc.returncode, stderr.strip())
            raise RuntimeError(f"Ollama CLI error (rc={proc.returncode}): {stderr.strip()}")

        # Try to parse JSON response if possible; otherwise treat stdout as text
        # Ollama may return multiple JSON lines (one per chunk) or a single JSON object
        text = ""
        lines = stdout.strip().split("\n")
        
        # Try to parse each line as JSON and extract response/text
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    # Extract response or text field
                    if "response" in parsed:
                        text += parsed["response"]
                    elif "text" in parsed:
                        text += parsed["text"]
                    elif "content" in parsed:
                        text += parsed["content"]
                    # If no text field, skip this line (might be metadata)
                elif isinstance(parsed, str):
                    text += parsed
            except (json.JSONDecodeError, ValueError):
                # Not JSON, treat as plain text
                text += line + "\n"
        
        # If we didn't extract anything from JSON, use raw stdout
        if not text.strip():
            text = stdout.strip()

        return {"text": text, "provider": "ollama", "raw": stdout, "stderr": stderr}

    def _stream_ollama(self, prompt: str, *, model: Optional[str], max_tokens: Optional[int] = None,
                       temperature: Optional[float] = None, timeout: Optional[int] = None) -> Generator[str, None, None]:
        """
        Stream by spawning subprocess and yielding stdout as it arrives.
        This works well with 'ollama' CLI that flushes partial output.
        """
        timeout = timeout or self.config.timeout
        ollama_bin = shutil.which(self.config.ollama_cli_path) or self.config.ollama_cli_path
        if not shutil.which(ollama_bin):
            raise FileNotFoundError("Ollama CLI not found for streaming.")

        args = [ollama_bin, "run"]
        if model:
            args.append(model)
        if temperature is not None:
            args.extend(["--temperature", str(temperature)])
        if max_tokens is not None:
            args.extend(["--max-tokens", str(max_tokens)])

        # Start subprocess and stream stdout
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # send prompt to stdin then close so ollama reads it
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        except Exception:
            logger.exception("Failed to write prompt to ollama stdin")

        # Read stdout line by line
        assert proc.stdout is not None
        try:
            for raw_line in iter(proc.stdout.readline, b""):
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                
                # Ollama CLI outputs JSON lines with "response" field
                # Parse JSON and extract the response text
                try:
                    import json
                    data = json.loads(line)
                    if isinstance(data, dict) and "response" in data:
                        # Yield the response text chunk
                        response_text = data.get("response", "")
                        if response_text:
                            yield response_text
                    elif isinstance(data, dict) and "text" in data:
                        # Some Ollama versions use "text" instead of "response"
                        yield data.get("text", "")
                    else:
                        # If it's JSON but doesn't have response/text, skip it (might be metadata)
                        # Only yield if it's not JSON at all
                        if not isinstance(data, dict):
                            yield line
                        # Otherwise skip (it's JSON metadata without text)
                except (json.JSONDecodeError, ValueError):
                    # Not JSON, yield as plain text
                    yield line
            # wait for process to finish
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.exception("Ollama stream timed out")
            raise
        finally:
            # Drain stderr for logging
            try:
                if proc.stderr:
                    stderr = proc.stderr.read().decode("utf-8", errors="replace")
                    if stderr:
                        logger.debug("Ollama stderr: %s", stderr.strip())
            except Exception:
                pass

    def _ollama_http_generate(self, prompt: str, *, model: Optional[str], max_tokens: Optional[int],
                              temperature: Optional[float], timeout: int) -> Dict:
        """
        Calls Ollama HTTP API (if you run an Ollama server). The exact endpoint can differ;
        this is best-effort following typical "POST /api/generate" or "/v1/generate" patterns.
        You may need to adapt paths depending on Ollama version.
        """
        base = self.config.ollama_api_url.rstrip("/")
        # Try common endpoints in order
        candidates = [
            f"{base}/api/generate",
            f"{base}/v1/generate",
            f"{base}/completions",
            f"{base}/v1/completions",
        ]
        headers = {"Content-Type": "application/json"}
        payload = {"prompt": prompt}
        if model:
            payload["model"] = model
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        last_exc = None
        for url in candidates:
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            except Exception as e:
                last_exc = e
                logger.debug("Ollama HTTP candidate failed: %s -> %s", url, e)
                continue
            if not resp.ok:
                logger.debug("Ollama HTTP responded %s for %s: %s", resp.status_code, url, resp.text[:200])
                # try next endpoint
                last_exc = RuntimeError(f"{resp.status_code}: {resp.text}")
                continue
            # try parse
            try:
                j = resp.json()
                # heuristics to extract text
                text = ""
                if isinstance(j, dict):
                    for k in ("text", "completion", "output", "result"):
                        if k in j:
                            text = j[k]
                            break
                    # some APIs include choices array
                    if not text and "choices" in j and isinstance(j["choices"], list):
                        for c in j["choices"]:
                            if isinstance(c, dict):
                                text += c.get("text", "") or c.get("message", {}).get("content", "")
                else:
                    text = str(j)
            except Exception:
                text = resp.text
            return {"text": text, "provider": "ollama_http", "raw": resp.text}
        # if we reach here, all candidates failed
        raise RuntimeError(f"Ollama HTTP calls failed. Last error: {last_exc}")

    # ---- Groq implementations ----
    def _generate_groq(self, prompt: str, *, model: Optional[str], max_tokens: Optional[int],
                       temperature: Optional[float], timeout: int) -> Dict:
        """
        Simple Groq HTTP generation implementation.
        Expects GROQ_API_URL and GROQ_API_KEY (if required) in config.
        Adjust request/response parsing to match the active Groq API contract.
        """
        url = self.config.groq_api_url.rstrip("/")
        # Common Groq endpoint: /v1/models/{model}/completions or /v1/completions
        endpoints = []
        if model:
            endpoints.append(f"{url}/v1/models/{model}/completions")
            endpoints.append(f"{url}/models/{model}/completions")
        endpoints.extend([f"{url}/v1/completions", f"{url}/completions"])

        headers = {"Content-Type": "application/json"}
        if self.config.groq_api_key:
            headers["Authorization"] = f"Bearer {self.config.groq_api_key}"

        payload = {"prompt": prompt}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        last_exc = None
        for endpoint in endpoints:
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            except Exception as e:
                last_exc = e
                logger.debug("Groq candidate failed: %s -> %s", endpoint, e)
                continue
            if not resp.ok:
                logger.debug("Groq responded %s for %s: %s", resp.status_code, endpoint, resp.text[:200])
                last_exc = RuntimeError(f"{resp.status_code}: {resp.text}")
                continue
            # parse
            try:
                j = resp.json()
                text = ""
                # try a few common shapes
                if isinstance(j, dict):
                    if "text" in j:
                        text = j["text"]
                    elif "choices" in j and isinstance(j["choices"], list):
                        for c in j["choices"]:
                            if isinstance(c, dict):
                                text += c.get("text", "") or c.get("message", {}).get("content", "")
                    elif "output" in j:
                        text = j["output"]
                    else:
                        # fallback to full json dump
                        text = json.dumps(j)
                else:
                    text = str(j)
            except Exception:
                text = resp.text
            return {"text": text, "provider": "groq", "raw": resp.text}
        raise RuntimeError(f"Groq API calls failed. Last error: {last_exc}")

    # ---- Utilities ----
    def available_providers(self) -> Dict[str, bool]:
        """
        Quick capability probe: returns whether CLI/HTTP access appears available.
        """
        providers = {"ollama_cli": False, "ollama_http": False, "groq_http": False}
        # Ollama CLI
        try:
            if shutil.which(self.config.ollama_cli_path):
                providers["ollama_cli"] = True
        except Exception:
            pass
        # Ollama HTTP
        try:
            if self.config.ollama_api_url:
                # quick HEAD
                resp = requests.head(self.config.ollama_api_url, timeout=2)
                providers["ollama_http"] = resp.ok
        except Exception:
            pass
        # Groq HTTP
        try:
            if self.config.groq_api_url:
                resp = requests.head(self.config.groq_api_url, timeout=2)
                providers["groq_http"] = resp.ok
        except Exception:
            pass
        return providers


# Example usage in-code (not executed on import)
if __name__ == "__main__":
    # Quick demo when run directly (helpful for manual testing)
    cfg = LocalLLMConfig(provider=os.environ.get("DEFAULT_LOCAL_LLM", "ollama"), model=os.environ.get("DEFAULT_MODEL"))
    llm = LocalLLM(cfg)

    prompt = "Write a short, friendly hello message."
    try:
        print("Providers probe:", llm.available_providers())
        # Try streaming if using ollama CLI
        if cfg.provider == "ollama" and llm.available_providers().get("ollama_cli"):
            print("Streaming response:")
            for chunk in llm.stream(prompt, model=cfg.model):
                print(chunk, end="", flush=True)
            print("\n--- done")
        else:
            res = llm.generate(prompt, model=cfg.model)
            print("Result text:\n", res.get("text"))
    except Exception as e:
        logger.exception("Demo failed: %s", e)
