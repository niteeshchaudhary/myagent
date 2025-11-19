# agent/llm/model_selector.py
"""
Model selector for LLM adapters.

Responsibilities:
- Provide a simple, consistent way to obtain an LLM instance for a given provider name.
- Read defaults from configs/models.yaml (if present) or environment variables.
- Attempt to import common adapters: agent.llm.openai_llm.OpenAI_LLM and agent.llm.local_llm.LocalLLM.
- Expose helper functions to list supported providers and probe availability.

Usage:
    from agent.llm.model_selector import get_llm

    llm = get_llm(provider="ollama")  # or provider=None to pick default
    out = llm.generate("Hello")
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

# Try to use your project's logger, fallback to std logging
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

# Optional yaml loader
try:
    import yaml  # PyYAML
except Exception:
    yaml = None  # we'll fall back to env vars if PyYAML is not available


# Define a small protocol so static type checkers can reason about adapter shape.
class BaseLLM(Protocol):
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        ...

    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        ...


@dataclass
class ModelSelectorConfig:
    """
    Config for the selector. You can override these programmatically.
    - models_file: path to models.yaml describing default models per provider.
    - default_provider: provider name to prefer if none supplied (env DEFAULT_LOCAL_LLM or 'openai' fallback).
    """
    models_file: Optional[str] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "configs", "models.yaml")
    default_provider: str = os.environ.get("DEFAULT_LOCAL_LLM", "openai")


# Internal mapping of canonical provider name -> (module_path, class_name)
_PROVIDER_REGISTRY = {
    "openai": ("agent.llm.openai_llm", "OpenAI_LLM"),
    "ollama": ("agent.llm.local_llm", "LocalLLM"),
    "groq": ("agent.llm.groq_llm", "GroqLLM"),
    "local": ("agent.llm.local_llm", "LocalLLM"),
}


def _load_models_yaml(path: Optional[str]) -> Dict[str, Any]:
    """
    Load models mapping from YAML if available. Returns a dict mapping provider->default_model.
    If yaml is not installed or file missing/invalid, returns empty dict.
    """
    if not path:
        return {}
    # normalize path relative to repo root if a relative path was provided
    path = os.path.abspath(path)
    if not os.path.exists(path):
        logger.debug("models.yaml not found at %s; skipping", path)
        return {}
    if yaml is None:
        logger.warning("PyYAML not installed; cannot read models.yaml at %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                logger.debug("models.yaml content is not a mapping; ignoring")
                return {}
            return data
    except Exception as e:
        logger.exception("Failed to read models.yaml: %s", e)
        return {}


def list_supported_providers() -> List[str]:
    """Return a list of provider keys this selector knows about."""
    return list(_PROVIDER_REGISTRY.keys())


def _import_adapter(module_path: str, class_name: str):
    """
    Import and return adapter class. Raises ImportError with helpful message if not available.
    """
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        raise ImportError(f"Could not import module '{module_path}': {e}") from e

    try:
        adapter_cls = getattr(module, class_name)
    except AttributeError as e:
        raise ImportError(f"Module '{module_path}' does not expose class '{class_name}': {e}") from e

    return adapter_cls


def probe_provider_availability(provider: str, config: Optional[ModelSelectorConfig] = None) -> Tuple[bool, str]:
    """
    Attempt to probe whether a provider appears available.
    Returns (available: bool, message: str).
    If a provider adapter exposes an 'available_providers' method, call that to get a quick probe.
    """
    provider = (provider or "").lower()
    cfg = config or ModelSelectorConfig()

    if provider not in _PROVIDER_REGISTRY:
        return False, f"Unknown provider '{provider}'"

    module_path, class_name = _PROVIDER_REGISTRY[provider]
    try:
        adapter_cls = _import_adapter(module_path, class_name)
    except ImportError as e:
        return False, f"Import error: {e}"

    # instantiate with minimal config if possible
    try:
        # many adapters accept a config dataclass or no args
        instance = None
        try:
            instance = adapter_cls()  # try no-arg constructor
        except TypeError:
            # try passing an inferred config dataclass
            try:
                instance = adapter_cls(None)
            except Exception:
                # final try: instantiate with keyword config if it accepts it
                instance = adapter_cls(config=cfg)  # type: ignore
    except Exception as e:
        logger.debug("Could not instantiate adapter for probe: %s", e)
        return False, f"Adapter instantiation failed: {e}"

    # if adapter exposes available_providers, call it
    try:
        if hasattr(instance, "available_providers"):
            avail = instance.available_providers()
            # If the adapter returns a dict of booleans, interpret it; else truthiness
            if isinstance(avail, dict):
                # if any True, provider is available (e.g., ollama_cli or ollama_http)
                if any(avail.values()):
                    return True, f"Provider '{provider}' appears available: {avail}"
                else:
                    return False, f"Provider '{provider}' probes returned: {avail}"
            else:
                return (bool(avail), f"Provider '{provider}' probe returned: {avail}")
        else:
            # no probe method - assume okay if import succeeded
            return True, f"Provider '{provider}' imported successfully (no probe available)"
    except Exception as e:
        logger.debug("Probe call failed: %s", e)
        return False, f"Probe call raised exception: {e}"


def select_provider(preferences: Iterable[str], config: Optional[ModelSelectorConfig] = None) -> str:
    """
    Choose the first available provider from the iterable of provider names.
    Preferences are checked in order; returns the provider name chosen or raises RuntimeError if none available.
    """
    cfg = config or ModelSelectorConfig()
    for p in preferences:
        p_norm = (p or "").lower()
        if p_norm not in _PROVIDER_REGISTRY:
            logger.debug("Skipping unknown provider in preferences: %s", p)
            continue
        ok, msg = probe_provider_availability(p_norm, cfg)
        logger.debug("Probe %s -> %s (%s)", p_norm, ok, msg)
        if ok:
            logger.info("Selected provider: %s", p_norm)
            return p_norm
    raise RuntimeError(f"No available provider found in preferences: {list(preferences)}")


def get_default_provider_from_yaml(config: Optional[ModelSelectorConfig] = None) -> Optional[str]:
    """
    Read the default_model from models.yaml and return its provider.
    Returns None if not found or if YAML structure doesn't match.
    """
    cfg = config or ModelSelectorConfig()
    models_map = _load_models_yaml(cfg.models_file)
    
    # Check for default_model at top level
    default_model_name = models_map.get("default_model")
    if not default_model_name:
        return None
    
    # Look up the model in the models section
    models_section = models_map.get("models", {})
    if not isinstance(models_section, dict):
        return None
    
    model_config = models_section.get(default_model_name)
    if not isinstance(model_config, dict):
        return None
    
    # Extract provider from model config
    provider = model_config.get("provider")
    if provider:
        return provider.lower()
    
    return None


def get_default_model_for_provider(provider: str, config: Optional[ModelSelectorConfig] = None) -> Optional[str]:
    """
    Look up the default model for a provider from configs/models.yaml or environment variables.
    Returns None if not found.
    """
    cfg = config or ModelSelectorConfig()
    models_map = _load_models_yaml(cfg.models_file)
    provider = (provider or "").lower()
    
    # First, try to find model by provider in the models section
    models_section = models_map.get("models", {})
    if isinstance(models_section, dict):
        for model_name, model_config in models_section.items():
            if isinstance(model_config, dict) and model_config.get("provider", "").lower() == provider:
                # Return the model name if it matches the provider
                # Also check if there's a model_name or model field
                model_val = model_config.get("model_name") or model_config.get("model") or model_name
                if model_val:
                    return model_val
    
    # try YAML - old format where provider is a top-level key
    if provider in models_map:
        val = models_map[provider]
        # YAML mapping might be nested; accept string or mapping with 'default'
        if isinstance(val, str):
            return val
        if isinstance(val, dict) and "default" in val:
            return val["default"]
    
    # try environment variable fallback: MODEL_<PROVIDER> e.g. MODEL_OPENAI
    env_key = f"MODEL_{provider.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]
    # other common env names
    if "DEFAULT_MODEL" in os.environ:
        return os.environ["DEFAULT_MODEL"]
    return None


def get_llm(provider: Optional[str] = None, *, config: Optional[ModelSelectorConfig] = None, **kwargs) -> BaseLLM:
    """
    Return an instantiated LLM adapter for the requested provider.

    - provider: one of supported keys ('openai', 'ollama', 'groq', 'local').
      If None, the selector will try ModelSelectorConfig.default_provider or fall back to 'openai'.
    - kwargs: passed-through to the adapter constructor (e.g., model, api keys, timeout).
    """
    cfg = config or ModelSelectorConfig()
    
    # If no provider specified, try to get it from YAML default_model
    if not provider:
        yaml_provider = get_default_provider_from_yaml(cfg)
        if yaml_provider:
            requested = yaml_provider
            logger.debug("Using provider from default_model in YAML: %s", requested)
        else:
            requested = (cfg.default_provider or "openai").lower()
    else:
        requested = provider.lower()

    if requested not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider '{requested}'. Supported: {list_supported_providers()}")

    module_path, class_name = _PROVIDER_REGISTRY[requested]
    try:
        adapter_cls = _import_adapter(module_path, class_name)
    except ImportError as e:
        # raise a clearer message with hint about installing deps
        raise RuntimeError(f"Failed to import adapter for provider '{requested}': {e}") from e

    # Load YAML to get model configuration
    models_map = _load_models_yaml(cfg.models_file)
    default_model_name = models_map.get("default_model")
    models_section = models_map.get("models", {})
    
    # Find the model config from YAML for this provider
    model_config = None
    if default_model_name and isinstance(models_section, dict):
        candidate = models_section.get(default_model_name)
        if isinstance(candidate, dict) and candidate.get("provider", "").lower() == requested:
            model_config = candidate
    
    # Determine model default if not supplied in kwargs
    if not kwargs.get("model"):
        # First check if there's a default_model in YAML that matches this provider
        if model_config:
            # Use the model_name or model field from the config
            model_val = model_config.get("model_name") or model_config.get("model")
            if model_val:
                kwargs["model"] = model_val
                logger.debug("Using default model from YAML for provider %s: %s", requested, model_val)
        
        # If we didn't set it above, try the provider-based lookup
        if not kwargs.get("model"):
            default_model = get_default_model_for_provider(requested, cfg)
            if default_model:
                kwargs["model"] = default_model
                logger.debug("Using default model for provider %s: %s", requested, default_model)
    
    # For LocalLLM (ollama/local), extract additional config from YAML
    if requested in ("ollama", "local") and model_config:
        # Extract additional config fields
        if "endpoint" in model_config and requested == "ollama":
            kwargs.setdefault("ollama_api_url", model_config["endpoint"])
        if "temperature" in model_config:
            kwargs.setdefault("temperature", model_config["temperature"])
        if "max_tokens" in model_config:
            kwargs.setdefault("max_tokens", model_config["max_tokens"])
    
    # For Groq, extract additional config from YAML
    if requested == "groq" and model_config:
        if "temperature" in model_config:
            kwargs.setdefault("temperature", model_config["temperature"])
        if "max_tokens" in model_config:
            kwargs.setdefault("max_tokens", model_config["max_tokens"])
        if "api_key_env" in model_config:
            env_key = model_config["api_key_env"]
            api_key = os.environ.get(env_key)
            if api_key:
                kwargs.setdefault("api_key", api_key)

    # Construct adapter instance. We attempt a few common constructor signatures:
    #  - adapter_cls(config: ModelConfigDataclass)
    #  - adapter_cls(LocalLLMConfig / dict kwargs)
    #  - adapter_cls() and then set attributes
    instance = None
    
    # Special handling for GroqLLM which expects GroqConfig
    if requested == "groq":
        try:
            from agent.llm.groq_llm import GroqConfig
            # Create config with extracted kwargs
            config_kwargs = {}
            # Ensure model is set - it's required
            model_val = kwargs.pop("model", None)
            if not model_val and model_config:
                # Fallback: try to get model from model_config again
                model_val = model_config.get("model_name") or model_config.get("model")
            if model_val:
                config_kwargs["model"] = model_val
                logger.info("Using model '%s' for provider '%s'", model_val, requested)
            else:
                logger.warning("No model specified for provider '%s'. This may cause errors.", requested)
            if "temperature" in kwargs:
                config_kwargs["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                config_kwargs["max_tokens"] = kwargs.pop("max_tokens")
            if "timeout" in kwargs:
                config_kwargs["timeout"] = kwargs.pop("timeout")
            if "api_key" in kwargs:
                config_kwargs["api_key"] = kwargs.pop("api_key")
            
            groq_config = GroqConfig(**config_kwargs)
            instance = adapter_cls(groq_config)
            logger.debug("Instantiated GroqLLM adapter with config: %s", config_kwargs)
            return instance
        except Exception as e:
            logger.debug("Failed to instantiate GroqLLM with config: %s", e)
    
    # Special handling for LocalLLM which expects LocalLLMConfig
    if requested in ("ollama", "local"):
        try:
            from agent.llm.local_llm import LocalLLMConfig
            # Create config with provider and extracted kwargs
            config_kwargs = {"provider": requested}
            # Ensure model is set - it's required
            model_val = kwargs.pop("model", None)
            if not model_val and model_config:
                # Fallback: try to get model from model_config again
                model_val = model_config.get("model_name") or model_config.get("model")
            if model_val:
                config_kwargs["model"] = model_val
                logger.info("Using model '%s' for provider '%s'", model_val, requested)
            else:
                logger.warning("No model specified for provider '%s'. This may cause errors.", requested)
            if "temperature" in kwargs:
                config_kwargs["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                config_kwargs["max_tokens"] = kwargs.pop("max_tokens")
            if "timeout" in kwargs:
                config_kwargs["timeout"] = kwargs.pop("timeout")
            if "ollama_api_url" in kwargs:
                config_kwargs["ollama_api_url"] = kwargs.pop("ollama_api_url")
            
            llm_config = LocalLLMConfig(**config_kwargs)
            instance = adapter_cls(llm_config)
            logger.debug("Instantiated LocalLLM adapter with config: %s", config_kwargs)
            return instance
        except Exception as e:
            logger.debug("Failed to instantiate LocalLLM with config: %s", e)
    
    # 1) Try passing a config dataclass if adapter expects one
    try:
        instance = adapter_cls(**kwargs)  # most common: accept keyword args like model, timeout
        logger.debug("Instantiated adapter %s with kwargs %s", adapter_cls, kwargs)
        return instance
    except TypeError as e:
        logger.debug("Adapter %s did not accept kwargs directly: %s", adapter_cls, e)
    except Exception as e:
        logger.debug("Adapter raised when called with kwargs: %s", e)

    # 2) Try no-arg constructor then set attributes
    try:
        instance = adapter_cls()
        # set common attributes if present
        for k, v in kwargs.items():
            try:
                setattr(instance, k, v)
            except Exception:
                # ignore attributes that cannot be set
                pass
        logger.debug("Instantiated adapter %s with fallback no-arg and set attrs", adapter_cls)
        return instance
    except Exception as e:
        logger.debug("Adapter %s did not instantiate with no-arg: %s", adapter_cls, e)

    # 3) Try single-arg constructor passing a dict or config object
    try:
        instance = adapter_cls(kwargs)
        return instance
    except Exception as e:
        logger.exception("Failed to instantiate adapter %s with fallback attempts: %s", adapter_cls, e)
        raise RuntimeError(f"Failed to instantiate LLM adapter for provider '{requested}': {e}") from e


# Example quick CLI-style utility (not executed when imported by tests)
def _demo_select_and_run():
    """
    Quick demo utility. Not intended for unit tests; run manually.
    """
    from pprint import pprint

    selector_cfg = ModelSelectorConfig()
    print("Supported providers:", list_supported_providers())
    # try preferences from environment
    preferred = os.environ.get("PREFERRED_LLMS", "ollama,openai").split(",")
    try:
        chosen = select_provider(preferred, selector_cfg)
        print("Chosen provider:", chosen)
        llm = get_llm(chosen, config=selector_cfg)
        print("Probe availability:", probe_provider_availability(chosen, selector_cfg))
        pprint(llm.generate("Say hello in one sentence."))
    except Exception as e:
        print("Demo failed:", e)


if __name__ == "__main__":
    _demo_select_and_run()
