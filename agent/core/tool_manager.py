# agent/core/tool_manager.py
"""
ToolManager for the coding agent.

Responsibilities
- Discover and load tool classes from the `agent.tools` package.
- Provide a uniform runtime API to run tools by name:
    - tool_manager.run(name, input=..., timeout=...)
    - tool_manager.get_tool(name)
    - tool_manager.list_tools()
- Allow registering/unregistering custom tool instances programmatically.
- Normalize different tool return shapes into a consistent dict with at least 'output' key.

Expectations for tool modules
- Each tool module should expose either:
    - a class named `Tool` (preferred), or
    - one or more classes whose names end with `Tool` (e.g., `ShellTool`), or
    - a factory function `get_tool()` that returns an instance.
- Tool instances are expected to implement at least one of:
    - `run(input, **kwargs)` or `execute(input, **kwargs)` (sync).
  The ToolManager will attempt both.

Notes
- This manager is defensive: it will load best-effort and skip modules that fail to import.
- It does not impose concurrency — tools that need async or subprocess control should handle it themselves.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# Prefer the project's logger if available
try:
    from agent.utils.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout if hasattr(sys, "stdout") else None)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ToolMeta:
    """Lightweight metadata for a tool."""
    name: str
    module: str
    cls_name: Optional[str] = None
    instance: Any = None
    description: Optional[str] = None


class ToolManager:
    """
    Manages discovery, registration, and execution of tools.

    Example:
        tm = ToolManager()
        tm.load_tools_from_package("agent.tools")
        tm.register_tool("echo", MyEchoTool())
        res = tm.run("shell", input="echo hello")
    """

    def __init__(self, tool_configs: Optional[Dict[str, Any]] = None, mcp_manager: Optional[Any] = None):
        # mapping name -> ToolMeta
        self._tools: Dict[str, ToolMeta] = {}
        # Tool configurations from tools.yaml
        self._tool_configs: Dict[str, Any] = tool_configs or {}
        # MCP manager for external MCP tools
        self._mcp_manager = mcp_manager

    # -----------------------
    # Discovery & registration
    # -----------------------
    def register_tool(self, name: str, instance: Any, module: Optional[str] = None, cls_name: Optional[str] = None,
                      description: Optional[str] = None) -> None:
        """
        Register a tool instance under a canonical name.
        Overwrites any existing tool with same name.
        """
        meta = ToolMeta(name=name, module=module or getattr(instance, "__module__", "unknown"), cls_name=cls_name, instance=instance, description=description)
        self._tools[name] = meta
        logger.info("Registered tool '%s' (module=%s, class=%s)", name, meta.module, cls_name or instance.__class__.__name__)

    def unregister_tool(self, name: str) -> bool:
        """Remove a registered tool. Returns True if removed."""
        if name in self._tools:
            del self._tools[name]
            logger.info("Unregistered tool '%s'", name)
            return True
        return False

    def list_tools(self) -> List[ToolMeta]:
        """Return metadata for all registered tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[Any]:
        """Return the tool instance or None if not found."""
        meta = self._tools.get(name)
        return meta.instance if meta else None

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def load_tools_from_package(self, package_name: str = "agent.tools", *, prefix_strip: bool = True) -> int:
        """
        Discover tool modules inside the given package and attempt to load them.

        Strategy:
        - import the package
        - iterate its submodules using pkgutil.walk_packages
        - for each submodule, attempt to import and discover a tool instance by:
            1) looking for function `get_tool()` -> call it
            2) looking for top-level `Tool` class -> instantiate without args
            3) looking for any class whose name endswith 'Tool' -> instantiate first found
            4) looking for module-level instances named 'tool' or 'TOOL'
        - when a tool instance is found, register it under a canonical name derived from module or class.
        - return count of registered tools from the package
        """
        try:
            pkg = importlib.import_module(package_name)
        except Exception as e:
            logger.exception("Failed to import tools package %s: %s", package_name, e)
            return 0

        count = 0
        # Walk modules
        pkg_path = getattr(pkg, "__path__", None)
        if not pkg_path:
            logger.debug("Package %s has no __path__; skipping discovery", package_name)
            return 0

        for finder, modname, ispkg in pkgutil.walk_packages(pkg_path, prefix=package_name + "."):
            # skip subpackages for now (but they will still be walked)
            try:
                module = importlib.import_module(modname)
            except Exception as e:
                logger.debug("Failed to import tool module %s: %s", modname, e)
                continue

            try:
                instance, canonical_name, cls_name, desc = self._discover_tool_in_module(module, modname, prefix_strip)
            except Exception as e:
                logger.debug("Error discovering tool in module %s: %s", modname, e)
                continue

            if instance is None:
                logger.debug("No tool discovered in %s", modname)
                continue

            # Avoid name collisions: prefer explicit canonical name
            if canonical_name in self._tools:
                # if the same instance is already registered, skip
                existing = self._tools[canonical_name]
                if existing.module == modname and existing.cls_name == cls_name:
                    logger.debug("Tool %s from %s already registered; skipping", canonical_name, modname)
                    continue
                # otherwise, append module suffix
                orig_name = canonical_name
                i = 1
                while canonical_name in self._tools:
                    canonical_name = f"{orig_name}_{i}"
                    i += 1
                logger.info("Name collision for tool '%s'; registering as '%s'", orig_name, canonical_name)

            # Apply tool configuration if available
            self._apply_tool_config(canonical_name, instance)
            
            self.register_tool(canonical_name, instance, module=modname, cls_name=cls_name, description=desc)
            count += 1

        logger.info("Discovered and registered %d tools from package %s", count, package_name)
        return count
    
    def _apply_tool_config(self, tool_name: str, tool_instance: Any) -> None:
        """
        Apply configuration from tools.yaml to a tool instance.
        Sets attributes on the tool instance based on config.
        """
        if not self._tool_configs:
            return
        
        # Get tool config - check both direct name and enabled_tools list
        tool_config = None
        if "tools" in self._tool_configs:
            tools_section = self._tool_configs["tools"]
            if isinstance(tools_section, dict):
                tool_config = tools_section.get(tool_name)
        
        if not tool_config or not isinstance(tool_config, dict):
            return
        
        logger.debug("Applying config to tool '%s': %s", tool_name, tool_config)
        
        # Apply configuration attributes to tool instance
        for key, value in tool_config.items():
            try:
                # Skip module/class keys (they're metadata, not config)
                if key in ("module", "class"):
                    continue
                # Set attribute on tool instance
                setattr(tool_instance, key, value)
                logger.debug("Set %s.%s = %s", tool_name, key, value)
            except Exception as e:
                logger.debug("Could not set %s.%s: %s", tool_name, key, e)
    
    def load_tool_configs(self, configs: Dict[str, Any]) -> None:
        """
        Load tool configurations (typically from tools.yaml).
        
        Args:
            configs: Dictionary containing tool configurations, typically from UnifiedConfig.tools
        """
        self._tool_configs = configs
        # Re-apply configs to already registered tools
        for name, meta in self._tools.items():
            if meta.instance:
                self._apply_tool_config(name, meta.instance)
    
    def load_mcp_tools(self) -> int:
        """
        Load tools from MCP servers via MCP manager.
        
        Returns:
            Number of MCP tools registered
        """
        if not self._mcp_manager:
            return 0
        
        count = 0
        try:
            mcp_tools = self._mcp_manager.list_tools()
            for tool_wrapper in mcp_tools:
                # Register MCP tool with ToolManager
                self.register_tool(
                    tool_wrapper.name,
                    tool_wrapper,
                    module="agent.mcp",
                    cls_name="MCPToolWrapper",
                    description=tool_wrapper.description
                )
                count += 1
            logger.info("Registered %d MCP tools", count)
        except Exception as e:
            logger.exception("Failed to load MCP tools: %s", e)
        
        return count

    def _discover_tool_in_module(self, module, modname: str, prefix_strip: bool) -> Tuple[Optional[Any], Optional[str], Optional[str], Optional[str]]:
        """
        Inspect the module object and attempt to find/instantiate a tool instance.
        Returns: (instance_or_None, canonical_name, cls_name_or_None, description_or_None)
        """
        # 1) factory function get_tool()
        if hasattr(module, "get_tool") and callable(getattr(module, "get_tool")):
            try:
                inst = getattr(module, "get_tool")()
                name = getattr(inst, "name", None) or self._derive_name_from_module(modname, prefix_strip)
                desc = getattr(inst, "description", None)
                return inst, name, inst.__class__.__name__, desc
            except Exception as e:
                logger.debug("get_tool() in %s failed: %s", modname, e)

        # 2) top-level Tool class
        if hasattr(module, "Tool") and inspect.isclass(getattr(module, "Tool")):
            cls = getattr(module, "Tool")
            try:
                inst = cls()  # try no-arg constructor
                name = getattr(inst, "name", None) or self._derive_name_from_class(cls, modname, prefix_strip)
                desc = getattr(inst, "description", None) or (cls.__doc__ or "").splitlines()[0] if cls.__doc__ else None
                return inst, name, cls.__name__, desc
            except Exception as e:
                logger.debug("Instantiating Tool class in %s failed: %s", modname, e)

        # 3) any class name ending with 'Tool'
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if obj.__name__.endswith("Tool"):
                try:
                    inst = obj()
                    name = getattr(inst, "name", None) or self._derive_name_from_class(obj, modname, prefix_strip)
                    desc = getattr(inst, "description", None) or (obj.__doc__ or "").splitlines()[0] if obj.__doc__ else None
                    return inst, name, obj.__name__, desc
                except Exception as e:
                    logger.debug("Failed to instantiate %s in %s: %s", obj.__name__, modname, e)
                    continue

        # 4) module-level instances named 'tool' or 'TOOL'
        for candidate in ("tool", "TOOL"):
            if hasattr(module, candidate):
                inst = getattr(module, candidate)
                name = getattr(inst, "name", None) or self._derive_name_from_module(modname, prefix_strip)
                desc = getattr(inst, "description", None)
                return inst, name, inst.__class__.__name__, desc

        return None, None, None, None

    @staticmethod
    def _derive_name_from_module(modname: str, prefix_strip: bool) -> str:
        # e.g. agent.tools.shell_tool -> shell_tool -> shell
        base = modname.split(".")[-1]
        if prefix_strip and base.endswith("_tool"):
            base = base[: -len("_tool")]
        return base

    @staticmethod
    def _derive_name_from_class(cls, modname: str, prefix_strip: bool) -> str:
        name = cls.__name__
        # ShellTool -> shell
        if name.endswith("Tool"):
            return name[: -len("Tool")].lower()
        # fallback to module-derived name
        return ToolManager._derive_name_from_module(modname, prefix_strip)

    # -----------------------
    # Execution
    # -----------------------
    def run(self, name: str, *, input: Any = None, timeout: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute a registered tool by name.

        Returns a normalized dict with keys:
          - output (str)
          - success (bool)
          - error (optional str)
          - meta (optional dict)

        Behavior:
        - try `tool.run(input, **kwargs)` if present
        - else try `tool.execute(input, **kwargs)`
        - else if tool is callable, call it directly
        - if the returned value is dict-like, normalize; else coerce to string
        """
        meta = {"tool": name, "start_time": time.time()}
        if name not in self._tools:
            msg = f"Tool '{name}' not found"
            logger.warning(msg)
            return {"output": "", "success": False, "error": msg, "meta": meta}

        tool = self._tools[name].instance
        try:
            # prefer run()
            if hasattr(tool, "run") and callable(getattr(tool, "run")):
                res = tool.run(input, **kwargs) if self._accepts_input_arg(tool.run) else tool.run(**kwargs)
            elif hasattr(tool, "execute") and callable(getattr(tool, "execute")):
                res = tool.execute(input, **kwargs) if self._accepts_input_arg(tool.execute) else tool.execute(**kwargs)
            elif callable(tool):
                # callable tool: pass input if function accepts it
                res = tool(input) if self._callable_accepts_arg(tool) else tool()
            else:
                raise RuntimeError(f"Tool '{name}' is not runnable (no run/execute/callable).")

            norm = self._normalize_tool_result(res)
            norm.setdefault("success", True)
            norm["meta"] = {**meta, "end_time": time.time()}
            return norm
        except Exception as e:
            logger.exception("Tool '%s' raised an exception: %s", name, e)
            return {"output": "", "success": False, "error": str(e), "meta": {**meta, "end_time": time.time()}}

    @staticmethod
    def _normalize_tool_result(res: Any) -> Dict[str, Any]:
        """
        Normalize tool return shapes into a dict with 'output' and optional metadata.
        Acceptable incoming shapes:
        - None -> {'output': '', 'success': False}
        - dict -> if contains 'output' / 'stdout' / 'result' / 'text', map to 'output'
        - str/bytes/int -> coerce to string in 'output'
        """
        if res is None:
            return {"output": "", "success": False}
        if isinstance(res, dict):
            if "output" in res:
                return res
            for key in ("stdout", "result", "text"):
                if key in res:
                    out = res[key]
                    return {"output": out, **{k: v for k, v in res.items() if k != key}}
            # no recognized output key: stringify and return preserved dict
            return {"output": str(res), **res}
        if isinstance(res, (str, bytes)):
            out = res.decode("utf-8") if isinstance(res, bytes) else res
            return {"output": out}
        # fallback: coerce to str
        return {"output": str(res)}

    @staticmethod
    def _accepts_input_arg(fn: Callable) -> bool:
        """
        Determine whether function `fn` accepts a first positional 'input' argument.
        """
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            if not params:
                return False
            # accept if first parameter is positional or positional_or_keyword
            first = params[0]
            return first.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        except Exception:
            return True

    @staticmethod
    def _callable_accepts_arg(fn: Callable) -> bool:
        """Simpler check whether a callable accepts at least one positional argument."""
        return ToolManager._accepts_input_arg(fn)

    # -----------------------
    # Convenience helpers
    # -----------------------
    def run_all(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run a sequence of tasks of shape [{"tool": "<name>", "input": ...}, ...] and return list of results.
        """
        out = []
        for t in tasks:
            name = t.get("tool") or t.get("name")
            if not name:
                out.append({"output": "", "success": False, "error": "missing tool name"})
                continue
            res = self.run(name, input=t.get("input"), timeout=t.get("timeout"))
            out.append(res)
        return out


# If run as a script, demonstrate discovery (manual test)
if __name__ == "__main__":
    tm = ToolManager()
    n = tm.load_tools_from_package("agent.tools")
    print(f"Loaded {n} tools:")
    for m in tm.list_tools():
        print(f"- {m.name} ({m.module}) : {m.cls_name} -- {m.description or ''}")
