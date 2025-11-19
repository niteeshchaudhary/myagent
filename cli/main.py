# cli/main.py
"""
CLI entrypoint for the coding-agent.

Commands (Typer):
- run       : Run a single-shot prompt through the agent loop.
- repl      : Start interactive REPL loop.
- probe     : Probe available LLM providers and tools.
- list-tools: List discovered tools.
- memory    : Show / clear memory.
- select    : Show selected/default LLM provider and model.

This CLI uses Typer for a pleasant UX. If you prefer argparse, replace Typer usage.
"""
from __future__ import annotations

import os
import sys
import json
import traceback
from typing import Optional

import typer
from dotenv import load_dotenv
load_dotenv()

# Prefer the project's logger if available
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

# Import core pieces of the agent
from agent.core.agent_loop import AgentLoop, AgentLoopConfig
from agent.core.tool_manager import ToolManager
from agent.core.planner import Planner, PlannerConfig
from agent.core.memory import Memory, MemoryConfig
from agent.llm.model_selector import get_llm, list_supported_providers, probe_provider_availability, ModelSelectorConfig

app = typer.Typer(help="Coding Agent CLI")

# Shared factory functions
def _build_tool_manager() -> ToolManager:
    tm = ToolManager()
    # attempt to load tools from the conventional package; ignore failures
    try:
        tm.load_tools_from_package("agent.tools")
    except Exception as e:
        logger.debug("Tool discovery failed: %s", e)
    return tm

def _build_memory(persist_path: Optional[str] = None, enable_persistence: bool = False, max_items: Optional[int] = None) -> Memory:
    cfg = MemoryConfig(
        persist_path=persist_path,
        enable_persistence=enable_persistence,
        max_items=max_items,
    )
    return Memory(cfg)

def _build_planner(llm=None, model: Optional[str] = None, prefer_structured: bool = True) -> Planner:
    cfg = PlannerConfig(model=model, prefer_structured=prefer_structured)
    return Planner(llm=llm, config=cfg)

def _build_llm(provider: Optional[str] = None, model: Optional[str] = None):
    try:
        # model_selector.get_llm accepts model kw
        return get_llm(provider, model=model)
    except Exception as e:
        logger.warning("Could not create LLM for provider=%s model=%s: %s", provider, model, e)
        raise

def _build_agent(llm_provider: Optional[str], llm_model: Optional[str], persist_path: Optional[str], enable_persistence: bool, max_memory_items: Optional[int], verbose: bool = True, review_mode: bool = False) -> AgentLoop:
    tm = _build_tool_manager()
    mem = _build_memory(persist_path=persist_path, enable_persistence=enable_persistence, max_items=max_memory_items)
    try:
        llm = _build_llm(llm_provider, llm_model)
    except Exception:
        llm = None
    planner = _build_planner(llm=llm, model=llm_model)
    cfg = AgentLoopConfig(verbose=verbose, review_mode=review_mode)
    return AgentLoop(planner=planner, memory=mem, tool_manager=tm, llm=llm, config=cfg)

# ------------------------
# CLI commands
# ------------------------
@app.command()
def run(
    prompt: str = typer.Option(..., help="Prompt to run through the agent loop."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider name (e.g. 'openai', 'ollama', 'groq')."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name to use for the LLM."),
    persist_path: Optional[str] = typer.Option(None, "--persist-path", help="Path to persist memory JSON."),
    enable_persistence: bool = typer.Option(False, "--persist", help="Enable memory persistence to file."),
    max_memory_items: Optional[int] = typer.Option(None, "--max-memory", help="Cap memory to this many items."),
    verbose: bool = typer.Option(True, "--verbose/--no-verbose", help="Show streaming output when available."),
    review_mode: bool = typer.Option(False, "--review/--no-review", help="Enable review mode: changes require approval before applying (like IDE diff review)."),
):
    """
    Run a single prompt through the agent loop and print the final result.
    """
    try:
        loop = _build_agent(provider, model, persist_path, enable_persistence, max_memory_items, verbose, review_mode)
        res = loop.run_once(prompt)
        final = res.get("final") or res.get("raw") or ""
        if isinstance(final, dict):
            typer.echo(json.dumps(final, indent=2))
        else:
            typer.echo(final)
    except Exception as e:
        logger.exception("Run failed: %s", e)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def repl(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider name."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name."),
    persist_path: Optional[str] = typer.Option(None, "--persist-path", help="Path to persist memory JSON."),
    enable_persistence: bool = typer.Option(False, "--persist", help="Enable memory persistence to file."),
    max_memory_items: Optional[int] = typer.Option(None, "--max-memory", help="Cap memory to this many items."),
    review_mode: bool = typer.Option(False, "--review/--no-review", help="Enable review mode: changes require approval before applying (like IDE diff review)."),
):
    """
    Start an interactive REPL agent session.
    """
    try:
        loop = _build_agent(provider, model, persist_path, enable_persistence, max_memory_items, verbose=True, review_mode=review_mode)
        loop.run_interactive()
    except Exception as e:
        logger.exception("REPL failed: %s", e)
        typer.echo(f"Error starting REPL: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def probe(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider name to probe. If omitted, probes all supported providers."),
):
    """
    Probe LLM providers and show which are available.
    """
    providers = [provider] if provider else list_supported_providers()
    results = {}
    for p in providers:
        ok, msg = probe_provider_availability(p)
        results[p] = {"available": ok, "msg": msg}
    typer.echo(json.dumps(results, indent=2))

@app.command(name="list-tools")
def list_tools():
    """
    Discover and list available tools from agent.tools package.
    """
    tm = _build_tool_manager()
    metas = tm.list_tools()
    out = []
    for m in metas:
        out.append({"name": m.name, "module": m.module, "class": m.cls_name, "description": m.description})
    typer.echo(json.dumps(out, indent=2))

@app.command()
def memory(
    show: bool = typer.Option(True, "--show/--no-show", help="Show memory contents."),
    clear: bool = typer.Option(False, "--clear", help="Clear memory."),
    persist_path: Optional[str] = typer.Option(None, "--persist-path", help="Path used for memory persistence."),
    enable_persistence: bool = typer.Option(False, "--persist", help="Enable persistence for the memory backend."),
    limit: int = typer.Option(25, "--limit", help="Max records to display when showing memory."),
):
    """
    Inspect or clear the agent memory.
    """
    mem = _build_memory(persist_path=persist_path, enable_persistence=enable_persistence)
    if clear:
        confirm = typer.confirm("Are you sure you want to clear all memory? This action cannot be undone.")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit()
        mem.clear()
        typer.echo("Memory cleared.")
        raise typer.Exit()

    if show:
        items = mem.recall_all()[:limit]
        typer.echo(f"Showing {len(items)} memory items (newest first):\n")
        for it in items:
            typer.echo(json.dumps(it, indent=2))
        raise typer.Exit()

@app.command()
def select(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Provider preference (comma separated) or blank to show defaults."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override to display."),
):
    """
    Show provider/model selection info. If provider list supplied, shows first available provider from the list.
    """
    if provider:
        prefs = [p.strip() for p in provider.split(",") if p.strip()]
        try:
            chosen = None
            # attempt to select first available provider using model_selector.select_provider if available
            from agent.llm.model_selector import select_provider

            chosen = select_provider(prefs)
            typer.echo(f"Selected provider from preferences {prefs} -> {chosen}")
        except Exception as e:
            typer.echo(f"Selection failed: {e}")
            typer.echo("Falling back to probing each provider in order:")
            for p in prefs:
                ok, msg = probe_provider_availability(p)
                typer.echo(f"  - {p}: available={ok}, msg={msg}")
    else:
        # show supported providers and defaults
        providers = list_supported_providers()
        typer.echo("Supported providers:\n")
        for p in providers:
            ok, msg = probe_provider_availability(p)
            typer.echo(f"- {p}: available={ok}   ({msg})")
        # show default model env / config hints
        default_model = os.environ.get("DEFAULT_MODEL") or os.environ.get("MODEL_OPENAI") or "<not-set>"
        typer.echo(f"\nDEFAULT_MODEL env: {default_model}")

# ------------------------
# Entrypoint
# ------------------------
def main():
    try:
        app()
    except Exception:
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
