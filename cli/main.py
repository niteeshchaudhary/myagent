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
from agent.utils.config_loader import get_config
from agent.llm.rag_query import RAGQuery
from agent.rag.config import RagConfig
from agent.mcp.manager import MCPManager

app = typer.Typer(help="Coding Agent CLI")

# Shared factory functions
def _build_tool_manager() -> ToolManager:
    # Load tool configs from unified config
    config = get_config()
    tool_configs = config.tools if config else {}
    
    # Build MCP manager if enabled
    mcp_manager = None
    if config and config.mcp and config.mcp.enabled:
        try:
            server_configs = config.mcp.servers
            if server_configs:
                mcp_manager = MCPManager(server_configs=server_configs)
                mcp_manager.connect_all()
                logger.info("MCP manager initialized with %d servers", len(server_configs))
        except Exception as e:
            logger.warning("Failed to initialize MCP manager: %s", e)
    
    tm = ToolManager(tool_configs=tool_configs, mcp_manager=mcp_manager)
    # Load tool configurations
    if tool_configs:
        tm.load_tool_configs(tool_configs)
    
    # attempt to load tools from the conventional package; ignore failures
    try:
        tm.load_tools_from_package("agent.tools")
    except Exception as e:
        logger.debug("Tool discovery failed: %s", e)
    
    # Load MCP tools
    if mcp_manager:
        try:
            tm.load_mcp_tools()
        except Exception as e:
            logger.warning("Failed to load MCP tools: %s", e)
    
    return tm

def _build_memory(persist_path: Optional[str] = None, enable_persistence: bool = False, max_items: Optional[int] = None) -> Memory:
    # Default to enabling persistence if path is provided or use default path
    if persist_path is None and not enable_persistence:
        # Check config for default persistence settings
        config = get_config()
        if config and hasattr(config, 'agent') and getattr(config.agent, 'enable_memory', True):
            # Use default memory path if persistence should be enabled
            persist_path = ".agent_memory.json"
            enable_persistence = True
    
    cfg = MemoryConfig(
        persist_path=persist_path,
        enable_persistence=enable_persistence,
        max_items=max_items,
    )
    return Memory(cfg)

def _build_planner(llm=None, model: Optional[str] = None, prefer_structured: bool = True) -> Planner:
    # Load config for planner settings
    config = get_config()
    cfg = PlannerConfig(model=model, prefer_structured=prefer_structured)
    
    # Set repo_root from config if available, otherwise use current working directory
    if config and hasattr(config, 'paths'):
        cfg.repo_root = getattr(config.paths, 'workspace', None) or os.getcwd()
    else:
        cfg.repo_root = os.getcwd()
    logger.info("Building planner with repo_root: %s", cfg.repo_root)
    return Planner(llm=llm, config=cfg)

def _build_llm(provider: Optional[str] = None, model: Optional[str] = None):
    try:
        # model_selector.get_llm accepts model kw
        return get_llm(provider, model=model)
    except Exception as e:
        logger.warning("Could not create LLM for provider=%s model=%s: %s", provider, model, e)
        raise

def _ensure_codebase_indexed(repo_root: Optional[str] = None, auto_index: Optional[bool] = None) -> bool:
    """
    Check if codebase is indexed and auto-index if needed.
    
    Args:
        repo_root: Repository root directory (default: current working directory)
        auto_index: If True, automatically index if index doesn't exist. If None, check config.
    
    Returns:
        True if index exists or was created, False otherwise
    """
    try:
        from agent.rag.config import RagConfig
        from agent.rag.indexer import Indexer
        import os
        
        # Check config for auto_index setting if not explicitly provided
        if auto_index is None:
            config = get_config()
            if config and hasattr(config, 'agent'):
                auto_index = getattr(config.agent, 'auto_index', True)
            else:
                auto_index = True  # Default to True
        
        repo_root = repo_root or os.getcwd()
        cfg = RagConfig.from_env(repo_root=repo_root)
        
        # Check if RAG is enabled
        if not cfg or not getattr(cfg, "enabled", True):
            logger.debug("RAG is disabled, skipping auto-indexing")
            return False
        
        # Check if index exists
        index_path = cfg.vectorstore.index_path
        index_meta_path = os.path.join(index_path, "rag_config.json")
        vector_index_path = os.path.join(index_path, "vectors.index")
        meta_json_path = os.path.join(index_path, "meta.json")
        
        index_exists = (
            os.path.exists(index_meta_path) or 
            os.path.exists(vector_index_path) or 
            os.path.exists(meta_json_path)
        )
        
        if not index_exists and auto_index:
            typer.echo(f"\n[INFO] Codebase index not found. Indexing codebase at: {repo_root}")
            
            # Check embedding backend and warn if using Groq
            embed_backend = getattr(cfg.embedding, 'prefer_openai', False)
            groq_model = getattr(cfg.embedding, 'groq_model', None)
            if groq_model:
                typer.echo("⚠️  WARNING: Using Groq LLM for embeddings is VERY SLOW (can take 10-20+ minutes).")
                typer.echo("   Consider switching to local embeddings for faster indexing:")
                typer.echo("   Set embed_backend: 'local' in configs/rag.yaml\n")
            
            typer.echo("This may take a few minutes on first run...\n")
            try:
                indexer = Indexer(cfg)
                result = indexer.index_repo(force=False)
                chunks_indexed = result.get("chunks_indexed", 0)
                elapsed = result.get("elapsed_s", 0)
                typer.echo(f"\n✓ Indexing completed: {chunks_indexed} chunks indexed in {elapsed:.2f}s")
                typer.echo(f"Index location: {index_path}\n")
                return True
            except Exception as e:
                logger.exception("Auto-indexing failed: %s", e)
                typer.echo(f"⚠️  Warning: Auto-indexing failed: {e}", err=True)
                typer.echo("You can manually index later with: agent index\n", err=True)
                return False
        elif not index_exists:
            logger.warning(
                f"Codebase index not found at {index_path}. "
                f"To index your codebase, run: agent index (from {repo_root})"
            )
            return False
        else:
            logger.debug(f"Codebase index found at {index_path}")
            return True
    except Exception as e:
        logger.debug("Could not check codebase index status: %s", e)
        return False

def _build_agent(llm_provider: Optional[str], llm_model: Optional[str], persist_path: Optional[str], enable_persistence: bool, max_memory_items: Optional[int], verbose: bool = True, review_mode: bool = False) -> AgentLoop:
    # Load unified config
    config = get_config()
    agent_config = config.agent if config else None
    
    # Override with CLI args if provided, otherwise use config
    if agent_config:
        verbose = verbose if verbose is not None else agent_config.verbose
        review_mode = review_mode if review_mode is not None else agent_config.review_mode
    
    tm = _build_tool_manager()
    mem = _build_memory(persist_path=persist_path, enable_persistence=enable_persistence, max_items=max_memory_items)
    try:
        llm = _build_llm(llm_provider, llm_model)
    except Exception:
        llm = None
    planner = _build_planner(llm=llm, model=llm_model)
    
    # Build AgentLoopConfig from config.yaml
    if agent_config:
        cfg = AgentLoopConfig(
            max_steps=agent_config.max_steps,
            verbose=verbose,
            ci_auto_apply=agent_config.ci_auto_apply,
            ci_paths=agent_config.ci_paths,
            review_mode=review_mode,
            max_error_retries=agent_config.max_error_retries,
            enable_error_recovery=agent_config.enable_error_recovery
        )
    else:
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
    auto_index: Optional[bool] = typer.Option(None, "--auto-index/--no-auto-index", help="Automatically index codebase if index doesn't exist (overrides config)."),
):
    """
    Run a single prompt through the agent loop and print the final result.
    """
    try:
        # Auto-index codebase if needed
        _ensure_codebase_indexed(auto_index=auto_index)
        
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
    auto_index: Optional[bool] = typer.Option(None, "--auto-index/--no-auto-index", help="Automatically index codebase if index doesn't exist (overrides config)."),
):
    """
    Start an interactive REPL agent session.
    
    The agent will automatically index your codebase on first run if no index exists.
    This ensures the agent understands your codebase structure before you start asking questions.
    """
    try:
        # Auto-index codebase if needed (before starting REPL)
        _ensure_codebase_indexed(auto_index=auto_index)
        
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
def list_tools(
    include_mcp: bool = typer.Option(True, "--mcp/--no-mcp", help="Include MCP tools in the list."),
):
    """
    Discover and list available tools from agent.tools package and MCP servers.
    """
    tm = _build_tool_manager()
    metas = tm.list_tools()
    out = []
    for m in metas:
        # Filter MCP tools if requested
        if not include_mcp and m.module == "agent.mcp":
            continue
        out.append({
            "name": m.name,
            "module": m.module,
            "class": m.cls_name,
            "description": m.description,
            "type": "mcp" if m.module == "agent.mcp" else "native"
        })
    typer.echo(json.dumps(out, indent=2))

@app.command(name="list-mcp")
def list_mcp():
    """
    List connected MCP servers and their available tools.
    """
    config = get_config()
    if not config or not config.mcp or not config.mcp.enabled:
        typer.echo("MCP is disabled in config.yaml")
        raise typer.Exit()
    
    try:
        mcp_manager = MCPManager(server_configs=config.mcp.servers)
        mcp_manager.connect_all()
        
        servers = mcp_manager.list_servers()
        tools = mcp_manager.list_tools()
        
        result = {
            "servers": servers,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "server": tool.client.config.name
                }
                for tool in tools
            ]
        }
        
        typer.echo(json.dumps(result, indent=2))
        
        mcp_manager.disconnect_all()
    except Exception as e:
        logger.exception("Failed to list MCP servers: %s", e)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

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
def ask(
    question: str = typer.Argument(..., help="Question to ask about the codebase."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider name (e.g. 'openai', 'ollama', 'groq')."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name to use for the LLM."),
    k: int = typer.Option(8, "--k", help="Number of code chunks to retrieve."),
    prefer_exact: bool = typer.Option(False, "--exact/--semantic", help="Prefer exact search over semantic search."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Only retrieve chunks, don't call LLM (show retrieved context)."),
    stream: bool = typer.Option(False, "--stream", help="Stream LLM output."),
):
    """
    Ask a question about the codebase using RAG (Retrieval-Augmented Generation).
    
    This command uses RAG to retrieve relevant code snippets and answer questions
    directly without running the full agent loop. It's faster for simple questions
    that just need codebase knowledge.
    
    Examples:
        agent ask "How does authentication work?"
        agent ask "Where is the main entry point?" --exact
        agent ask "What tools are available?" --k 5
        agent ask "Show me the config loader" --no-llm
    """
    try:
        # Get current working directory - this is the codebase we want to search
        cwd = os.getcwd()
        logger.info("Using current working directory as repo_root: %s", cwd)
        
        # Load config
        config = get_config()
        
        # Build RAG config from rag.yaml, using current working directory as repo_root
        rag_cfg = RagConfig.from_env(repo_root=cwd)
        
        # Check if RAG is enabled
        if config and config.rag:
            enabled = config.rag.get("enabled", True)
            if not enabled:
                typer.echo("RAG is disabled in configs/rag.yaml. Enable it to use the ask command.", err=True)
                raise typer.Exit(code=1)
        
        # Build LLM if needed
        llm = None
        if not no_llm:
            try:
                llm = _build_llm(provider, model)
            except Exception as e:
                logger.warning("Could not create LLM: %s", e)
                typer.echo(f"Warning: Could not create LLM: {e}", err=True)
                typer.echo("Use --no-llm to see retrieved chunks without LLM answer.", err=True)
                raise typer.Exit(code=1)
        
        # Create RAG query instance
        rag_query = RAGQuery(cfg=rag_cfg, llm=llm)
        
        # Run query
        if stream and llm and hasattr(llm, "stream"):
            typer.echo("Retrieving relevant code...")
            result = rag_query.answer(
                question,
                k=k,
                prefer_exact=prefer_exact,
                use_llm=True,
                model=model,
                stream=True
            )
            typer.echo("\n=== ANSWER ===\n")
            typer.echo(result.get("answer", ""))
        else:
            typer.echo("Retrieving relevant code and generating answer...")
            result = rag_query.answer(
                question,
                k=k,
                prefer_exact=prefer_exact,
                use_llm=not no_llm,
                model=model,
                stream=False
            )
            
            if no_llm:
                # Show retrieved chunks
                chunks = result.get("chunks", [])
                typer.echo(f"\n=== RETRIEVED {len(chunks)} CHUNKS ===\n")
                for i, chunk in enumerate(chunks, 1):
                    path = chunk.get("path", "unknown")
                    start_line = chunk.get("start_line")
                    end_line = chunk.get("end_line")
                    source_type = chunk.get("source_type", "unknown")
                    score = chunk.get("score", 0.0)
                    text = chunk.get("text", "")[:500]  # First 500 chars
                    
                    location = path
                    if start_line:
                        location += f":{start_line}"
                        if end_line and end_line != start_line:
                            location += f"-{end_line}"
                    
                    typer.echo(f"[{i}] {location} ({source_type}, score={score:.3f})")
                    typer.echo(f"```\n{text}\n```\n")
            else:
                # Show answer
                typer.echo("\n=== ANSWER ===\n")
                typer.echo(result.get("answer", ""))
                
                # Show sources (deduplicated and filtered)
                sources = result.get("sources", [])
                if sources:
                    typer.echo("\n=== SOURCES ===\n")
                    if isinstance(sources, list):
                        # Already a list of strings
                        for source in sources:
                            typer.echo(f"  {source}")
                    elif isinstance(sources, str):
                        # Legacy: string format, split by newlines
                        for line in sources.strip().split("\n"):
                            if line.strip():
                                typer.echo(f"  {line.strip()}")
                    else:
                        typer.echo(f"  {sources}")
                
                elapsed = result.get("elapsed_s", 0)
                if elapsed:
                    typer.echo(f"\n(Retrieved and answered in {elapsed:.2f}s)")
        
    except Exception as e:
        logger.exception("Ask command failed: %s", e)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

@app.command()
def index(
    force: bool = typer.Option(False, "--force", help="Force reindexing even if index exists."),
    changed_only: bool = typer.Option(False, "--changed", help="Index only changed files (requires git)."),
    repo: Optional[str] = typer.Option(None, "--repo", help="Repository root directory (default: current directory)."),
):
    """
    Index the codebase for RAG (Retrieval-Augmented Generation).
    
    This creates a searchable index of your codebase that helps the agent understand
    the code structure and find relevant code snippets when answering questions or
    planning actions.
    
    Examples:
        agent index                    # Index entire codebase
        agent index --force            # Force reindex (rebuild from scratch)
        agent index --changed          # Index only changed files (git-based)
    """
    try:
        from agent.rag.indexer import Indexer
        from agent.rag.config import RagConfig
        
        # Use provided repo or current working directory
        repo_root = repo or os.getcwd()
        logger.info("Indexing repository at: %s", repo_root)
        cfg = RagConfig.from_env(repo_root=repo_root)
        
        # Check if RAG is enabled
        if not cfg or not getattr(cfg, "enabled", True):
            typer.echo("RAG is disabled in configs/rag.yaml. Enable it to use indexing.", err=True)
            raise typer.Exit(code=1)
        
        typer.echo(f"Initializing indexer for repository: {repo_root}")
        indexer = Indexer(cfg)
        
        if changed_only:
            typer.echo("Indexing changed files (git-based)...")
            result = indexer.index_changed_files()
            if result.get("changed", 0) == 0:
                typer.echo("No changed files to index.")
            else:
                typer.echo(f"✓ Indexed {result.get('chunks_indexed', 0)} chunks from changed files")
        else:
            typer.echo("Indexing entire codebase (this may take a few minutes)...")
            result = indexer.index_repo(force=force)
            typer.echo(f"✓ Indexing completed: {result.get('chunks_indexed', 0)} chunks indexed in {result.get('elapsed_s', 0):.2f}s")
        
        typer.echo(f"\nIndex location: {cfg.vectorstore.index_path}")
        typer.echo("The agent can now better understand your codebase structure.")
        
    except Exception as e:
        logger.exception("Indexing failed: %s", e)
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

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
