# agent/core/agent_loop.py
"""
Agent loop core with CI integration.

Enhancements over the earlier version:
- After any tool step that *may* modify repository files (heuristic), the loop will:
    1. Run the "ci" tool in dry-run mode (attempt_fix=False). This summarizes linter/tests.
    2. Store CI results in memory.
    3. If CI reports failures and AgentLoopConfig.ci_auto_apply is True AND an LLM adapter
       is available, call the "ci" tool again with attempt_fix=True and llm=self.llm to
       attempt to apply an LLM-generated patch (the CI tool itself handles safety branches
       and rollback according to its configuration).
- All CI runs are best-effort and do not raise if they fail — results are logged and recorded.

Note: The CI tool must be discoverable by ToolManager (name 'ci') and follow the
interface described in `agent/tools/ci_tool.py`.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from agent.utils.file_ref import expand_file_refs
from agent.utils.change_tracker import get_change_tracker
from typing import Callable

# Try to reuse project logger if available
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
class AgentLoopConfig:
    """
    High-level runtime config for the AgentLoop.

    New CI-related fields:
      - ci_auto_apply: if True, and LLM available, attempt to auto-apply fixes suggested by the CI tool
                      after a failed dry-run. Default: False (safe).
      - ci_paths: paths passed to the CI tool (default: ".")
      - max_steps: maximum number of execution steps in a single run (safety).
      - step_delay: optional pause (seconds) between steps (useful for rate-limited tools or readable REPL).
      - verbose: if True, print streaming outputs as they arrive.
      - review_mode: if True, file changes require user approval before applying (like IDE diff review).
    """
    max_steps: int = 20
    step_delay: float = 0.0
    verbose: bool = True
    ci_auto_apply: bool = False
    ci_paths: str = "."
    ci_max_fix_attempts: int = 2
    review_mode: bool = False
    max_error_retries: int = 5  # Maximum number of times to retry fixing errors
    enable_error_recovery: bool = True  # Enable automatic error recovery loop


class AgentLoop:
    """
    Orchestrates the agent's sense-plan-act loop with optional CI checks.

    Args:
        planner: an object responsible for producing plans from prompts & memory.
        memory: an object responsible for storing & retrieving memory.
        tool_manager: object responsible for discovering and running tools.
        llm: an LLM adapter with generate(prompt, ...) and stream(prompt, ...) methods.
        config: AgentLoopConfig for runtime behaviour.
    """

    def __init__(self, planner: Any, memory: Any, tool_manager: Any, llm: Any,
                 config: Optional[AgentLoopConfig] = None):
        self.planner = planner
        self.memory = memory
        self.tool_manager = tool_manager
        self.llm = llm
        self.config = config or AgentLoopConfig()
        self._running = False
        
        # Set up review mode if configured
        if self.config.review_mode:
            tracker = get_change_tracker()
            tracker.set_review_mode(True)

    # ---------------------------
    # Utilities for adapter interop
    # ---------------------------
    def _call_planner(self, prompt: str) -> Union[str, Dict, List]:
        """
        Call the planner using common method names; return whatever it returns.
        Uses semantic memory recall to find relevant previous actions.
        """
        # Get available tools to pass to planner
        available_tools = []
        if hasattr(self.tool_manager, "list_tools"):
            tools = self.tool_manager.list_tools()
            available_tools = [t.name for t in tools]
        
        # Get memory snapshot with semantic recall based on current prompt
        memory_snapshot = self._safe_memory_snapshot(prompt=prompt)
        # Add available tools to memory so planner can use them
        if isinstance(memory_snapshot, dict):
            memory_snapshot["available_tools"] = available_tools
        
        # Check if codebase needs indexing (for RAG)
        self._ensure_codebase_indexed()
        
        for name in ("plan", "create_plan", "generate_plan", "make_plan"):
            if hasattr(self.planner, name):
                try:
                    fn = getattr(self.planner, name)
                    try:
                        return fn(prompt, memory_snapshot)
                    except TypeError:
                        return fn(prompt)
                except Exception as e:
                    logger.exception("Planner method %s raised an exception: %s", name, e)
                    raise
        raise RuntimeError("Planner does not expose a recognized entrypoint (plan/create_plan/...).")

    def _safe_memory_snapshot(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Attempt to produce a lightweight snapshot / summary of memory to pass into planner.
        Uses semantic recall to find relevant previous actions based on the current prompt.
        Falls back to recent items if no prompt is provided.
        """
        try:
            snapshot = {}
            conversation_history = []
            
            # Get all memory items for fallback
            if hasattr(self.memory, "recall_all"):
                all_items = self.memory.recall_all()
                snapshot["total_memory_items"] = len(all_items)
            
            # Use semantic recall if prompt is provided and memory supports it
            if prompt and hasattr(self.memory, "recall"):
                try:
                    # Recall relevant items based on the current prompt
                    # Use keyword mode for better matching of actions/tools
                    relevant_items = self.memory.recall(prompt, k=15, mode="keyword")
                    logger.debug("Recalled %d relevant memory items for prompt", len(relevant_items))
                    
                    # Also get recent items to ensure we have context
                    recent_items = all_items[:10] if hasattr(self.memory, "recall_all") else []
                    
                    # Combine relevant and recent, deduplicate by id
                    seen_ids = set()
                    combined_items = []
                    for item in relevant_items + recent_items:
                        item_id = item.get("id")
                        if item_id and item_id not in seen_ids:
                            seen_ids.add(item_id)
                            combined_items.append(item)
                        elif not item_id:
                            # If no id, check by content hash
                            content_str = str(item.get("content", ""))
                            if content_str not in seen_ids:
                                seen_ids.add(content_str)
                                combined_items.append(item)
                    
                    # Sort by timestamp (most recent first)
                    combined_items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                    
                    # Extract conversation history from combined items
                    for item in combined_items[:25]:  # Limit to top 25 most relevant/recent
                        content = item.get("content", {})
                        if isinstance(content, dict):
                            # Look for prompts and responses
                            if "prompt" in content or "response" in content:
                                conversation_history.append({
                                    "prompt": content.get("prompt", ""),
                                    "response": content.get("response", ""),
                                    "timestamp": item.get("timestamp", content.get("timestamp", 0))
                                })
                            # Also include tool executions that might be relevant
                            elif "tool" in content:
                                conversation_history.append({
                                    "tool": content.get("tool"),
                                    "input": str(content.get("input", ""))[:200],  # Truncate long inputs
                                    "output": str(content.get("output", ""))[:200] if isinstance(content.get("output"), dict) else str(content.get("output", ""))[:200],
                                    "timestamp": item.get("timestamp", content.get("timestamp", 0)),
                                    "success": content.get("output", {}).get("success", True) if isinstance(content.get("output"), dict) else True
                                })
                            # Include todo execution results
                            elif "todo_index" in content:
                                conversation_history.append({
                                    "todo_index": content.get("todo_index"),
                                    "step": content.get("step", {}),
                                    "result": str(content.get("result", ""))[:200],
                                    "timestamp": item.get("timestamp", content.get("timestamp", 0))
                                })
                except Exception as e:
                    logger.debug("Semantic recall failed, falling back to recent items: %s", e)
                    # Fallback to recent items
                    if hasattr(self.memory, "recall_all"):
                        all_items = self.memory.recall_all()
                        for item in all_items[:20]:
                            content = item.get("content", {})
                            if isinstance(content, dict):
                                if "prompt" in content or "response" in content:
                                    conversation_history.append({
                                        "prompt": content.get("prompt", ""),
                                        "response": content.get("response", ""),
                                        "timestamp": item.get("timestamp", content.get("timestamp", 0))
                                    })
                                elif "tool" in content:
                                    conversation_history.append({
                                        "tool": content.get("tool"),
                                        "input": str(content.get("input", ""))[:200],
                                        "output": str(content.get("output", ""))[:200] if isinstance(content.get("output"), dict) else str(content.get("output", ""))[:200],
                                        "timestamp": item.get("timestamp", content.get("timestamp", 0))
                                    })
            else:
                # No prompt provided, use recent items
                if hasattr(self.memory, "recall_all"):
                    all_items = self.memory.recall_all()
                    for item in all_items[:20]:
                        content = item.get("content", {})
                        if isinstance(content, dict):
                            if "prompt" in content or "response" in content:
                                conversation_history.append({
                                    "prompt": content.get("prompt", ""),
                                    "response": content.get("response", ""),
                                    "timestamp": item.get("timestamp", content.get("timestamp", 0))
                                })
                            elif "tool" in content:
                                conversation_history.append({
                                    "tool": content.get("tool"),
                                    "input": str(content.get("input", ""))[:200],
                                    "output": str(content.get("output", ""))[:200] if isinstance(content.get("output"), dict) else str(content.get("output", ""))[:200],
                                    "timestamp": item.get("timestamp", content.get("timestamp", 0))
                                })
            
            snapshot["conversation_history"] = conversation_history
            
            # Also include full snapshot if available (for other context)
            if hasattr(self.memory, "snapshot"):
                full_snapshot = self.memory.snapshot()
                # Merge but prioritize conversation_history
                for key, value in full_snapshot.items():
                    if key not in snapshot:
                        snapshot[key] = value
            elif hasattr(self.memory, "to_dict"):
                full_dict = self.memory.to_dict()
                for key, value in full_dict.items():
                    if key not in snapshot:
                        snapshot[key] = value
            
            return snapshot
        except Exception as e:
            logger.debug("Memory snapshot failed: %s", e)
            return {}

    def _store_to_memory(self, item: Any, summary: Optional[str] = None) -> None:
        """
        Store an item into memory using common method names.
        Automatically generates a summary if not provided to help with semantic recall.
        """
        try:
            # Generate a summary if not provided to help with semantic recall
            if summary is None and isinstance(item, dict):
                summary = self._generate_memory_summary(item)
            
            if hasattr(self.memory, "store"):
                self.memory.store(item, summary=summary)
                return
            if hasattr(self.memory, "save"):
                self.memory.save(item)
                return
            if hasattr(self.memory, "append"):
                self.memory.append(item, summary=summary)
                return
            logger.debug("Memory does not expose store/save/append; skipping memory write.")
        except Exception as e:
            logger.exception("Failed to write to memory: %s", e)
    
    def _generate_memory_summary(self, item: Dict[str, Any]) -> str:
        """
        Generate a searchable summary from a memory item to help with semantic recall.
        """
        parts = []
        
        # Extract key information for searchability
        if "prompt" in item:
            parts.append(f"User asked: {str(item['prompt'])[:100]}")
        if "tool" in item:
            tool_name = item.get("tool", "unknown")
            tool_input = str(item.get("input", ""))[:100]
            parts.append(f"Executed tool '{tool_name}': {tool_input}")
        if "response" in item:
            response = str(item.get("response", ""))[:100]
            parts.append(f"Response: {response}")
        if "todo_index" in item:
            step = item.get("step", {})
            step_desc = str(step.get("input", step.get("prompt", "")))[:100]
            parts.append(f"Todo step {item.get('todo_index')}: {step_desc}")
        if "ci_dry_run" in item or "ci_apply" in item:
            parts.append("CI check performed")
        
        # Add success/failure status if available
        if isinstance(item.get("output"), dict):
            success = item.get("output", {}).get("success", True)
            if not success:
                parts.append("(failed)")
        
        return " | ".join(parts) if parts else str(item)[:200]
    
    def _ensure_codebase_indexed(self) -> None:
        """
        Check if the codebase is indexed for RAG. If not, log a warning.
        This helps ensure the agent can understand the codebase structure.
        """
        try:
            from agent.rag.config import RagConfig
            from agent.rag.indexer import Indexer
            import os
            
            # Load RAG config
            repo_root = getattr(self.planner, "repo_root", ".") if hasattr(self.planner, "repo_root") else "."
            cfg = RagConfig.from_env(repo_root=repo_root)
            
            # Check if RAG is enabled
            if not cfg or not getattr(cfg, "enabled", True):
                logger.debug("RAG is disabled, skipping index check")
                return
            
            # Check if index exists
            index_path = getattr(cfg.vectorstore, "index_path", ".agent_index")
            index_meta_path = os.path.join(index_path, "rag_config.json")
            vector_index_path = os.path.join(index_path, "vectors.index")
            meta_json_path = os.path.join(index_path, "meta.json")
            
            index_exists = (
                os.path.exists(index_meta_path) or 
                os.path.exists(vector_index_path) or 
                os.path.exists(meta_json_path)
            )
            
            if not index_exists:
                logger.warning(
                    "Codebase is not indexed for RAG. The agent may not fully understand the codebase structure. "
                    "To index the codebase, run: python -m agent.rag.indexer index"
                )
                # Optionally auto-index (commented out for safety - user should explicitly index)
                # logger.info("Attempting to auto-index codebase...")
                # try:
                #     indexer = Indexer(cfg)
                #     result = indexer.index_repo()
                #     logger.info("Auto-indexing completed: %s", result)
                # except Exception as e:
                #     logger.warning("Auto-indexing failed: %s", e)
            else:
                logger.debug("Codebase index found at %s", index_path)
        except Exception as e:
            logger.debug("Could not check codebase index status: %s", e)

    def _run_tool(self, tool_name: str, tool_input: Any, timeout: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool using ToolManager. Try several common method names and return a dict with results.
        The returned dict should include at least 'output' key (string) if possible.
        Additional kwargs (like previous_output) are passed to the tool.
        """
        logger.info("Executing tool '%s' with input: %s", tool_name, str(tool_input)[:200])
        try:
            if hasattr(self.tool_manager, "run"):
                res = self.tool_manager.run(tool_name, input=tool_input, timeout=timeout, **kwargs)
                return self._normalize_tool_result(res)
            if hasattr(self.tool_manager, "execute"):
                res = self.tool_manager.execute(tool_name, tool_input, timeout=timeout)
                return self._normalize_tool_result(res)
            if hasattr(self.tool_manager, "get_tool"):
                tool = self.tool_manager.get_tool(tool_name)
                if hasattr(tool, "run"):
                    return self._normalize_tool_result(tool.run(tool_input, **kwargs))
                if hasattr(tool, "execute"):
                    return self._normalize_tool_result(tool.execute(tool_input))
            if hasattr(self.tool_manager, "tools") and tool_name in getattr(self.tool_manager, "tools"):
                tool = self.tool_manager.tools[tool_name]
                if hasattr(tool, "run"):
                    return self._normalize_tool_result(tool.run(tool_input, **kwargs))
            raise RuntimeError(f"Tool manager couldn't run tool '{tool_name}' with known interfaces.")
        except Exception as e:
            logger.exception("Tool '%s' execution failed: %s", tool_name, e)
            return {"output": "", "success": False, "error": str(e)}

    @staticmethod
    def _normalize_tool_result(res: Any) -> Dict[str, Any]:
        """
        Normalize different tool return shapes into a dict with 'output' key and optional metadata.
        """
        if res is None:
            return {"output": "", "success": False}
        if isinstance(res, dict):
            if "output" in res:
                return res
            for key in ("stdout", "result", "text"):
                if key in res:
                    return {"output": res[key], **{k: v for k, v in res.items() if k != key}}
            return {"output": str(res), **res}
        return {"output": str(res), "success": True}

    # ---------------------------
    # CI integration helpers
    # ---------------------------
    def _tool_may_modify_files(self, tool_name: str, tool_input: Any) -> bool:
        """
        Heuristic to determine whether a tool invocation is likely to modify repository files.
        Looks at tool name and content for common patterns.
        """
        if not tool_name:
            return False
        tn = tool_name.lower()
        modifying_tool_keywords = ("git", "apply", "patch", "write", "file", "python", "install", "mv", "rm", "sed", "ed")
        # direct matches
        if any(k in tn for k in ("git", "apply", "patch", "file", "python", "installer")):
            return True
        # if input contains '*** Begin Patch' or 'diff --git' or 'patch' or 'apply_patch' it's likely modifying
        try:
            s = str(tool_input or "")
            if any(marker in s for marker in ("*** Begin Patch", "diff --git", "git apply", "patch -p", "apply_patch")):
                return True
            # if the input contains many newline-separated file headers or 'open(' write patterns
            if "def " in s and ("write(" in s or "open(" in s or "apply_patch" in s):
                return True
            # if tool name includes shell or os, be cautious
            if any(k in tn for k in ("shell", "os", "bash")) and any(tok in s for tok in ("rm ", "mv ", "cp ", "sed ", "truncate ")):
                return True
        except Exception:
            pass
        # default: not modifying
        return False

    def _run_ci_dry_run(self) -> Dict[str, Any]:
        """
        Invoke the CI tool in dry-run mode (attempt_fix=False).
        Returns the normalized CI result dict (or a failure description).
        IMPORTANT: auto_rollback is set to False for dry-run to prevent rolling back files.
        """
        if not hasattr(self.tool_manager, "run"):
            logger.debug("Tool manager does not support run(); skipping CI dry-run.")
            return {"output": "no-tool-manager-run", "success": False}
        try:
            ci_input = {
                "paths": self.config.ci_paths, 
                "attempt_fix": False,
                "auto_rollback": False  # Don't rollback on dry-run - we just want to check status
            }
            ci_res = self.tool_manager.run("ci", input=ci_input)
            ci_res_norm = self._normalize_tool_result(ci_res)
            return ci_res_norm
        except Exception as e:
            logger.exception("CI dry-run failed: %s", e)
            return {"output": "", "success": False, "error": str(e)}

    def _attempt_ci_auto_apply(self) -> Dict[str, Any]:
        """
        If configured, attempt to run the CI tool with attempt_fix=True and pass the LLM.
        Returns CI result dict (normalized).
        """
        if not self.config.ci_auto_apply:
            return {"output": "ci_auto_apply disabled", "success": False}
        if not self.llm:
            logger.info("ci_auto_apply requested but no LLM available; skipping auto-apply.")
            return {"output": "no-llm", "success": False}
        try:
            ci_input = {
                "paths": self.config.ci_paths,
                "attempt_fix": True,
                "llm": self.llm,
                "max_fix_attempts": self.config.ci_max_fix_attempts,
            }
            ci_res = self.tool_manager.run("ci", input=ci_input)
            ci_res_norm = self._normalize_tool_result(ci_res)
            return ci_res_norm
        except Exception as e:
            logger.exception("CI auto-apply failed: %s", e)
            return {"output": "", "success": False, "error": str(e)}

    # ---------------------------
    # Main loop logic
    # ---------------------------
    def run_once(self, prompt: str) -> Dict[str, Any]:
        """
        Run a single planning + execution cycle and return a final result dict.

        Flow (extended with CI and error recovery):
        1. Ask the planner for a plan
        2. For each step:
            - execute tool steps
            - after any step that likely modifies files, run CI dry-run, store results
            - if CI reports failures and ci_auto_apply=True, call CI with attempt_fix via LLM
            - for LLM steps, stream or run single-shot according to config
        3. If any tool execution fails and error recovery is enabled, create a recovery prompt
           and loop back to step 1 (up to max_error_retries times)
        4. Return final result (last tool/llm output or planner finish)
        """
        original_prompt = prompt
        error_retry_count = 0
        last_error_context = None
        
        while error_retry_count <= self.config.max_error_retries:
            if error_retry_count > 0:
                logger.info("Error recovery attempt %d/%d", error_retry_count, self.config.max_error_retries)
                # Build recovery prompt with error context
                recovery_prompt = self._build_error_recovery_prompt(original_prompt, last_error_context)
                prompt = recovery_prompt
            else:
                logger.info("Starting run_once with prompt: %s", prompt if len(prompt) < 200 else prompt[:200] + "...")
            
            result = self._execute_plan(prompt, original_prompt)
            
            # Check if execution failed and error recovery is enabled
            if self.config.enable_error_recovery and not result.get("success", True):
                error_info = result.get("error_info")
                if error_info and error_retry_count < self.config.max_error_retries:
                    last_error_context = error_info
                    error_retry_count += 1
                    logger.warning("Execution failed. Attempting error recovery (attempt %d/%d)...", 
                                  error_retry_count, self.config.max_error_retries)
                    continue
                else:
                    logger.warning("Error recovery disabled or max retries reached. Returning error result.")
                    return result
            else:
                # Success or error recovery disabled - return result
                return result
        
        # If we exhausted retries, return the last result
        logger.error("Exhausted error recovery retries (%d attempts)", error_retry_count)
        return result
    
    def _build_error_recovery_prompt(self, original_prompt: str, error_context: Dict[str, Any]) -> str:
        """
        Build a recovery prompt that includes the original prompt and error context.
        """
        error_msg = error_context.get("error", "Unknown error")
        error_output = error_context.get("output", "")
        failed_tool = error_context.get("tool", "unknown")
        failed_input = error_context.get("input", "")
        
        # Combine error message and output for better context
        # For shell commands, the error message often contains the stderr
        full_error_info = error_msg
        if error_output and error_output not in error_msg:
            full_error_info = f"{error_msg}\n\nOutput:\n{error_output}"
        
        recovery_prompt = f"""The previous attempt to complete the task failed. Please analyze the error and create a new plan to fix it.

Original task: {original_prompt}

Error details:
- Tool that failed: {failed_tool}
- Command/input that failed: {failed_input}
- Full error information:
{full_error_info}

Please create a plan to fix this error and complete the original task. Consider:
1. What went wrong with the previous attempt?
2. What steps are needed to fix the error?
3. How can we complete the original task successfully?

Generate a new plan to fix the error and proceed."""
        
        return recovery_prompt
    
    def _execute_plan(self, prompt: str, original_prompt: str) -> Dict[str, Any]:
        """
        Execute a single planning + execution cycle (internal method, called by run_once).
        Returns result dict with success flag and optional error_info for recovery.
        """
        plan = self._call_planner(prompt)

        # Handle string responses from planner - try to extract code blocks (conservatively)
        if isinstance(plan, str):
            logger.warning("Planner returned string result instead of structured steps. Attempting to extract code blocks (conservative mode)...")
            # Only extract if the prompt explicitly mentions creating files
            prompt_lower = prompt.lower()
            has_file_creation_intent = any(keyword in prompt_lower for keyword in [
                "create file", "write file", "generate file", "make file", "add file",
                "create code", "write code", "generate code", "create component", "create app"
            ])
            
            if has_file_creation_intent:
                # Try to extract code blocks from the string response and convert to file_editor steps
                file_steps = self._extract_code_blocks_from_output(plan)
            else:
                logger.info("No explicit file creation intent detected in prompt, skipping code block extraction")
                file_steps = []
            if file_steps:
                logger.info("Extracted %d file_editor steps from planner string response", len(file_steps))
                # Use the extracted file steps as the plan
                steps = file_steps
            else:
                logger.debug("No code blocks found in planner string response; treating as final answer.")
                memory_item = {
                    "prompt": prompt, 
                    "response": plan, 
                    "timestamp": time.time(),
                    "action_type": "planning",
                    "response_type": "plan"
                }
                summary = f"Planned for: {prompt[:100]}"
                self._store_to_memory(memory_item, summary=summary)
                return {"final": plan, "provider": "planner", "raw_plan": plan}
        elif isinstance(plan, dict) and plan:
            steps = [plan]
        elif isinstance(plan, (list, tuple)):
            steps = list(plan)
        else:
            logger.warning("Planner returned unexpected type (%s). Will ask LLM directly.", type(plan))
            llm_out = self._ask_llm(prompt)
            self._store_to_memory({"prompt": prompt, "response": llm_out.get("text"), "timestamp": time.time()})
            return {"final": llm_out.get("text"), "provider": "llm", "raw": llm_out}

        step_count = 0
        last_output = None

        for step in steps:
            if step_count >= self.config.max_steps:
                logger.warning("Reached max_steps (%s); aborting.", self.config.max_steps)
                break
            step_count += 1

            if isinstance(step, str):
                parsed = self._parse_simple_step_string(step)
                step = parsed

            if not isinstance(step, dict):
                logger.debug("Skipping non-dict/non-string step: %s", step)
                continue

            action = (step.get("action") or step.get("type") or "").lower()
            logger.info("Executing step %s: action=%s", step_count, action or "<none>")

            # HANDLE finish
            if action in ("finish", "return", "done", "final"):
                final = step.get("output") or step.get("result") or step.get("text") or step.get("content") or ""
                logger.info("Planner requested finish with: %s", final)
                
                # Check if the last operation actually failed (e.g., due to rollback)
                if isinstance(last_output, dict) and not last_output.get("success", True):
                    error_msg = last_output.get("error", "Operation failed")
                    output_msg = last_output.get("output", "")
                    final = f"{final}\n[ERROR] {error_msg}"
                    if output_msg and output_msg not in final:
                        final = f"{final}\n{output_msg}"
                    logger.warning("Finishing with error status due to failed last operation: %s", error_msg)
                    
                    # If error recovery is enabled, return error info instead of finishing
                    if self.config.enable_error_recovery:
                        error_info = {
                            "error": error_msg,
                            "output": output_msg,
                            "tool": last_output.get("tool"),
                            "input": last_output.get("input"),
                            "last_output": last_output
                        }
                        self._store_to_memory({"prompt": prompt, "response": final, "timestamp": time.time()})
                        return {
                            "final": final, 
                            "provider": "planner", 
                            "raw_plan": plan, 
                            "success": False,
                            "error_info": error_info
                        }
                
                self._store_to_memory({"prompt": prompt, "response": final, "timestamp": time.time()})
                success = isinstance(last_output, dict) and last_output.get("success", True) if last_output else True
                return {"final": final, "provider": "planner", "raw_plan": plan, "success": success}

            # TOOL execution
            if action in ("tool", "execute", "run_tool", "call_tool"):
                tool_name = step.get("tool") or step.get("name") or step.get("tool_name")
                tool_input = step.get("input") or step.get("args") or step.get("content") or ""
                timeout = step.get("timeout")
                if not tool_name:
                    logger.warning("Tool step missing 'tool' or 'name': %s", step)
                    continue

                # Pass last_output to tools so they can access previous step results
                tool_res = self._run_tool(tool_name, tool_input, timeout=timeout, previous_output=last_output)
                last_output = tool_res
                # store tool execution outcome in memory with enhanced context
                memory_item = {
                    "tool": tool_name, 
                    "input": tool_input, 
                    "output": tool_res, 
                    "timestamp": time.time(),
                    "action_type": "tool_execution",
                    "success": tool_res.get("success", True) if isinstance(tool_res, dict) else True
                }
                summary = f"Executed {tool_name}: {str(tool_input)[:100]}"
                self._store_to_memory(memory_item, summary=summary)
                
                # Check if tool execution failed and error recovery is enabled
                if self.config.enable_error_recovery and isinstance(tool_res, dict) and not tool_res.get("success", True):
                    error_msg = tool_res.get("error", "Tool execution failed")
                    error_output = tool_res.get("output", "")
                    logger.warning("Tool '%s' execution failed: %s", tool_name, error_msg)
                    
                    # Store error info for potential recovery
                    error_info = {
                        "error": error_msg,
                        "output": error_output,
                        "tool": tool_name,
                        "input": tool_input,
                        "last_output": tool_res
                    }
                    
                    # Don't immediately return - continue to see if planner handles it
                    # But mark that we have an error for later recovery if needed
                    last_output["error_info"] = error_info
                
                # Display file changes if this tool modified files
                if self._tool_may_modify_files(tool_name, tool_input):
                    try:
                        tracker = get_change_tracker()
                        if tracker.has_pending_changes():
                            # In review mode, show pending changes and ask for approval
                            if self.config.review_mode:
                                tracker.review_changes()
                            elif tracker.has_changes():
                                tracker.print_changes()
                        elif tracker.has_changes():
                            tracker.print_changes()
                    except Exception as e:
                        logger.debug("Failed to display changes: %s", e)

                # If the tool likely modified files, run CI dry-run
                try:
                    if self._tool_may_modify_files(tool_name, tool_input):
                        logger.info("Detected possible file-modifying tool '%s'. Running CI dry-run...", tool_name)
                        ci_dry = self._run_ci_dry_run()
                        logger.info("CI dry-run result: success=%s", ci_dry.get("success"))
                        # store ci result to memory for audit
                        self._store_to_memory({"ci_dry_run": ci_dry, "trigger_tool": tool_name, "timestamp": time.time()})

                        # Check if CI failed and if rollback happened
                        ci_failed = not bool(ci_dry.get("success"))
                        ci_output = ci_dry.get("output", "")
                        rollback_happened = "rollback" in ci_output.lower() or "Rollback succeeded" in ci_output
                        
                        # If rollback happened, the tool operation was effectively undone - mark as failed
                        if rollback_happened:
                            logger.warning("CI validation failed and rollback occurred. Tool operation '%s' was undone.", tool_name)
                            tool_res["success"] = False
                            tool_res["error"] = "Operation was rolled back due to CI validation failures"
                            tool_res["output"] = tool_res.get("output", "") + f"\n[WARNING] Changes were rolled back: {ci_output}"
                            last_output = tool_res

                        # If CI failed and auto apply is enabled, attempt fix with LLM
                        if ci_failed and self.config.ci_auto_apply and not rollback_happened:
                            logger.info("CI reported failures and ci_auto_apply is enabled. Attempting auto-apply via CI tool.")
                            ci_apply = self._attempt_ci_auto_apply()
                            logger.info("CI auto-apply result: success=%s", ci_apply.get("success"))
                            self._store_to_memory({"ci_apply": ci_apply, "trigger_tool": tool_name, "timestamp": time.time()})
                            # update last_output to include ci info
                            if not rollback_happened:
                                last_output = {"output": tool_res.get("output", ""), "ci": {"dry_run": ci_dry, "apply": ci_apply}}
                        elif ci_failed:
                            # CI failed but rollback happened - include CI info in output
                            last_output = {"output": tool_res.get("output", ""), "ci": {"dry_run": ci_dry}, "success": False}
                except Exception as e:
                    logger.exception("CI integration after tool '%s' raised: %s", tool_name, e)

                if self.config.step_delay:
                    time.sleep(self.config.step_delay)
                continue

            # LLM query / think step
            if action in ("llm", "think", "query", "ask", "prompt"):
                content = step.get("input") or step.get("prompt") or step.get("content") or ""
                if not content:
                    logger.debug("LLM step missing content; skipping.")
                    continue
                
                # CRITICAL: If content is suspiciously short or doesn't relate to the original prompt,
                # prepend the original prompt to provide context
                if len(content) < 50 or (original_prompt and not any(word.lower() in content.lower() for word in original_prompt.split()[:5] if len(word) > 3)):
                    logger.warning("LLM step content appears unrelated to original prompt. Original: '%s', Step content: '%s'", 
                                 original_prompt[:100], content[:200])
                    # Prepend original prompt to provide context
                    content = f"User request: {original_prompt}\n\n{content}"
                    logger.info("Prepending original prompt to LLM step content for context")
                
                logger.debug("LLM step content (first 200 chars): %s", content[:200])
                if self.config.verbose and hasattr(self.llm, "stream"):
                    logger.info("Streaming LLM output for step %s...", step_count)
                    streamed_text = ""
                    stream_completed = False
                    try:
                        stream_gen = self.llm.stream(content, model=step.get("model"))
                        chunk_count = 0
                        # Consume all chunks from the generator - this will wait for Ollama to finish
                        for chunk in stream_gen:
                            chunk_count += 1
                            if chunk:
                                if self.config.verbose:
                                    sys.stdout.write(chunk)
                                    sys.stdout.flush()
                                streamed_text += chunk
                        
                        # Generator is now exhausted, which means Ollama has finished
                        # Add a small delay to ensure all output is flushed
                        time.sleep(0.1)  # Small delay to ensure all output is captured
                        stream_completed = True
                        if self.config.verbose:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        logger.info("Stream completed. Received %d chunks, total length: %d", chunk_count, len(streamed_text))
                    except StopIteration:
                        # Generator exhausted normally
                        stream_completed = True
                        logger.debug("Stream generator exhausted normally")
                    except Exception as e:
                        logger.exception("Error while streaming LLM: %s", e)
                        if not stream_completed:
                            logger.warning("Streaming failed, falling back to non-streaming generation")
                            try:
                                out = self._ask_llm(content, model=step.get("model"))
                                streamed_text = out.get("text", "")
                                stream_completed = True
                            except Exception as e2:
                                logger.exception("Fallback generation also failed: %s", e2)
                                streamed_text = ""
                    finally:
                        # Ensure we have output even if streaming was interrupted
                        if not streamed_text and not stream_completed:
                            logger.warning("Streamed text is empty and stream not completed, attempting non-streaming fallback")
                            try:
                                out = self._ask_llm(content, model=step.get("model"))
                                streamed_text = out.get("text", "")
                            except Exception as e2:
                                logger.exception("Fallback generation failed: %s", e2)
                                streamed_text = ""
                    
                    last_output = {"output": streamed_text, "success": True}
                    logger.info("LLM step completed. Output length: %d, stream_completed: %s", len(streamed_text), stream_completed)
                else:
                    out = self._ask_llm(content, model=step.get("model"))
                    last_output = {"output": out.get("text", ""), "raw": out}
                
                # Check if LLM output contains code blocks that should be converted to tool steps
                output_text = last_output.get("output", "")
                if output_text and "```" in output_text:
                    # Check if this looks like file creation intent
                    prompt_lower = (content + " " + original_prompt).lower()
                    has_file_creation_intent = any(keyword in prompt_lower for keyword in [
                        "create", "write", "generate", "make", "add", "file", "code", "component", 
                        "application", "app", "implement", "build"
                    ])
                    
                    if has_file_creation_intent:
                        logger.info("LLM output contains code blocks and file creation intent detected. Extracting code blocks and converting to tool steps...")
                        # Extract code blocks and convert to tool steps
                        extracted_steps = self._extract_code_blocks_from_output(output_text)
                        if extracted_steps:
                            logger.info("Extracted %d tool steps from LLM output code blocks", len(extracted_steps))
                            # Execute the extracted tool steps
                            for extracted_step in extracted_steps:
                                if step_count >= self.config.max_steps:
                                    logger.warning("Reached max_steps while executing extracted steps")
                                    break
                                step_count += 1
                                
                                extracted_action = extracted_step.get("action", "").lower()
                                if extracted_action == "tool":
                                    tool_name = extracted_step.get("tool")
                                    tool_input = extracted_step.get("input", "")
                                    logger.info("Executing extracted step %s: tool=%s", step_count, tool_name)
                                    tool_res = self._run_tool(tool_name, tool_input, previous_output=last_output)
                                    last_output = tool_res
                                    
                                    # Store tool execution
                                    memory_item = {
                                        "tool": tool_name,
                                        "input": tool_input,
                                        "output": tool_res,
                                        "timestamp": time.time(),
                                        "action_type": "tool_execution",
                                        "success": tool_res.get("success", True) if isinstance(tool_res, dict) else True,
                                        "extracted_from_llm": True
                                    }
                                    summary = f"Executed {tool_name} (extracted from LLM output): {str(tool_input)[:100]}"
                                    self._store_to_memory(memory_item, summary=summary)
                                    
                                    # Check if tool execution failed
                                    if self.config.enable_error_recovery and isinstance(tool_res, dict) and not tool_res.get("success", True):
                                        error_msg = tool_res.get("error", "Tool execution failed")
                                        logger.warning("Extracted tool '%s' execution failed: %s", tool_name, error_msg)
                                        last_output["error_info"] = {
                                            "error": error_msg,
                                            "output": tool_res.get("output", ""),
                                            "tool": tool_name,
                                            "input": tool_input,
                                            "last_output": tool_res
                                        }
                                    
                                    # Display file changes if this tool modified files
                                    if self._tool_may_modify_files(tool_name, tool_input):
                                        try:
                                            tracker = get_change_tracker()
                                            if tracker.has_pending_changes():
                                                if self.config.review_mode:
                                                    tracker.review_changes()
                                                elif tracker.has_changes():
                                                    tracker.print_changes()
                                            elif tracker.has_changes():
                                                tracker.print_changes()
                                        except Exception as e:
                                            logger.debug("Failed to display changes: %s", e)
                                    
                                    # Run CI dry-run if tool modified files
                                    try:
                                        if self._tool_may_modify_files(tool_name, tool_input):
                                            logger.info("Detected possible file-modifying tool '%s'. Running CI dry-run...", tool_name)
                                            ci_dry = self._run_ci_dry_run()
                                            logger.info("CI dry-run result: success=%s", ci_dry.get("success"))
                                            self._store_to_memory({"ci_dry_run": ci_dry, "trigger_tool": tool_name, "timestamp": time.time()})
                                            
                                            ci_failed = not bool(ci_dry.get("success"))
                                            ci_output = ci_dry.get("output", "")
                                            rollback_happened = "rollback" in ci_output.lower() or "Rollback succeeded" in ci_output
                                            
                                            if rollback_happened:
                                                logger.warning("CI validation failed and rollback occurred. Tool operation '%s' was undone.", tool_name)
                                                tool_res["success"] = False
                                                tool_res["error"] = "Operation was rolled back due to CI validation failures"
                                                tool_res["output"] = tool_res.get("output", "") + f"\n[WARNING] Changes were rolled back: {ci_output}"
                                                last_output = tool_res
                                            
                                            if ci_failed and self.config.ci_auto_apply and not rollback_happened:
                                                logger.info("CI reported failures and ci_auto_apply is enabled. Attempting auto-apply via CI tool.")
                                                ci_apply = self._attempt_ci_auto_apply()
                                                logger.info("CI auto-apply result: success=%s", ci_apply.get("success"))
                                                self._store_to_memory({"ci_apply": ci_apply, "trigger_tool": tool_name, "timestamp": time.time()})
                                    except Exception as e:
                                        logger.exception("CI integration after extracted tool '%s' raised: %s", tool_name, e)
                                    
                                    if self.config.step_delay:
                                        time.sleep(self.config.step_delay)
                        else:
                            logger.warning("LLM output contains code blocks but extraction failed. Output snippet: %s", output_text[:500])
                    else:
                        logger.debug("LLM output contains code blocks but no file creation intent detected. Output snippet: %s", output_text[:200])
                elif output_text:
                    logger.debug("LLM output (no code blocks): %s", output_text[:200])
                
                self._store_to_memory({"prompt": content, "response": last_output, "timestamp": time.time()})
                if self.config.step_delay:
                    time.sleep(self.config.step_delay)
                continue

            # Unknown action: forward to LLM
            logger.debug("Unknown action '%s' — forwarding step to LLM for interpretation.", action)
            step_text = str(step)
            out = self._ask_llm(step_text)
            last_output = {"output": out.get("text", ""), "raw": out}
            self._store_to_memory({"prompt": step_text, "response": last_output, "timestamp": time.time()})

        # End of steps — return last_output if available
        final_text = ""
        success = True
        error_info = None
        if isinstance(last_output, dict):
            final_text = last_output.get("output") or last_output.get("text") or ""
            success = last_output.get("success", True)
            # If there's an error, include it in the final text
            if not success and last_output.get("error"):
                final_text = f"{final_text}\n[ERROR] {last_output.get('error')}"
                # Extract error_info if present
                error_info = last_output.get("error_info")
                if not error_info:
                    # Create error_info from last_output
                    error_info = {
                        "error": last_output.get("error", "Execution failed"),
                        "output": last_output.get("output", ""),
                        "tool": last_output.get("tool"),
                        "input": last_output.get("input"),
                        "last_output": last_output
                    }
        elif isinstance(last_output, str):
            final_text = last_output
        else:
            final_text = ""

        logger.info("Run finished. final_text length=%s, success=%s", len(final_text), success)
        self._store_to_memory({"prompt": prompt, "response": final_text, "timestamp": time.time()})
        result = {"final": final_text, "provider": "agent", "raw_plan": plan, "success": success}
        if error_info:
            result["error_info"] = error_info
        return result

    def _parse_simple_step_string(self, step: str) -> Dict[str, Any]:
        """
        Heuristic parser for quick planner strings such as:
            "TOOL: shell :: echo hello"
            "LLM: Summarize the repository"
        Returns a dict describing the step.
        """
        s = step.strip()
        if s.upper().startswith("TOOL:"):
            try:
                _, rest = s.split(":", 1)
                name_part, input_part = (rest.split("::", 1) + [""])[:2]
                tool_name = name_part.strip()
                tool_input = input_part.strip()
                return {"action": "tool", "tool": tool_name, "input": tool_input}
            except Exception:
                pass
        if s.upper().startswith("LLM:") or s.upper().startswith("PROMPT:") or s.upper().startswith("ASK:"):
            try:
                _, rest = s.split(":", 1)
                content = rest.strip()
                return {"action": "llm", "input": content}
            except Exception:
                pass
        return {"action": "llm", "input": s}

    def _extract_code_blocks_from_output(self, output_text: str) -> List[Dict[str, Any]]:
        """
        Extract code blocks from LLM output and return list of tool steps (file_editor or shell).
        Looks for patterns like:
        - Shell commands -> shell tool steps
        - Code files -> file_editor tool steps
        - "Create a new file named X" followed by code blocks
        
        Returns list of step dicts that can be inserted into the plan.
        """
        import re
        steps = []
        
        if not output_text or "```" not in output_text:
            return steps
        
        # Extract all code blocks with their language hints
        code_block_pattern = re.compile(
            r'```(?:(\w+))?\s*\n(.*?)```',
            re.DOTALL | re.IGNORECASE
        )
        
        code_blocks = code_block_pattern.findall(output_text)
        logger.info("Found %d code blocks in output, attempting to extract tool steps", len(code_blocks))
        
        # Shell command patterns - if code matches these, it's a command, not a file
        shell_command_patterns = [
            r'^\s*(npm|npx|yarn|pnpm)\s+',  # Package managers
            r'^\s*(git|docker|kubectl|helm)\s+',  # Common CLI tools
            r'^\s*(cd|mkdir|rm|cp|mv|ls|cat|grep|find|chmod|chown)\s+',  # Unix commands
            r'^\s*(python|node|ruby|php|java|go|rustc|cargo|g\+\+|gcc|clang)\s+',  # Runtimes/compilers
            r'^\s*(curl|wget|ssh|scp|rsync)\s+',  # Network/file transfer
            r'^\s*(sudo|su|apt|yum|brew|pip|conda)\s+',  # System/package commands
            r'^\s*(echo|export|source|\.\/|\.\.\/)',  # Shell builtins/paths
        ]
        
        # Try to match code blocks with file paths mentioned nearby
        for i, (lang, code) in enumerate(code_blocks):
            code = code.strip()
            # Skip very short code blocks and JSON blocks (those are for plans, not code)
            if not code or len(code) < 10:
                continue
            # Skip if it looks like JSON (already tried parsing it)
            if lang and lang.lower() == "json":
                continue
            if code.strip().startswith("{") or code.strip().startswith("["):
                # Might be JSON, skip
                continue
            
            # Check if this is a shell command
            is_shell_command = False
            if lang and lang.lower() in ("bash", "sh", "shell", "cmd", "powershell"):
                is_shell_command = True
            else:
                # Check if code matches shell command patterns
                for pattern in shell_command_patterns:
                    if re.match(pattern, code, re.IGNORECASE):
                        is_shell_command = True
                        break
            
            if is_shell_command:
                # This is a shell command - create shell tool step
                logger.info("Detected shell command in code block %d: %s", i+1, code[:50])
                steps.append({
                    "action": "tool",
                    "tool": "shell",
                    "input": code
                })
                continue
            
            # Look for file path mentions near this code block (look further back)
            block_start = output_text.find(f"```{lang or ''}")
            if block_start == -1:
                block_start = output_text.find("```")
            context_start = max(0, block_start - 500)  # Look back 500 chars
            context = output_text[context_start:block_start]
            
            # Multiple patterns to find file paths
            file_patterns = [
                r'(?:create|write|add|generate|make|file|save|as|named|call(?:ed)?)\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:named\s+)?["\']?([^\s"\']+\.(?:js|jsx|ts|tsx|py|java|cpp|c|h|html|css|json|xml|yaml|yml|md|txt|sh|bash))["\']?',
                r'([^\s"\']+\.(?:js|jsx|ts|tsx|py|java|cpp|c|h|html|css|json|xml|yaml|yml|md|txt|sh|bash))',  # Any file path
            ]
            
            file_path = None
            for pattern in file_patterns:
                file_match = re.search(pattern, context, re.IGNORECASE)
                if file_match:
                    file_path = file_match.group(1)
                    # Clean up the path (remove quotes, etc.)
                    file_path = file_path.strip('"\'')
                    break
            
            if not file_path:
                # Infer filename from language or context
                lang_ext_map = {
                    "javascript": "js", "js": "js", "jsx": "jsx",
                    "typescript": "ts", "ts": "ts", "tsx": "tsx",
                    "python": "py", "py": "py",
                    "java": "java",
                    "cpp": "cpp", "c++": "cpp", "c": "c",
                    "html": "html", "css": "css",
                    "json": "json", "xml": "xml",
                    "yaml": "yaml", "yml": "yml",
                    "markdown": "md", "md": "md",
                    "bash": "sh", "sh": "sh", "shell": "sh"
                }
                ext = lang_ext_map.get((lang or "").lower(), "txt") if lang else "txt"
                
                # React-specific detection
                if "import React" in code or "from 'react'" in code or "from \"react\"" in code:
                    if ext == "js":
                        ext = "jsx"
                    # Try to extract component name
                    comp_match = re.search(r'(?:function|const|class)\s+(\w+)', code)
                    if comp_match:
                        comp_name = comp_match.group(1)
                        file_path = f"src/{comp_name}.{ext}"
                    else:
                        file_path = f"src/App.{ext}"
                elif "function App" in code or "const App" in code or "export default App" in code:
                    file_path = f"src/App.{ext}"
                elif "export default" in code and ext in ["js", "jsx", "ts", "tsx"]:
                    # Try to extract component name
                    comp_match = re.search(r'(?:export\s+default\s+)?(?:function|const|class)\s+(\w+)', code)
                    if comp_match:
                        comp_name = comp_match.group(1)
                        file_path = f"src/{comp_name}.{ext}"
                    else:
                        file_path = f"src/Component{i+1}.{ext}"
                else:
                    file_path = f"src/Component{i+1}.{ext}"
            
            # Only create file step if we have substantial code and it's not a command
            # Additional check: if code looks like a single command line, it's probably a shell command
            lines = code.split('\n')
            if len(lines) == 1 and (' ' in code or code.startswith('./') or code.startswith('../')):
                # Single line with spaces or path - likely a command
                logger.info("Code block %d looks like a single command, skipping file creation", i+1)
                continue
            
            # Validation: ensure code has actual content (not just whitespace/comments)
            content_lines = [line.strip() for line in lines if line.strip()]
            if len(content_lines) < 2:
                logger.info("Code block %d has too few non-empty lines (%d), skipping file creation", i+1, len(content_lines))
                continue
            
            # Validation: for code files, check if there's actual code (not just comments)
            code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.php'}
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in code_extensions:
                non_comment_lines = 0
                for line in content_lines:
                    stripped = line.strip()
                    # Skip comment-only lines
                    if stripped and not stripped.startswith('#') and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.endswith('*/'):
                        non_comment_lines += 1
                
                if non_comment_lines < 2:
                    logger.info("Code block %d appears to be only comments (%d non-comment lines), skipping file creation", i+1, non_comment_lines)
                    continue
            
            # Validation: ensure code is substantial (at least 50 chars of actual content)
            if len(code.strip()) < 50:
                logger.info("Code block %d is too short (%d chars), skipping file creation", i+1, len(code.strip()))
                continue
            
            # Validation: check if file path is reasonable (not too generic or suspicious)
            if file_path.count('/') > 5:  # Too deeply nested
                logger.warning("File path %s is too deeply nested, skipping file creation", file_path)
                continue
            
            if file_path.startswith('../') or file_path.startswith('/'):  # Suspicious paths
                logger.warning("File path %s looks suspicious (absolute or parent directory), skipping file creation", file_path)
                continue
            
            steps.append({
                "action": "tool",
                "tool": "file_editor",
                "input": {
                    "action": "write",
                    "path": file_path,
                    "content": code
                }
            })
            logger.info("Extracted code block %d as file_editor step: %s (%d lines, %d chars)", i+1, file_path, len(content_lines), len(code))
        
        return steps

    def _ask_llm(self, prompt: str, *, model: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Call the LLM adapter using a couple of common method shapes.
        Returns the adapter's raw response or a normalized dict {"text": ...}
        """
        logger.debug("Asking LLM with prompt length %s...", len(prompt))
        try:
            prompt_to_send = expand_file_refs(prompt, repo_root=".")
            if hasattr(self.llm, "generate"):
                return self.llm.generate(prompt_to_send, model=model, timeout=timeout)
            if callable(self.llm):
                out = self.llm(prompt)
                if isinstance(out, dict):
                    return out
                return {"text": str(out)}
            raise RuntimeError("LLM adapter has no generate() method and is not callable.")
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            return {"text": "", "error": str(e)}

    # ---------------------------
    # Interactive / CLI helpers
    # ---------------------------
    def run_interactive(self) -> None:
        """
        Simple REPL: read prompt from stdin, run run_once, print final answer.
        """
        print("Agent REPL. Enter prompt (blank to quit).")
        while True:
            try:
                prompt = input("\n> ")
            except (KeyboardInterrupt, EOFError):
                print("\nexiting.")
                break
            if not prompt.strip():
                print("bye.")
                break
            try:
                res = self.run_once(prompt)
                final = res.get("final") or res.get("raw") or ""
                print("\n=== RESULT ===\n")
                if isinstance(final, dict):
                    import json as _json
                    print(_json.dumps(final, indent=2))
                else:
                    print(final)
            except Exception as e:
                logger.exception("Run failed: %s", e)
                print("Error during run:", e)
    
    def run_todo(self, prompt: str, *, confirm: bool = True, retry_on_fail: int = 1, interactive_confirm: Callable[[List[Dict[str, Any]]], bool] = None) -> Dict[str, Any]:
        """
        High-level: generate a to-do list from prompt, show to user (or call interactive_confirm),
        then execute each step sequentially.

        Args:
            prompt: user request
            confirm: if True, ask for confirmation before executing the list (interactive REPL only)
            retry_on_fail: how many times to attempt a failing step (0 = don't retry)
            interactive_confirm: optional function(List[steps]) -> bool. If provided, it's called instead of CLI input for confirmation.

        Returns:
            dict summary containing per-step results and overall status.
        """
        # generate todos
        todos = []
        try:
            todos = self.planner.generate_todo_list(prompt, memory=self._safe_memory_snapshot())
        except Exception as e:
            logger.exception("Failed to generate todo list: %s", e)
            return {"final": "", "success": False, "error": "planner_failed", "details": str(e)}

        if not todos:
            logger.info("Planner returned no todos; nothing to do.")
            return {"final": "", "success": True, "details": "no_todos"}

        # show todos and ask for confirmation if requested and running interactively
        if confirm:
            proceed = True
            if interactive_confirm:
                try:
                    proceed = bool(interactive_confirm(todos))
                except Exception:
                    proceed = False
            else:
                # only try console interactive confirm if running in a TTY
                try:
                    if sys.stdin and sys.stdin.isatty():
                        print("\nPlanned to-do list:")
                        for i, t in enumerate(todos, 1):
                            desc = t.get("input") or t.get("tool") or str(t)
                            if t.get("action") == "tool":
                                print(f"{i}. [TOOL:{t.get('tool')}] {desc}")
                            else:
                                print(f"{i}. {desc}")
                        ans = input("\nExecute these steps? (y/N): ").strip().lower()
                        proceed = ans in ("y", "yes")
                    else:
                        # not interactive; proceed only if confirm is False
                        proceed = False
                except Exception:
                    proceed = False

            if not proceed:
                logger.info("User declined to execute todo list.")
                return {"final": "", "success": False, "details": "user_declined"}

        # Execute sequentially
        step_results = []
        overall_ok = True
        for idx, step in enumerate(todos, start=1):
            logger.info("Executing todo %d/%d: %s", idx, len(todos), step)
            attempt = 0
            step_ok = False
            last_res = None
            while attempt <= retry_on_fail and not step_ok:
                attempt += 1
                try:
                    # Normalize step shape and delegate to existing run logic
                    # If step is a tool instruction
                    if step.get("action") == "tool":
                        tool_name = step.get("tool")
                        tool_input = step.get("input", "")
                        res = self._run_tool(tool_name, tool_input)
                    elif step.get("action") in ("llm", "think", None, ""):
                        # call LLM with content
                        content = step.get("input") or step.get("prompt") or ""
                        # support streaming if verbose
                        if self.config.verbose and hasattr(self.llm, "stream"):
                            out_text = ""
                            try:
                                for chunk in self.llm.stream(content, model=step.get("model")):
                                    if self.config.verbose:
                                        sys.stdout.write(chunk)
                                        sys.stdout.flush()
                                    out_text += chunk
                                if self.config.verbose:
                                    sys.stdout.write("\n")
                                res = {"output": out_text, "success": True}
                            except Exception:
                                res = self._ask_llm(content, model=step.get("model"))
                        else:
                            res = self._ask_llm(content, model=step.get("model"))
                    else:
                        # unknown action: forward entire step to LLM for interpretation
                        res = self._ask_llm(str(step))

                    last_res = self._normalize_tool_result(res)

                    # Store step result in memory
                    self._store_to_memory({"todo_index": idx, "step": step, "result": last_res, "timestamp": time.time()})
                    
                    # Display file changes if this step modified files
                    if step.get("action") == "tool" and self._tool_may_modify_files(step.get("tool", ""), step.get("input", "")):
                        try:
                            tracker = get_change_tracker()
                            if tracker.has_pending_changes():
                                # In review mode, show pending changes and ask for approval
                                if self.config.review_mode:
                                    tracker.review_changes()
                                elif tracker.has_changes():
                                    tracker.print_changes()
                            elif tracker.has_changes():
                                tracker.print_changes()
                        except Exception as e:
                            logger.debug("Failed to display changes: %s", e)

                    # If step likely modified files, run CI dry-run and possibly auto-apply
                    try:
                        if step.get("action") == "tool" and self._tool_may_modify_files(step.get("tool", ""), step.get("input", "")):
                            ci_dry = self._run_ci_dry_run()
                            self._store_to_memory({"ci_dry_run": ci_dry, "todo_index": idx, "timestamp": time.time()})
                            if not ci_dry.get("success") and self.config.ci_auto_apply:
                                ci_apply = self._attempt_ci_auto_apply()
                                self._store_to_memory({"ci_apply": ci_apply, "todo_index": idx, "timestamp": time.time()})
                                # if apply failed, consider step failed
                                if not ci_apply.get("success"):
                                    last_res["ci_apply"] = ci_apply
                                    step_ok = False
                                    # break or retry
                                    step_ok = False
                                    # continue retry loop if attempts remain
                                    continue
                    except Exception as e:
                        logger.exception("CI check after todo step raised: %s", e)

                    # Consider success if tool reports success True or LLM returned text
                    if isinstance(last_res, dict):
                        step_ok = bool(last_res.get("success", True) and last_res.get("output", "") is not None)
                    else:
                        step_ok = True
                except Exception as e:
                    logger.exception("Error while executing todo step: %s", e)
                    last_res = {"output": "", "success": False, "error": str(e)}
                    step_ok = False

            step_results.append({"step": step, "result": last_res, "ok": step_ok, "attempts": attempt})
            if not step_ok:
                overall_ok = False
                # depending on policy: either stop on first failure or continue; here we stop
                logger.warning("Step %d failed after %d attempts: %s", idx, attempt, last_res)
                break

        summary = {"prompt": prompt, "todos": todos, "results": step_results, "success": overall_ok}
        self._store_to_memory({"todo_run_summary": summary, "timestamp": time.time()})
        return {"final": summary, "success": overall_ok}

# ---------------------------
# CLI entrypoint helpers
# ---------------------------
def _build_argparser():
    p = argparse.ArgumentParser(prog="agent-loop", description="Run the coding agent loop (single-shot or REPL).")
    p.add_argument("--repl", action="store_true", help="Run interactive REPL loop.")
    p.add_argument("--prompt", type=str, help="Prompt to run in a single-shot run.")
    p.add_argument("--max-steps", type=int, help="Override config.max_steps for this run.")
    return p


def _cli_main(planner: Any, memory: Any, tool_manager: Any, llm: Any):
    parser = _build_argparser()
    args = parser.parse_args()
    cfg = AgentLoopConfig()
    if args.max_steps:
        cfg.max_steps = args.max_steps
    loop = AgentLoop(planner=planner, memory=memory, tool_manager=tool_manager, llm=llm, config=cfg)
    if args.repl:
        loop.run_interactive()
        return
    if args.prompt:
        out = loop.run_once(args.prompt)
        final = out.get("final") or out.get("raw") or ""
        if isinstance(final, dict):
            import json as _json
            print(_json.dumps(final, indent=2))
        else:
            print(final)
        return
    parser.print_help()
