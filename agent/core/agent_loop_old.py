# agent/core/agent_loop.py
"""
Agent loop core.

Provides:
- AgentLoop: orchestration class that ties together planner, memory, tool_manager and an LLM.
- run_once(prompt): run a single planning+execution cycle and return final result.
- run_interactive(): simple REPL loop for CLI use.
- Flexible integration: tries a few plausible method names on planner/memory/tool_manager adapters
  so it will work with slight differences in those implementations.

Assumptions (best-effort):
- planner has a method `plan(prompt, memory)` or `create_plan(prompt, memory)` that returns either:
    * a string (final text/answer)
    * a dict representing a single step
    * a list of steps (each step is dict-like with at least an 'action' key)
- memory has `recall(query)` and `store(item)` or `load()` / `save()` alternatives.
- tool_manager can run tools by name. Preferred call is `tool_manager.run(tool_name, input=...)`
  or `tool_manager.execute(tool_name, input=...)` or `tool_manager.get_tool(tool_name).run(input)`.

This file purposefully uses defensive programming and logs clear messages so it can be dropped-in
and adapted to concrete planner/memory/tool_manager implementations in your project.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

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
    - max_steps: maximum number of execution steps in a single run (safety).
    - step_delay: optional pause (seconds) between steps (useful for rate-limited tools or readable REPL).
    - verbose: if True, print streaming outputs as they arrive.
    """
    max_steps: int = 20
    step_delay: float = 0.0
    verbose: bool = True


class AgentLoop:
    """
    Orchestrates the agent's sense-plan-act loop.

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

    # ---------------------------
    # Utilities for adapter interop
    # ---------------------------
    def _call_planner(self, prompt: str) -> Union[str, Dict, List]:
        """
        Call the planner using common method names; return whatever it returns.
        """
        # Try common planner entrypoints
        for name in ("plan", "create_plan", "generate_plan", "make_plan"):
            if hasattr(self.planner, name):
                try:
                    fn = getattr(self.planner, name)
                    # Some planners accept memory or context as second arg
                    try:
                        return fn(prompt, getattr(self.memory, "context", None) or self._safe_memory_snapshot())
                    except TypeError:
                        return fn(prompt)
                except Exception as e:
                    logger.exception("Planner method %s raised an exception: %s", name, e)
                    raise
        raise RuntimeError("Planner does not expose a recognized entrypoint (plan/create_plan/...).")

    def _safe_memory_snapshot(self) -> Dict[str, Any]:
        """
        Attempt to produce a lightweight snapshot / summary of memory to pass into planner.
        """
        try:
            if hasattr(self.memory, "snapshot"):
                return self.memory.snapshot()
            if hasattr(self.memory, "to_dict"):
                return self.memory.to_dict()
            # Some memories expose `load` or `recall_all`
            if hasattr(self.memory, "recall_all"):
                return {"history": self.memory.recall_all()}
            if hasattr(self.memory, "load"):
                # load may return the stored content
                return {"loaded": self.memory.load()}
            # last resort: empty
            return {}
        except Exception as e:
            logger.debug("Memory snapshot failed: %s", e)
            return {}

    def _store_to_memory(self, item: Any) -> None:
        """
        Store an item into memory using common method names.
        """
        try:
            if hasattr(self.memory, "store"):
                self.memory.store(item)
                return
            if hasattr(self.memory, "save"):
                self.memory.save(item)
                return
            if hasattr(self.memory, "append"):
                self.memory.append(item)
                return
            logger.debug("Memory does not expose store/save/append; skipping memory write.")
        except Exception as e:
            logger.exception("Failed to write to memory: %s", e)

    def _run_tool(self, tool_name: str, tool_input: Any, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a tool using ToolManager. Try several common method names and return a dict with results.
        The returned dict should include at least 'output' key (string) if possible.
        """
        logger.info("Executing tool '%s' with input: %s", tool_name, str(tool_input)[:200])
        # Try ToolManager.run(tool_name, input=..)
        try:
            if hasattr(self.tool_manager, "run"):
                res = self.tool_manager.run(tool_name, input=tool_input, timeout=timeout)
                return self._normalize_tool_result(res)
            if hasattr(self.tool_manager, "execute"):
                res = self.tool_manager.execute(tool_name, tool_input, timeout=timeout)
                return self._normalize_tool_result(res)
            # get tool object and call its run/execute method
            if hasattr(self.tool_manager, "get_tool"):
                tool = self.tool_manager.get_tool(tool_name)
                if hasattr(tool, "run"):
                    return self._normalize_tool_result(tool.run(tool_input))
                if hasattr(tool, "execute"):
                    return self._normalize_tool_result(tool.execute(tool_input))
            # as a fallback, if tool_manager exposes a dict-like mapping
            if hasattr(self.tool_manager, "tools") and tool_name in getattr(self.tool_manager, "tools"):
                tool = self.tool_manager.tools[tool_name]
                if hasattr(tool, "run"):
                    return self._normalize_tool_result(tool.run(tool_input))
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
            # ensure 'output' exists
            if "output" in res:
                return res
            # common fields: 'stdout', 'result', 'text'
            for key in ("stdout", "result", "text"):
                if key in res:
                    return {"output": res[key], **{k: v for k, v in res.items() if k != key}}
            # fallback: stringify
            return {"output": str(res), **res}
        # other primitives -> coerce to string
        return {"output": str(res), "success": True}

    # ---------------------------
    # Main loop logic
    # ---------------------------
    def run_once(self, prompt: str) -> Dict[str, Any]:
        """
        Run a single planning + execution cycle and return a final result dict.

        The flow:
        1. Ask the planner for a plan given the prompt and memory snapshot.
        2. If planner returns a simple string -> treat as final answer.
        3. If planner returns a sequence of steps -> iterate steps and execute:
            - action == 'tool' or 'execute': run tool by name
            - action == 'query' or 'llm' or 'think': call the llm with content
            - action == 'finish' or 'return': return the provided content
        4. Save interesting outputs to memory via memory.store/save if available.

        The function attempts to be forgiving about planner step shape to support many planners.
        """
        logger.info("Starting run_once with prompt: %s", prompt if len(prompt) < 200 else prompt[:200] + "...")
        plan = self._call_planner(prompt)

        # If planner returned a raw string, consider it the final answer
        if isinstance(plan, str):
            logger.debug("Planner returned string result; treating as final answer.")
            self._store_to_memory({"prompt": prompt, "response": plan, "timestamp": time.time()})
            return {"final": plan, "provider": "planner", "raw_plan": plan}

        # If planner returned a dict, allow single-step semantics
        if isinstance(plan, dict) and plan:
            steps = [plan]
        elif isinstance(plan, (list, tuple)):
            steps = list(plan)
        else:
            logger.warning("Planner returned unexpected type (%s). Will ask LLM directly.", type(plan))
            # fallback: ask LLM directly
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

            # Accept both dict-like and simple strings
            if isinstance(step, str):
                # if step is a string, probe for special markers like "TOOL: name :: input" or "LLM: <text>"
                parsed = self._parse_simple_step_string(step)
                step = parsed

            # Normalize step to dict
            if not isinstance(step, dict):
                logger.debug("Skipping non-dict/non-string step: %s", step)
                continue

            action = (step.get("action") or step.get("type") or "").lower()
            logger.info("Executing step %s: action=%s", step_count, action or "<none>")

            # Handle finish/return
            if action in ("finish", "return", "done", "final"):
                final = step.get("output") or step.get("result") or step.get("text") or step.get("content") or ""
                logger.info("Planner requested finish with: %s", final)
                self._store_to_memory({"prompt": prompt, "response": final, "timestamp": time.time()})
                return {"final": final, "provider": "planner", "raw_plan": plan}

            # Tool execution
            if action in ("tool", "execute", "run_tool", "call_tool"):
                tool_name = step.get("tool") or step.get("name") or step.get("tool_name")
                tool_input = step.get("input") or step.get("args") or step.get("content") or ""
                timeout = step.get("timeout")
                if not tool_name:
                    logger.warning("Tool step missing 'tool' or 'name': %s", step)
                    continue
                tool_res = self._run_tool(tool_name, tool_input, timeout=timeout)
                last_output = tool_res
                # optionally store or feed tool output back to planner/memory
                self._store_to_memory({"tool": tool_name, "input": tool_input, "output": tool_res, "timestamp": time.time()})
                # some planners expect the tool output to be pushed into next step as 'observation' — try to set it
                if isinstance(step, dict) and "store_observation_as" in step:
                    try:
                        key = step["store_observation_as"]
                        self._store_to_memory({key: tool_res})
                    except Exception:
                        pass
                # continue to next step
                if self.config.step_delay:
                    time.sleep(self.config.step_delay)
                continue

            # LLM query / think step
            if action in ("llm", "think", "query", "ask", "prompt"):
                content = step.get("input") or step.get("prompt") or step.get("content") or ""
                if not content:
                    logger.debug("LLM step missing content; skipping.")
                    continue
                # Support streaming output if available and configured verbose
                if self.config.verbose and hasattr(self.llm, "stream"):
                    logger.info("Streaming LLM output for step %s...", step_count)
                    streamed_text = ""
                    try:
                        for chunk in self.llm.stream(content, model=step.get("model")):
                            # print for CLI immediate feedback
                            if self.config.verbose:
                                sys.stdout.write(chunk)
                                sys.stdout.flush()
                            streamed_text += chunk
                        # newline after streaming
                        if self.config.verbose:
                            sys.stdout.write("\n")
                    except Exception as e:
                        logger.exception("Error while streaming LLM: %s", e)
                        # fallback to single-shot
                        out = self._ask_llm(content, model=step.get("model"))
                        streamed_text = out.get("text", "")
                    last_output = {"output": streamed_text, "success": True}
                else:
                    out = self._ask_llm(content, model=step.get("model"))
                    last_output = {"output": out.get("text", ""), "raw": out}
                self._store_to_memory({"prompt": content, "response": last_output, "timestamp": time.time()})
                if self.config.step_delay:
                    time.sleep(self.config.step_delay)
                continue

            # Unknown / unsupported action: attempt to stringify and send to LLM
            logger.debug("Unknown action '%s' — forwarding step to LLM for interpretation.", action)
            step_text = str(step)
            out = self._ask_llm(step_text)
            last_output = {"output": out.get("text", ""), "raw": out}
            self._store_to_memory({"prompt": step_text, "response": last_output, "timestamp": time.time()})

        # End of steps — return last_output if available
        final_text = ""
        if isinstance(last_output, dict):
            final_text = last_output.get("output") or last_output.get("text") or ""
        elif isinstance(last_output, str):
            final_text = last_output
        else:
            final_text = ""

        logger.info("Run finished. final_text length=%s", len(final_text))
        self._store_to_memory({"prompt": prompt, "response": final_text, "timestamp": time.time()})
        return {"final": final_text, "provider": "agent", "raw_plan": plan}

    def _parse_simple_step_string(self, step: str) -> Dict[str, Any]:
        """
        Heuristic parser for quick planner strings such as:
            "TOOL: shell :: echo hello"
            "LLM: Summarize the repository"
        Returns a dict describing the step.
        """
        s = step.strip()
        # TOOL: name :: input
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
        # default: let planner treat it as generic text to send to LLM
        return {"action": "llm", "input": s}

    def _ask_llm(self, prompt: str, *, model: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Call the LLM adapter using a couple of common method shapes.
        Returns the adapter's raw response or a normalized dict {"text": ...}
        """
        logger.debug("Asking LLM with prompt length %s...", len(prompt))
        try:
            if hasattr(self.llm, "generate"):
                return self.llm.generate(prompt, model=model, timeout=timeout)
            # fallback to direct call
            if callable(self.llm):
                # maybe llm is a function
                out = self.llm(prompt)
                if isinstance(out, dict):
                    return out
                return {"text": str(out)}
            # cannot call
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
                    # try to pretty print
                    import json as _json

                    print(_json.dumps(final, indent=2))
                else:
                    print(final)
            except Exception as e:
                logger.exception("Run failed: %s", e)
                print("Error during run:", e)

    # ---------------------------
    # Small CLI entrypoint for manual testing
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
        # print final
        final = out.get("final") or out.get("raw") or ""
        if isinstance(final, dict):
            import json as _json

            print(_json.dumps(final, indent=2))
        else:
            print(final)
        return
    parser.print_help()


# NOTE:
# This module provides AgentLoop class. The _cli_main function is a convenience entrypoint for manual testing
# — to use it you must provide concrete planner/memory/tool_manager/llm objects from your project.
#
# Example usage (in a separate script):
#
# from agent.core.agent_loop import _cli_main
# from agent.llm.model_selector import get_llm
# from agent.core.planner import Planner
# from agent.core.memory import Memory
# from agent.core.tool_manager import ToolManager
#
# planner = Planner(...)            # your implementation
# memory = Memory(...)              # your implementation
# tools = ToolManager(...)          # your implementation
# llm = get_llm("ollama")           # or get_llm(None) depending on your model_selector
#
# _cli_main(planner, memory, tools, llm)
#
