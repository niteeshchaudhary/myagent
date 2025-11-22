# agent/core/planner.py
"""
Planner component for the coding agent.

Responsibilities:
- Produce a plan (either a final answer or a sequence of steps) given a user prompt and optional memory/context.
- Support both simple rule-based planning (heuristics) and LLM-driven planning.
- Attempt to return a structured plan as a list of step dicts with common keys:
    { "action": "tool" | "llm" | "finish", "tool": "<tool_name>", "input": "<...>", "meta": {...} }
- Be defensive and forgiving: parse imperfect JSON, extract code blocks, or fall back to line-based steps.

Integration:
- Accepts an LLM adapter instance at init (should expose generate(prompt, ...) and stream(...)).
- If no LLM is provided, it will try to load one via model_selector.get_llm() (best-effort).
- Exposes `plan(prompt, memory)` and alias `create_plan(...)`. These are the methods `AgentLoop` expects.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from agent.utils.file_ref import expand_file_refs

# Prefer project logger if present
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
class PlannerConfig:
    """
    Configuration for the Planner.
    - llm_provider: name used by model_selector (optional) if no llm instance provided.
    - model: model name to pass to LLM when generating plans (optional).
    - max_steps: maximum number of steps to request/accept in a plan.
    - temperature: sampling temperature to use when calling the LLM for planning.
    - prefer_structured: if True, ask LLM to output JSON steps; else ask for a short plan/answer.
    - plan_template_path: optional path to a custom plan template (string file).
    - max_prompt_tokens: maximum tokens to allow in prompts (default: 10000 for Groq on_demand tier).
                        Prompts exceeding this will be truncated/summarized.
    """
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    max_steps: int = 12
    temperature: float = 0.0
    prefer_structured: bool = True
    plan_template_path: Optional[str] = None
    max_prompt_tokens: int = 10000  # Conservative limit for Groq on_demand tier (12000 TPM limit)
    # prompt prefix/suffix to help LLM produce step-by-step plan
    prompt_prefix: Optional[str] = None
    prompt_suffix: Optional[str] = None


class Planner:
    """
    Planner orchestrates plan generation.

    Example:
        from agent.core.planner import Planner, PlannerConfig
        p = Planner(PlannerConfig(model="gpt-4o"))
        plan = p.plan("How do I add a new API endpoint to this repo?", memory={"files": [...]})
    """

    def __init__(self, llm: Any = None, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.llm = llm
        # lazily import model_selector if llm is not provided and we have a configured provider
        if self.llm is None and self.config.llm_provider:
            try:
                from agent.llm.model_selector import get_llm

                # pass model and config through
                kwargs = {}
                if self.config.model:
                    kwargs["model"] = self.config.model
                self.llm = get_llm(self.config.llm_provider, **kwargs)
            except Exception as e:
                logger.debug("Could not auto-load LLM via model_selector: %s", e)
                self.llm = None

        # load template if present
        self._plan_template = None
        if self.config.plan_template_path:
            try:
                with open(self.config.plan_template_path, "r", encoding="utf-8") as fh:
                    self._plan_template = fh.read()
            except Exception as e:
                logger.warning("Failed to read plan_template_path '%s': %s", self.config.plan_template_path, e)
        
        # Initialize RAG retriever for codebase context (lazy)
        self._retriever = None
        # Get repo_root from config, or use current working directory
        self.repo_root = getattr(self.config, "repo_root", None) or os.getcwd()
        logger.info("Planner initialized with repo_root: %s", self.repo_root)
        
        # Detect if using Groq and adjust token limits
        if self.llm:
            llm_class_name = self.llm.__class__.__name__ if hasattr(self.llm, "__class__") else ""
            provider_attr = getattr(self.llm, "provider", None)
            config_has_groq = False
            if hasattr(self.llm, "config"):
                config = self.llm.config
                if hasattr(config, "model") and config.model:
                    model_name = str(config.model).lower()
                    config_has_groq = "groq" in model_name or any(x in model_name for x in ["qwen", "llama-3"])
            
            if ("groq" in str(provider_attr).lower() or 
                "GroqLLM" in llm_class_name or 
                "groq" in llm_class_name.lower() or
                config_has_groq):
                # Groq free tier: 6,000 TPM limit, be very conservative
                self.config.max_prompt_tokens = min(self.config.max_prompt_tokens, 5000)
                logger.info("Detected Groq LLM - reducing max_prompt_tokens to %d for free tier compatibility", 
                           self.config.max_prompt_tokens)

    # Public API
    def plan(self, prompt: str, memory: Optional[Union[Dict, str]] = None) -> Union[str, List[Dict[str, Any]]]:
        """
        Generate a plan for the provided prompt.

        Returns:
            - str: If the planner returns a direct textual answer (e.g., final result).
            - List[Dict]: Structured plan consisting of steps (recommended).
        """
        # Basic heuristic: very short prompts that look like questions can be answered directly
        if not prompt or len(prompt.strip()) == 0:
            return ""

        logger.info("Planner.plan() called for prompt length=%d", len(prompt))

        # Check if this is an execution request - these MUST use structured planning with tools
        prompt_lower = prompt.lower()
        is_execution_request = any(keyword in prompt_lower for keyword in [
            "run", "execute", "compile", "test", "build", "launch", "start"
        ])
        
        # If prefer_structured is False or prompt looks like "What is X?", ask for a short answer
        # BUT: Always use structured planning for execution requests
        if not is_execution_request and (not self.config.prefer_structured or self._looks_like_simple_question(prompt)):
            logger.debug("Using quick-answer heuristic (prefer_structured=%s)", self.config.prefer_structured)
            # If no LLM is available, return prompt as-is for downstream LLM
            if not self.llm:
                return prompt
            # Ask LLM for a short final answer (not a multi-step plan)
            ans = self._ask_llm_for_answer(prompt)
            return ans.get("text", "")

        # Otherwise, ask the LLM to create a structured plan
        # Try to get available tools from memory if available (agent loop may pass this)
        available_tools = None
        if isinstance(memory, dict):
            available_tools = memory.get("available_tools")
        
        # Truncate memory if needed to fit within token limits
        max_prompt_tokens = getattr(self.config, "max_prompt_tokens", 10000)
        truncated_memory = self._truncate_memory(memory, max_tokens=max_prompt_tokens // 3)  # Reserve 1/3 for memory
        
        plan_prompt = self._build_plan_prompt(prompt, memory=truncated_memory, available_tools=available_tools)
        
        # Truncate the full prompt if it still exceeds limits
        plan_prompt = self._truncate_prompt(plan_prompt, max_tokens=max_prompt_tokens)
        
        raw = None
        if not self.llm:
            logger.warning("No LLM available to produce a structured plan; falling back to string response.")
            return prompt

        try:
            prompt_tokens = self._estimate_tokens(plan_prompt)
            logger.debug("Requesting structured plan from LLM (model=%s, estimated_tokens=%d)", 
                        getattr(self.config, "model", None), prompt_tokens)
            if prompt_tokens > max_prompt_tokens:
                logger.warning("Prompt still exceeds token limit (%d > %d) after truncation", 
                             prompt_tokens, max_prompt_tokens)
            res = self.llm.generate(plan_prompt, model=self.config.model, temperature=self.config.temperature)
            raw = res.get("text") if isinstance(res, dict) else str(res)
            logger.info("LLM plan response length: %d", len(raw) if raw else 0)
            if raw:
                logger.debug("First 500 chars of plan response: %s", raw[:500])
        except Exception as e:
            logger.exception("LLM call to generate plan failed: %s", e)
            # fallback: return prompt (AgentLoop will handle asking LLM)
            return prompt

        # Parse the raw plan text into structured steps if possible
        steps = self._parse_plan_text(raw, original_prompt=prompt)
        logger.info("Parsed %d steps from planner response", len(steps))
        if steps:
            logger.debug("First step type: %s, action: %s", type(steps[0]), steps[0].get("action") if isinstance(steps[0], dict) else None)
            # Log the content of LLM steps to debug context issues
            for i, step in enumerate(steps):
                if isinstance(step, dict) and step.get("action") == "llm":
                    step_input = step.get("input", "")
                    logger.debug("LLM step %d content (first 200 chars): %s", i+1, step_input[:200] if step_input else "(empty)")
        if steps:
            # Trim or enforce max steps
            if len(steps) > self.config.max_steps:
                logger.debug("Trimming plan steps from %d to max_steps=%d", len(steps), self.config.max_steps)
                steps = steps[: self.config.max_steps]
            return steps

        # if parsing fails, return the raw text to be interpreted later
        return raw or ""

    # alias
    create_plan = plan

    # -------------------------
    # Internal helpers
    # -------------------------
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text string.
        Uses a simple approximation: ~4 characters per token (conservative estimate).
        For more accurate counting, consider using tiktoken, but this avoids extra dependencies.
        """
        if not text:
            return 0
        # Rough approximation: 1 token ≈ 4 characters
        # This is conservative and works reasonably well for English text
        return len(text) // 4
    
    def _truncate_memory(self, memory: Union[Dict, str], max_tokens: int) -> Union[Dict, str]:
        """
        Truncate or summarize memory to fit within token limit.
        Prioritizes keeping the most recent/relevant information.
        """
        if not memory:
            return memory
        
        if isinstance(memory, str):
            mem_tokens = self._estimate_tokens(memory)
            if mem_tokens <= max_tokens:
                return memory
            # Truncate string to fit
            max_chars = max_tokens * 4
            truncated = memory[:max_chars]
            logger.warning("Truncated memory string from %d to %d tokens", mem_tokens, max_tokens)
            return truncated + "\n[... memory truncated due to size ...]"
        
        if isinstance(memory, dict):
            # Estimate total tokens
            mem_str = json.dumps(memory, indent=2)
            mem_tokens = self._estimate_tokens(mem_str)
            
            if mem_tokens <= max_tokens:
                return memory
            
            # Try to intelligently reduce memory size
            # Priority: keep available_tools, recent items, summaries
            truncated_memory = {}
            
            # Always keep available_tools (small and important)
            if "available_tools" in memory:
                truncated_memory["available_tools"] = memory["available_tools"]
            
            # Keep file references if present (usually small)
            if "files" in memory:
                files = memory["files"]
                if isinstance(files, list):
                    # Limit to most recent files
                    truncated_memory["files"] = files[:20]  # Keep first 20 files
                else:
                    truncated_memory["files"] = files
            
            # ALWAYS prioritize conversation_history - it's critical for context
            if "conversation_history" in memory:
                conv_history = memory["conversation_history"]
                if isinstance(conv_history, list):
                    # Keep most recent conversations (they're already sorted newest-first)
                    truncated_memory["conversation_history"] = conv_history[:15]  # Keep last 15 conversations
                    if len(conv_history) > 15:
                        truncated_memory["_truncated_conversations"] = len(conv_history) - 15
            
            # Keep recent items (if memory has items list)
            if "items" in memory and isinstance(memory["items"], list):
                # Keep most recent items, but limit total
                items = memory["items"][:10]  # Keep first 10 items
                truncated_memory["items"] = items
                truncated_memory["_truncated_items"] = len(memory["items"]) - len(items)
            
            # Copy other small keys (but prioritize conversation_history)
            for key, value in memory.items():
                if key not in ("available_tools", "files", "items", "conversation_history"):
                    value_str = json.dumps(value) if not isinstance(value, str) else value
                    if self._estimate_tokens(value_str) < 500:  # Keep small values
                        truncated_memory[key] = value
            
            # Check if truncated version fits
            truncated_str = json.dumps(truncated_memory, indent=2)
            truncated_tokens = self._estimate_tokens(truncated_str)
            
            if truncated_tokens > max_tokens:
                # Still too large, create a minimal summary
                summary = {
                    "available_tools": truncated_memory.get("available_tools", []),
                    "_summary": f"Memory truncated: original had {mem_tokens} tokens, reduced to fit {max_tokens} token limit",
                    "_original_keys": list(memory.keys())
                }
                logger.warning("Memory too large (%d tokens), using minimal summary", mem_tokens)
                return summary
            
            logger.warning("Truncated memory from %d to %d tokens", mem_tokens, truncated_tokens)
            return truncated_memory
        
        return memory
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Truncate a prompt string to fit within token limit.
        Tries to preserve the most important parts (instructions, user request).
        """
        prompt_tokens = self._estimate_tokens(prompt)
        if prompt_tokens <= max_tokens:
            return prompt
        
        # Try to keep the beginning (instructions) and end (user request)
        # Split roughly 60% instructions, 40% user request
        max_chars = max_tokens * 4
        truncation_msg = "\n\n[... middle content truncated due to size ...]\n\n"
        truncation_msg_chars = len(truncation_msg)
        
        if len(prompt) <= max_chars:
            return prompt
        
        # Calculate how much we can keep after reserving space for truncation message
        available_chars = max_chars - truncation_msg_chars
        keep_start = int(available_chars * 0.6)  # Keep first 60% of available
        keep_end = available_chars - keep_start   # Keep remaining from the end
        
        truncated = prompt[:keep_start] + truncation_msg + prompt[-keep_end:]
        logger.warning("Truncated prompt from %d to ~%d tokens", prompt_tokens, max_tokens)
        return truncated
    
    def _looks_like_simple_question(self, prompt: str) -> bool:
        """
        Heuristic that returns True if prompt appears to expect a short answer rather than a multi-step plan.
        Excludes execution requests (run, execute, compile, etc.) which should always use tools.
        """
        p = prompt.strip().lower()
        
        # Execution requests should NEVER be treated as simple questions
        execution_keywords = ["run", "execute", "compile", "test", "build", "launch", "start"]
        if any(keyword in p for keyword in execution_keywords):
            return False
        
        q_words = ("?", "explain", "what is", "who is", "when is", "why", "define", "difference between")
        if any(p.startswith(w) for w in q_words):
            return True
        # very short prompts likely direct (but not execution requests)
        if len(p.split()) <= 6:
            return True
        return False

    def _ask_llm_for_answer(self, prompt: str) -> Dict[str, Any]:
        """
        Ask the LLM for a concise single-shot answer.
        """
        try:
            return self.llm.generate(prompt, model=self.config.model, temperature=self.config.temperature)
        except Exception as e:
            logger.exception("LLM single-shot answer failed: %s", e)
            return {"text": ""}

    def _get_codebase_context(self, prompt: str, max_chunks: int = 5, memory: Optional[Union[Dict, str]] = None) -> str:
        """
        Retrieve relevant codebase context using RAG.
        Returns a formatted string with codebase structure and relevant code snippets.
        """
        context_parts = []
        
        # Get codebase structure
        try:
            structure = self._get_codebase_structure()
            if structure:
                context_parts.append("=== CODEBASE STRUCTURE ===\n" + structure + "\n")
        except Exception as e:
            logger.warning("Failed to get codebase structure: %s", e)
        
        # Get relevant action history from memory
        action_history_context = self._get_action_history_context(prompt, memory, max_items=5)
        if action_history_context:
            context_parts.append(action_history_context)
            context_parts.append("\n")
        
        # Use RAG to retrieve relevant code snippets
        try:
            if self._retriever is None:
                try:
                    from agent.rag.retriever import Retriever
                    from agent.rag.config import RagConfig
                    # Load RAG config from YAML file (configs/rag.yaml)
                    # Use current working directory if repo_root not set
                    repo_root = self.repo_root if self.repo_root and self.repo_root != "." else os.getcwd()
                    logger.info("Initializing RAG retriever for repo_root: %s", repo_root)
                    cfg = RagConfig.from_env(repo_root=repo_root)
                    self._retriever = Retriever(cfg)
                    logger.info("RAG retriever initialized successfully")
                except Exception as e:
                    logger.warning("Could not initialize RAG retriever: %s", e)
                    logger.debug("RAG retriever initialization error details", exc_info=True)
                    self._retriever = None
            
            if self._retriever:
                logger.info("Retrieving codebase context for prompt: %s", prompt[:100])
                chunks = self._retriever.retrieve(prompt, k=max_chunks, use_hybrid=True)
                if chunks:
                    logger.info("Retrieved %d code chunks for context", len(chunks))
                    context_parts.append("=== RELEVANT CODE SNIPPETS ===\n")
                    context_parts.append("Use these code snippets to understand the codebase structure and existing patterns:\n\n")
                    for i, chunk in enumerate(chunks[:max_chunks], 1):
                        path = chunk.get("path", "unknown")
                        start_line = chunk.get("start_line")
                        end_line = chunk.get("end_line")
                        text = chunk.get("text", "")
                        source_type = chunk.get("source_type", "unknown")
                        
                        location = f"{path}"
                        if start_line:
                            location += f":{start_line}"
                            if end_line and end_line != start_line:
                                location += f"-{end_line}"
                        
                        context_parts.append(f"[{i}] {location} ({source_type})")
                        context_parts.append(f"```\n{text[:1000]}\n```\n")  # Limit each chunk to 1000 chars
                else:
                    logger.warning("RAG retrieval returned no chunks for prompt: %s", prompt[:100])
            else:
                logger.warning("RAG retriever not available - codebase context will be limited")
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)
            logger.debug("RAG retrieval error details", exc_info=True)
        
        result = "\n".join(context_parts) if context_parts else ""
        if result:
            logger.info("Codebase context retrieved: %d characters", len(result))
        else:
            logger.warning("No codebase context retrieved for prompt: %s", prompt[:100])
        return result
    
    def _get_action_history_context(self, prompt: str, memory: Optional[Union[Dict, str]], max_items: int = 5) -> str:
        """
        Extract relevant previous actions from memory that are related to the current prompt.
        Returns formatted string with relevant action history.
        """
        if not memory:
            return ""
        
        context_parts = []
        relevant_actions = []
        
        try:
            # Extract conversation history from memory
            if isinstance(memory, dict) and "conversation_history" in memory:
                conv_history = memory.get("conversation_history", [])
                
                # Simple keyword matching to find relevant actions
                prompt_lower = prompt.lower()
                prompt_keywords = set(prompt_lower.split())
                
                for conv in conv_history:
                    # Check if this action is relevant to the current prompt
                    relevance_score = 0
                    action_text = ""
                    
                    # Check prompt/request
                    prev_prompt = conv.get("prompt", "")
                    if prev_prompt:
                        prev_prompt_lower = prev_prompt.lower()
                        prev_keywords = set(prev_prompt_lower.split())
                        # Count matching keywords
                        matching = prompt_keywords.intersection(prev_keywords)
                        relevance_score += len(matching) * 2
                        action_text += f"Previous request: {prev_prompt[:200]}\n"
                    
                    # Check tool/action
                    tool_info = conv.get("tool", "")
                    if tool_info:
                        tool_lower = tool_info.lower()
                        if any(kw in tool_lower for kw in prompt_keywords if len(kw) > 3):
                            relevance_score += 3
                        action_text += f"Tool: {tool_info}\n"
                    
                    # Check input
                    input_data = conv.get("input", "")
                    if input_data:
                        input_str = str(input_data).lower()
                        if any(kw in input_str for kw in prompt_keywords if len(kw) > 3):
                            relevance_score += 2
                        action_text += f"Input: {str(input_data)[:200]}\n"
                    
                    # Check output/result
                    output_data = conv.get("output") or conv.get("response") or conv.get("result")
                    if output_data:
                        output_str = str(output_data).lower()
                        if any(kw in output_str for kw in prompt_keywords if len(kw) > 3):
                            relevance_score += 1
                        action_text += f"Result: {str(output_data)[:200]}\n"
                    
                    # Include if relevant (score > 0) or if it's a recent action
                    if relevance_score > 0 or len(relevant_actions) < 3:
                        success = conv.get("success", True)
                        relevant_actions.append({
                            "score": relevance_score,
                            "text": action_text,
                            "success": success,
                            "timestamp": conv.get("timestamp", 0)
                        })
                
                # Sort by relevance score (descending) and timestamp (newest first)
                relevant_actions.sort(key=lambda x: (x["score"], x["timestamp"]), reverse=True)
                
                # Take top relevant actions
                if relevant_actions:
                    context_parts.append("=== RELEVANT PREVIOUS ACTIONS ===\n")
                    context_parts.append("These are previous actions that are relevant to the current request. Use them to understand what was done before and build upon it:\n\n")
                    
                    for i, action in enumerate(relevant_actions[:max_items], 1):
                        context_parts.append(f"--- Relevant Action {i} ---")
                        context_parts.append(action["text"])
                        if not action["success"]:
                            context_parts.append("⚠️ This action FAILED - avoid repeating the same approach")
                        context_parts.append("")
                    
                    logger.info("Included %d relevant previous actions in context", len(relevant_actions[:max_items]))
        except Exception as e:
            logger.warning("Failed to extract action history context: %s", e)
            logger.debug("Action history extraction error details", exc_info=True)
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _get_codebase_structure(self) -> str:
        """
        Get a summary of the codebase structure (file tree, key config files, etc.)
        """
        structure_parts = []
        
        try:
            repo_root = self.repo_root
            if not os.path.exists(repo_root):
                return ""
            
            # Get directory structure (top 2 levels)
            structure_parts.append("Directory structure:")
            for root, dirs, files in os.walk(repo_root):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git', 'venv', 'env')]
                
                level = root.replace(repo_root, '').count(os.sep)
                if level > 2:  # Only show top 2 levels
                    dirs[:] = []  # Don't recurse deeper
                    continue
                
                indent = '  ' * level
                rel_path = os.path.relpath(root, repo_root)
                if rel_path == '.':
                    structure_parts.append(f"{indent}.")
                else:
                    structure_parts.append(f"{indent}{os.path.basename(root)}/")
                
                # Show key files at each level
                key_files = [f for f in files if any(f.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.toml', '.py', '.js', '.ts', '.jsx', '.tsx', '.md', 'package.json', 'requirements.txt', 'setup.py'])]
                for f in sorted(key_files)[:5]:  # Limit to 5 files per directory
                    structure_parts.append(f"{indent}  {f}")
            
            # Read key configuration files
            key_config_files = {
                'package.json': 'JavaScript/Node.js project configuration',
                'requirements.txt': 'Python dependencies',
                'setup.py': 'Python package setup',
                'pyproject.toml': 'Python project configuration',
                'tsconfig.json': 'TypeScript configuration',
                'README.md': 'Project documentation'
            }
            
            structure_parts.append("\nKey configuration files:")
            for config_file, description in key_config_files.items():
                config_path = os.path.join(repo_root, config_file)
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()[:500]  # First 500 chars
                            structure_parts.append(f"\n{config_file} ({description}):")
                            structure_parts.append(f"```\n{content}\n```")
                    except Exception:
                        pass
        
        except Exception as e:
            logger.debug("Error getting codebase structure: %s", e)
        
        return "\n".join(structure_parts)
    
    def _build_plan_prompt(self, prompt: str, memory: Optional[Union[Dict, str]] = None, available_tools: Optional[List[str]] = None) -> str:
        """
        Compose a prompt asking the LLM to return a structured plan.
        If a plan template file was provided, use it. Otherwise use a helpful default template
        that requests JSON of steps.
        """
        if self._plan_template:
            template = self._plan_template
            # allow simple placeholders
            template = template.replace("{{PROMPT}}", prompt)
            template = template.replace("{{MEMORY}}", json.dumps(memory) if memory else "")
            return template

        # default template (encourage JSON array of steps)
        parts = []
        if self.config.prompt_prefix:
            parts.append(self.config.prompt_prefix)
        
        parts.append(
            "You are a helpful coding assistant working with an existing codebase. Given the user's request, produce a step-by-step plan "
            "to accomplish the task.\n\n"
            "CRITICAL: Before creating or writing any file, SEARCH the codebase context above and choose an existing file to modify."
            "If the change affects an existing file, RETURN A GIT-STYLE UNIFIED DIFF (diff --git a/... b/...) wrapped in a ```diff code fence."
            "If you must add a new file, return a diff that adds the new file (git unified diff format)."
            "Do NOT return raw file contents unless the step is explicitly a file write tool step and you reference an exact path already present in the codebase context."
            "CRITICAL OUTPUT FORMAT: You MUST return ONLY a valid JSON array of step objects. Do NOT include any reasoning, explanation, or text outside the JSON array. "
            "Start your response with '[' and end with ']'. No markdown, no code blocks, just pure JSON.\n\n"
            "Each step is an object with keys:\n"
            "  - action: one of 'tool', 'llm', or 'finish'\n"
            "  - tool: (for 'tool' actions) the canonical tool name to call\n"
            "  - input: input to the tool or content for the LLM\n"
            "  - meta: optional metadata\n\n"
            "CRITICAL RULES:\n"
            "1. ALWAYS understand the existing codebase structure before creating or modifying files. Use the codebase context provided below.\n"
            "2. When the user asks to 'run', 'execute', 'compile', or 'test' something, you MUST use the 'shell' tool to actually execute commands.\n"
            "3. When the user asks to create, write, or generate code/files, you MUST use the 'file_editor' tool to create files. Do NOT just output code in LLM steps.\n"
            "4. For file creation, use: {\"action\": \"tool\", \"tool\": \"file_editor\", \"input\": {\"action\": \"write\", \"path\": \"filename.ext\", \"content\": \"file content\"}}\n"
            "5. Match the existing code style, patterns, and structure. Look at similar files in the codebase.\n"
            "6. Do not just provide instructions - use tools to perform actions.\n"
            "7. NEVER write code directly in LLM output - ALWAYS use file_editor tool to create files.\n"
            "8. DO NOT include reasoning or explanations in your response - ONLY return the JSON array.\n\n"
        )
        
        # Add codebase context - CRITICAL for understanding the codebase
        try:
            codebase_context = self._get_codebase_context(prompt, max_chunks=8, memory=memory)  # Include memory for action history
            if codebase_context:
                parts.append("=== CODEBASE CONTEXT ===")
                parts.append("IMPORTANT: Use the codebase context below to understand the existing code structure, patterns, and files.")
                parts.append("This context shows relevant code from the repository. Use it to:")
                parts.append("1. Understand what files exist and their structure")
                parts.append("2. Match existing code style and patterns")
                parts.append("3. Find where to add new code or make changes")
                parts.append("4. Understand navigation/routing patterns if working with web apps\n")
                parts.append(codebase_context)
                parts.append("\n")
            else:
                logger.warning("No codebase context available - planner will work without codebase knowledge")
                parts.append("NOTE: Codebase context is not available. You may need to ask the user for more details about their codebase structure.\n")
        except Exception as e:
            logger.warning("Failed to get codebase context: %s", e)
            logger.debug("Codebase context error details", exc_info=True)
        
        # Add available tools information
        if available_tools:
            parts.append("Available tools:\n")
            for tool_name in available_tools:
                parts.append(f"  - {tool_name}")
            parts.append("\n")
            parts.append(
                "Key tools:\n"
                "  - shell: Execute shell commands (use for running, compiling, testing files)\n"
                "  - file_editor or file: Read/write files (use for file operations)\n"
                "  - python: Execute Python code\n\n"
                "CRITICAL FILE CREATION RULE: When you generate code or content that needs to be saved to a file, "
                "you MUST use the 'file_editor' tool (or 'file' tool) with action='write' to actually create the file. "
                "Do NOT just output code in an 'llm' step - you must create tool steps that write files. "
                "Example: { \"action\": \"tool\", \"tool\": \"file_editor\", \"input\": { \"action\": \"write\", \"path\": \"src/App.js\", \"content\": \"...code here...\" } }\n\n"
            )
        
        if memory:
            # Format memory with emphasis on conversation history
            memory_parts = []
            
            # Prioritize conversation history
            if isinstance(memory, dict) and "conversation_history" in memory:
                conv_history = memory.get("conversation_history", [])
                if conv_history:
                    memory_parts.append("=== PREVIOUS ACTIONS AND CONVERSATIONS ===\n")
                    memory_parts.append("These are relevant previous actions based on the current request. Use them to understand context and avoid repeating work.\n")
                    for i, conv in enumerate(conv_history[:15], 1):  # Show last 15 conversations
                        prompt = conv.get("prompt", "")
                        response = conv.get("response", "")
                        tool_info = conv.get("tool", "")
                        success = conv.get("success", True)
                        todo_index = conv.get("todo_index")
                        
                        if prompt or response:
                            memory_parts.append(f"\n--- Previous Request {i} ---")
                            if prompt:
                                memory_parts.append(f"User asked: {prompt[:300]}{'...' if len(prompt) > 300 else ''}")
                            if response:
                                resp_str = str(response)[:300] + ('...' if len(str(response)) > 300 else '')
                                memory_parts.append(f"Agent response: {resp_str}")
                        elif tool_info:
                            memory_parts.append(f"\n--- Previous Action {i} ---")
                            memory_parts.append(f"Tool executed: {tool_info}")
                            if conv.get("input"):
                                input_str = str(conv.get("input"))[:200]
                                memory_parts.append(f"Command/Input: {input_str}")
                            if "output" in conv:
                                output_str = str(conv.get("output"))[:200]
                                memory_parts.append(f"Result: {output_str}")
                            if not success:
                                memory_parts.append("⚠️ This action FAILED - avoid repeating it")
                        elif todo_index is not None:
                            memory_parts.append(f"\n--- Previous Todo Step {i} ---")
                            step = conv.get("step", {})
                            step_desc = str(step.get("input", step.get("prompt", "")))[:200]
                            memory_parts.append(f"Step: {step_desc}")
                            if conv.get("result"):
                                result_str = str(conv.get("result"))[:200]
                                memory_parts.append(f"Result: {result_str}")
                    
                    memory_parts.append("\n")
            
            # Add other memory context
            other_memory = {}
            if isinstance(memory, dict):
                for key, value in memory.items():
                    if key != "conversation_history" and key != "available_tools":
                        # Only include small values to avoid token bloat
                        value_str = json.dumps(value) if not isinstance(value, str) else value
                        if self._estimate_tokens(value_str) < 1000:
                            other_memory[key] = value
            
            if other_memory:
                memory_parts.append("Other context:\n")
                memory_parts.append(json.dumps(other_memory, indent=2))
            
            if memory_parts:
                parts.append("Memory/context:\n" + "\n".join(memory_parts) + "\n")
                # Add instruction to use conversation history
                if isinstance(memory, dict) and "conversation_history" in memory and memory.get("conversation_history"):
                    parts.append(
                        "CRITICAL INSTRUCTIONS FOR USING MEMORY:\n"
                        "1. Review the previous actions above to understand what has already been done.\n"
                        "2. If the current request is similar to a previous one, build upon that work instead of starting from scratch.\n"
                        "3. If a previous action failed, learn from the error and try a different approach.\n"
                        "4. If files were created or modified in previous actions, reference them in your plan.\n"
                        "5. Avoid repeating actions that were already completed successfully.\n"
                        "6. Use the codebase context below to understand the current state of files.\n\n"
                    )
            else:
                # Fallback to JSON dump if no conversation history found
                parts.append(f"Memory/context (useful facts):\n{json.dumps(memory, indent=2)}\n")

        expanded_prompt = expand_file_refs(prompt, repo_root=getattr(self, "repo_root", "."))
        parts.append("User request (expanded file references):\n" + expanded_prompt + "\n")
        
        # Detect execution requests and provide specific guidance
        prompt_lower = expanded_prompt.lower()
        if any(keyword in prompt_lower for keyword in ["run", "execute", "compile", "test", "build", "launch", "start"]):
            parts.append(
                "CRITICAL: The user wants to execute/run something. You MUST use the 'shell' tool to actually execute commands. "
                "Do NOT just provide instructions - you must use tools to perform the action.\n"
                "For C++ files: Use shell tool with 'g++ filename.cpp -o output && ./output' (Linux/Mac) or 'g++ filename.cpp -o output.exe && output.exe' (Windows).\n"
                "For Python files: Use shell tool with 'python filename.py'.\n"
                "For other files: Determine the appropriate command and use the shell tool.\n"
            )
        
        # Detect file creation requests
        if any(keyword in prompt_lower for keyword in ["create", "write", "generate", "make", "code", "file", "application", "app", "react", "component"]):
            parts.append(
                "CRITICAL: The user wants to create files or code. You MUST:\n"
                "1. First understand the existing codebase structure from the context above.\n"
                "2. Match the existing code style, import patterns, and file organization.\n"
                "3. Use the 'file_editor' tool to create files. Do NOT just output code in LLM steps.\n"
                "4. Use tool steps with file_editor to write files.\n"
                "Example for creating a React component: {\"action\": \"tool\", \"tool\": \"file_editor\", \"input\": {\"action\": \"write\", \"path\": \"src/App.jsx\", \"content\": \"import React from 'react';\\n\\nfunction App() {\\n  return <div>Hello</div>;\\n}\\n\\nexport default App;\"}}\n"
                "Break down multiple files into separate tool steps, one per file.\n"
                "Do NOT use 'llm' steps to output code - use file_editor tool steps to actually create files.\n"
                "ALWAYS check the codebase structure to place files in the correct directories.\n\n"
            )
        
        parts.append(
            "\n=== OUTPUT FORMAT ===\n"
            "You MUST return ONLY a valid JSON array. No reasoning, no explanations, no markdown code blocks, just pure JSON starting with '[' and ending with ']'.\n\n"
            "Example output (copy this format exactly):\n"
            '[\n'
            '  { "action": "tool", "tool": "file_editor", "input": { "action": "write", "path": "src/App.js", "content": "import React..." } },\n'
            '  { "action": "tool", "tool": "shell", "input": "npm install", "meta": {} },\n'
            '  { "action": "finish", "output": "Done." }\n'
            "]\n\n"
            "REMEMBER: Start with '[' and end with ']'. No other text before or after. Just the JSON array.\n"
        )
        if self.config.prompt_suffix:
            parts.append(self.config.prompt_suffix)
        return "\n".join(parts)

    def _parse_plan_text(self, raw: Optional[str], original_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Try to parse a plan returned by the LLM into a list of step dicts.
        Handles:
         - bare JSON arrays
         - JSON inside markdown/code blocks
         - simple line-by-line "1) do X" lists -> converted to steps with action 'llm'
         - reasoning text with code blocks -> extracts code blocks as file_editor steps
        Returns an empty list on failure.
        """
        if not raw:
            return []

        text = raw.strip()
        logger.debug("Parsing plan text (len=%d)", len(text))

        # 1) Try to extract JSON code block first (```json ... ``` or ``` ... ```)
        code_block_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if code_block_match:
            candidate = code_block_match.group(1).strip()
            steps = self._try_parse_json(candidate)
            if steps:
                logger.info("Successfully parsed JSON from code block")
                return self._normalize_steps(steps, original_prompt=original_prompt)

        # 2) Try to parse entire text as JSON
        steps = self._try_parse_json(text)
        if steps:
            logger.info("Successfully parsed JSON from entire text")
            return self._normalize_steps(steps, original_prompt=original_prompt)

        # 3) Heuristic: find first bracketed JSON substring if present
        # Try multiple patterns to find JSON array
        bracket_patterns = [
            r"(\[[\s\S]*?\])",  # Simple array
            r'(\{[\s\S]*?"steps"[\s\S]*?:[\s\S]*?\[[\s\S]*?\][\s\S]*?\})',  # Object with "steps" key
        ]
        for pattern in bracket_patterns:
            bracket_match = re.search(pattern, text, flags=re.DOTALL)
            if bracket_match:
                candidate = bracket_match.group(1)
                steps = self._try_parse_json(candidate)
                if steps:
                    logger.info("Successfully parsed JSON from bracket pattern")
                    return self._normalize_steps(steps, original_prompt=original_prompt)

        # 4) Fallback: If we have code blocks, extract them as tool steps (shell or file_editor)
        # This handles cases where LLM returns reasoning text with code blocks
        code_block_pattern = re.compile(r'```(?:(\w+))?\s*\n(.*?)```', re.DOTALL | re.IGNORECASE)
        code_blocks = code_block_pattern.findall(text)
        
        if code_blocks:
            logger.warning("Planner returned reasoning text instead of JSON. Found %d code blocks, converting to tool steps", len(code_blocks))
            steps_list = []
            
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
            
            for i, (lang, code) in enumerate(code_blocks):
                code = code.strip()
                # Skip very short code blocks and JSON blocks (already tried those)
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
                    steps_list.append({
                        "action": "tool",
                        "tool": "shell",
                        "input": code
                    })
                    continue
                
                # Try to find filename in context before code block (look further back)
                block_start = text.find(f"```{lang or ''}")
                if block_start == -1:
                    # Try without language
                    block_start = text.find("```")
                context_start = max(0, block_start - 500)  # Look back 500 chars
                context = text[context_start:block_start]
                
                # Look for file path mentions with more patterns
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
                    # Infer from language and code content
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
                    ext = lang_ext_map.get((lang or "").lower(), "txt")
                    
                    # React-specific detection
                    if "import React" in code or "from 'react'" in code or "from \"react\"" in code or "from \"react\"" in code:
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
                    elif "export default" in code:
                        # Try to extract component name
                        comp_match = re.search(r'(?:export\s+default\s+)?(?:function|const|class)\s+(\w+)', code)
                        if comp_match:
                            comp_name = comp_match.group(1)
                            file_path = f"src/{comp_name}.{ext}"
                        else:
                            file_path = f"src/Component{i+1}.{ext}"
                    else:
                        file_path = f"src/Component{i+1}.{ext}"
                
                # Additional check: if code looks like a single command line, it's probably a shell command
                lines = code.split('\n')
                if len(lines) == 1 and (' ' in code or code.startswith('./') or code.startswith('../')):
                    # Single line with spaces or path - likely a command
                    logger.info("Code block %d looks like a single command, creating shell step instead", i+1)
                    steps_list.append({
                        "action": "tool",
                        "tool": "shell",
                        "input": code
                    })
                    continue
                
                steps_list.append({
                    "action": "tool",
                    "tool": "file_editor",
                    "input": {
                        "action": "write",
                        "path": file_path,
                        "content": code
                    }
                })
                logger.info("Converted code block %d to file_editor step: %s", i+1, file_path)
            
            if steps_list:
                logger.info("Returning %d tool steps extracted from code blocks (%d shell, %d file_editor)", 
                           len(steps_list), 
                           sum(1 for s in steps_list if s.get("tool") == "shell"),
                           sum(1 for s in steps_list if s.get("tool") == "file_editor"))
                return self._normalize_steps(steps_list, original_prompt=original_prompt)
        
        # 5) Final fallback: parse as numbered or bulleted list into LLM steps
        lines = [l.strip(" -\t\r\n") for l in text.splitlines() if l.strip()]
        parsed_steps: List[Dict[str, Any]] = []
        if lines:
            for ln in lines:
                # skip headings like "Steps:" or "Plan:"
                if re.match(r"^(steps?|plan|procedure)[:\-]*$", ln.strip(), flags=re.IGNORECASE):
                    continue
                # remove leading "1.", "a)" etc
                ln_clean = re.sub(r"^\s*[\d\-\.\)\(a-zA-Z]+\s*", "", ln)
                parsed_steps.append({"action": "llm", "input": ln_clean})
            if parsed_steps:
                return self._normalize_steps(parsed_steps, original_prompt=original_prompt)

        # 5) give up
        logger.debug("Could not parse plan into structured steps; returning empty list.")
        return []

    def _try_parse_json(self, candidate: str) -> List[Dict[str, Any]]:
        """
        Attempt to json.loads the candidate. If it yields a dict with 'steps' key, use that.
        Otherwise expect a list of step objects.
        Returns [] on failure.
        """
        try:
            data = json.loads(candidate)
        except Exception:
            # attempt to fix common issues: trailing commas, single quotes -> double quotes
            fixed = candidate.strip()
            fixed = re.sub(r",\s*]", "]", fixed)
            fixed = re.sub(r",\s*}", "}", fixed)
            fixed = fixed.replace("'", '"')
            try:
                data = json.loads(fixed)
            except Exception:
                return []
        # If a dict with top-level 'steps', extract
        if isinstance(data, dict):
            if "steps" in data and isinstance(data["steps"], list):
                return data["steps"]
            # if dict is itself a single step (action/output) wrap in list
            if {"action"}.issubset(set(data.keys())):
                return [data]
            # else, maybe it's {"plan": [...]} or similar
            for k in ("plan", "steps_list", "sequence"):
                if k in data and isinstance(data[k], list):
                    return data[k]
            return []
        if isinstance(data, list):
            return data
        return []

    def _extract_code_blocks_from_llm_input(self, input_text: str, original_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract code blocks from LLM step input and convert to tool steps.
        Similar to agent_loop's _extract_code_blocks_from_output but for planner context.
        """
        import re
        steps = []
        
        if not input_text or "```" not in input_text:
            return steps
        
        # Extract all code blocks with their language hints
        code_block_pattern = re.compile(
            r'```(?:(\w+))?\s*\n(.*?)```',
            re.DOTALL | re.IGNORECASE
        )
        
        code_blocks = code_block_pattern.findall(input_text)
        logger.debug("Found %d code blocks in LLM step input", len(code_blocks))
        
        # Shell command patterns
        shell_command_patterns = [
            r'^\s*(npm|npx|yarn|pnpm)\s+',
            r'^\s*(git|docker|kubectl|helm)\s+',
            r'^\s*(cd|mkdir|rm|cp|mv|ls|cat|grep|find|chmod|chown)\s+',
            r'^\s*(python|node|ruby|php|java|go|rustc|cargo|g\+\+|gcc|clang)\s+',
            r'^\s*(curl|wget|ssh|scp|rsync)\s+',
            r'^\s*(sudo|su|apt|yum|brew|pip|conda)\s+',
            r'^\s*(echo|export|source|\.\/|\.\.\/)',
        ]
        
        for i, (lang, code) in enumerate(code_blocks):
            code = code.strip()
            if not code or len(code) < 10:
                continue
            # Skip JSON blocks (those are for plans, not code)
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
                for pattern in shell_command_patterns:
                    if re.match(pattern, code, re.IGNORECASE):
                        is_shell_command = True
                        break
            
            if is_shell_command:
                steps.append({
                    "action": "tool",
                    "tool": "shell",
                    "input": code
                })
                continue
            
            # Look for file path mentions
            block_start = input_text.find(f"```{lang or ''}")
            if block_start == -1:
                block_start = input_text.find("```")
            context_start = max(0, block_start - 500)
            context = input_text[context_start:block_start]
            
            file_patterns = [
                r'(?:create|write|add|generate|make|file|save|as|named|call(?:ed)?)\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:named\s+)?["\']?([^\s"\']+\.(?:js|jsx|ts|tsx|py|java|cpp|c|h|html|css|json|xml|yaml|yml|md|txt|sh|bash))["\']?',
                r'([^\s"\']+\.(?:js|jsx|ts|tsx|py|java|cpp|c|h|html|css|json|xml|yaml|yml|md|txt|sh|bash))',
            ]
            
            file_path = None
            for pattern in file_patterns:
                file_match = re.search(pattern, context, re.IGNORECASE)
                if file_match:
                    file_path = file_match.group(1).strip('"\'')
                    break
            
            if not file_path:
                # Infer from language
                lang_ext_map = {
                    "javascript": "js", "js": "js", "jsx": "jsx",
                    "typescript": "ts", "ts": "ts", "tsx": "tsx",
                    "python": "py", "py": "py",
                    "java": "java",
                    "cpp": "cpp", "c++": "cpp", "c": "c",
                    "html": "html", "css": "css",
                    "yaml": "yaml", "yml": "yml",
                    "markdown": "md", "md": "md",
                    "bash": "sh", "sh": "sh", "shell": "sh"
                }
                ext = lang_ext_map.get((lang or "").lower(), "txt") if lang else "txt"
                
                if "import React" in code or "from 'react'" in code or "from \"react\"" in code:
                    if ext == "js":
                        ext = "jsx"
                    comp_match = re.search(r'(?:function|const|class)\s+(\w+)', code)
                    if comp_match:
                        file_path = f"src/{comp_match.group(1)}.{ext}"
                    else:
                        file_path = f"src/App.{ext}"
                elif "function App" in code or "const App" in code:
                    file_path = f"src/App.{ext}"
                elif "export default" in code:
                    comp_match = re.search(r'(?:export\s+default\s+)?(?:function|const|class)\s+(\w+)', code)
                    if comp_match:
                        file_path = f"src/{comp_match.group(1)}.{ext}"
                    else:
                        file_path = f"src/Component{i+1}.{ext}"
                else:
                    file_path = f"src/Component{i+1}.{ext}"
            
            # Validation checks
            lines = code.split('\n')
            if len(lines) == 1 and (' ' in code or code.startswith('./') or code.startswith('../')):
                continue
            
            content_lines = [line.strip() for line in lines if line.strip()]
            if len(content_lines) < 2:
                continue
            
            if len(code.strip()) < 50:
                continue
            
            if file_path.count('/') > 5:
                continue
            
            if file_path.startswith('../') or file_path.startswith('/'):
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
            logger.debug("Extracted code block %d as file_editor step: %s", i+1, file_path)
        
        return steps

    def _normalize_steps(self, raw_steps: Sequence, original_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Normalize various step shapes into canonical dicts.
        
        Args:
            raw_steps: Raw step data to normalize
            original_prompt: Original user prompt (optional, used to validate LLM step content)
        """
        steps_out: List[Dict[str, Any]] = []
        for item in raw_steps:
            if isinstance(item, dict):
                # ensure action key exists
                action = (item.get("action") or item.get("type") or "").lower()
                if not action:
                    # try to infer
                    if "tool" in item or "tool_name" in item:
                        action = "tool"
                    elif "output" in item or "result" in item:
                        action = "finish"
                    else:
                        action = "llm"
                step = {
                    "action": action,
                    "tool": item.get("tool") or item.get("tool_name"),
                    "input": item.get("input") or item.get("prompt") or item.get("content") or item.get("query"),
                    "meta": item.get("meta") or {},
                }
                
                # For LLM steps, ensure content relates to original prompt
                if action == "llm" and original_prompt:
                    step_input = step.get("input", "")
                    # If content is suspiciously short or doesn't relate to prompt, prepend original prompt
                    if not step_input or len(step_input) < 50:
                        logger.warning("LLM step has suspiciously short or empty content. Prepending original prompt.")
                        step["input"] = f"User request: {original_prompt}\n\n{step_input}" if step_input else original_prompt
                    elif not any(word.lower() in step_input.lower() for word in original_prompt.split()[:5] if len(word) > 3):
                        logger.warning("LLM step content doesn't appear related to original prompt. Prepending original prompt for context.")
                        step["input"] = f"User request: {original_prompt}\n\n{step_input}"
                
                # Check if LLM step input contains code blocks that should be converted to tool steps
                if action == "llm":
                    step_input = step.get("input", "")
                    if step_input and "```" in step_input:
                        # Check if this looks like file creation intent
                        prompt_lower = (str(original_prompt or "") + " " + str(step_input)).lower()
                        has_file_creation_intent = any(keyword in prompt_lower for keyword in [
                            "create", "write", "generate", "make", "add", "file", "code", "component",
                            "application", "app", "implement", "build"
                        ])
                        
                        if has_file_creation_intent:
                            logger.info("LLM step input contains code blocks and file creation intent. Attempting to extract tool steps...")
                            # Try to extract code blocks from the input
                            extracted_steps = self._extract_code_blocks_from_llm_input(step_input, original_prompt)
                            if extracted_steps:
                                logger.info("Converted LLM step with code blocks into %d tool steps", len(extracted_steps))
                                # Replace the LLM step with extracted tool steps
                                steps_out.extend(extracted_steps)
                                continue
                
                # include output/result if present
                if "output" in item:
                    step["output"] = item["output"]
                if "result" in item:
                    step["output"] = item["result"]
                steps_out.append(step)
            else:
                # non-dict item -> coerce to llm step
                input_content = str(item)
                if original_prompt and (not input_content or len(input_content) < 50):
                    input_content = original_prompt
                steps_out.append({"action": "llm", "input": input_content})
        return steps_out

    # -------------------------
    # Convenience / debug
    # -------------------------
    def explain_plan(self, prompt: str, memory: Optional[Union[Dict, str]] = None) -> str:
        """
        Generate a short natural-language explanation of the plan without returning structured steps.
        Useful for debugging or user-facing summaries.
        """
        if not self.llm:
            return "No LLM configured to explain the plan."
        try:
            p = self._build_plan_prompt(prompt, memory=memory)
            summary_prompt = (
                "In one or two sentences, summarize the plan you would produce for the following request:\n\n"
                + p
            )
            res = self.llm.generate(summary_prompt, model=self.config.model, temperature=0.0)
            return res.get("text", "") if isinstance(res, dict) else str(res)
        except Exception as e:
            logger.exception("explain_plan failed: %s", e)
            return ""
    
    def generate_todo_list(self, prompt: str, memory: Optional[Union[Dict, str]] = None, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Ask the planner/LLM to produce a concise to-do list for the user's request.

        Returns a list of normalized step dicts:
           [{ "action": "tool"|"llm"|"finish", "input": "...", "tool": "<name?>" }, ...]

        This method is defensive: if the LLM returns plain text, it will attempt to parse
        numbered or bulleted lists into steps. It's suitable to call before run_todo().
        """
        if not prompt or prompt.strip() == "":
            return []

        max_items = max_items or self.config.max_steps

        # Build a concise prompt encouraging JSON output
        preface = (
            "You are a concise coding assistant. Given the user's request, produce a "
            "short numbered to-do list (2-12 items) of practical implementation steps. "
            "Return ONLY json in the form: {\"todos\": [\"step 1\", \"step 2\", ...]}. "
            "Each todo should be a single sentence describing one action (e.g., 'Create a new CLI command `hello` that prints hello world', "
            "'Run flake8 and fix lint errors', 'Add tests in tests/test_hello.py').\n\n"
        )
        if memory:
            mem_str = json.dumps(memory) if not isinstance(memory, str) else memory
            preface += f"Useful context / memory:\n{mem_str}\n\n"

        plan_prompt = preface + "User request:\n" + prompt + "\n\nReturn JSON only."

        # If no LLM available, fallback to simplistic heuristics (split sentences)
        if not self.llm:
            # naive split on sentences / lines
            lines = [ln.strip() for ln in re.split(r"[。\n\r]|[.?!]\s+", prompt) if ln.strip()]
            todos = [{"action": "llm", "input": ln} for ln in lines[:max_items]]
            return todos

        try:
            resp = self.llm.generate(plan_prompt, model=self.config.model, temperature=0.0)
            raw_text = resp.get("text") if isinstance(resp, dict) else str(resp)
        except Exception as e:
            logger.exception("LLM failed to generate todo list: %s", e)
            raw_text = ""

        # Try to parse JSON
        todos: List[Dict[str, Any]] = []
        try:
            # extract JSON block if present
            m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text, flags=re.IGNORECASE)
            candidate = m.group(1) if m else raw_text
            parsed = json.loads(candidate)
            arr = parsed.get("todos") if isinstance(parsed, dict) else parsed
            if isinstance(arr, list):
                for t in arr[:max_items]:
                    ts = str(t).strip()
                    # Heuristic: if it looks like "Use tool X: do Y", extract tool name
                    tool_match = re.match(r"(?i)(?:use|run|call)\s+([A-Za-z0-9_-]+)[:\-]\s*(.*)", ts)
                    if tool_match:
                        tool = tool_match.group(1).lower()
                        content = tool_match.group(2).strip()
                        todos.append({"action": "tool", "tool": tool, "input": content})
                    else:
                        todos.append({"action": "llm", "input": ts})
        except Exception:
            # fallback: parse lines / bullets
            for ln in re.split(r"\n+", raw_text or prompt):
                s = ln.strip()
                if not s:
                    continue
                # strip leading list markers
                s = re.sub(r"^\s*[-\d\.\)\*]+\s*", "", s)
                if len(todos) >= max_items:
                    break
                todos.append({"action": "llm", "input": s})
        # enforce a limit
        return todos[:max_items]

    # quick manual test
if __name__ == "__main__":
    # Basic demonstration (does not require an LLM to run; will just show prompt composition)
    cfg = PlannerConfig(prefer_structured=True)
    planner = Planner(llm=None, config=cfg)
    demo_prompt = "Add a new CLI command `hello` that prints 'hello world' and a unit test for it."
    print("Plan prompt:\n", planner._build_plan_prompt(demo_prompt, memory={"repo_files": ["cli.py", "tests/"]})[:1000])
    print("\nAttempting to parse a sample JSON plan string:")
    sample = """
    ```json
    [
      {"action":"tool","tool":"shell","input":"git checkout -b feature/hello"},
      {"action":"tool","tool":"python","input":"apply_patch <<'PATCH'\n*** Begin Patch\n*** Add File: hello.py\nprint('hello world')\n*** End Patch\nPATCH"},
      {"action":"llm","input":"Write a short unit test that imports hello.py and asserts output"},
      {"action":"finish","output":"Added hello.py and a test"}
    ]
    ```
    """
    steps = planner._parse_plan_text(sample)
    print("Parsed steps:", steps)
