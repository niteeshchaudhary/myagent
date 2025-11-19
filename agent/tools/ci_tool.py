# agent/tools/ci_tool.py
"""
CI Tool for running linters/tests and optionally auto-fixing via an LLM.

Usage via ToolManager:
    tm.run("ci", input={"paths": ".", "attempt_fix": True, "use_pylint": False, "apply_patch": True})

The tool returns:
    {
      "output": "<human readable summary>",
      "success": bool,
      "details": { "validator": <ValidatorResult fields> }
    }
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from agent.core.validator import CodeValidator, ValidatorResult
except Exception:
    # Adjust import path if running as script
    from core.validator import CodeValidator, ValidatorResult  # type: ignore

# Minimal tool class — ToolManager will instantiate with no args
class Tool:
    """
    CI Tool exposes a run(input) method.
    Input (dict) accepts:
      - paths: path(s) to check (default: repo root)
      - use_pylint: bool (default: False)
      - pytest_args: optional pytest args string
      - attempt_fix: bool (default: False) - attempt to call llm to fix issues
      - llm: an LLM adapter instance (optional). If provided and attempt_fix True, will be used.
      - auto_rollback: bool (default: True)
      - commit_message: message for commits created by auto-fix
    """
    name = "ci"
    description = "Run linters and tests; optionally rollback and request LLM patches and apply them."

    def __init__(self):
        pass

    def run(self, input: Any = None, **kwargs) -> Dict[str, Any]:
        cfg = input or {}
        paths = cfg.get("paths") or "."
        use_pylint = cfg.get("use_pylint", False)
        pytest_args = cfg.get("pytest_args")
        attempt_fix = cfg.get("attempt_fix", False)
        apply_patch = cfg.get("apply_patch", True)
        auto_rollback = cfg.get("auto_rollback", True)
        commit_message = cfg.get("commit_message", "ci: apply linter/test fixes (LLM)")
        llm = cfg.get("llm")  # expected to be an LLM adapter instance compatible with generate()

        validator = CodeValidator(repo_path=cfg.get("repo_path", "."), python_bin=cfg.get("python_bin", "python"))

        try:
            if attempt_fix and not llm:
                return {"output": "attempt_fix requested but no LLM provided", "success": False, "details": {}}

            result = validator.validate_and_fix(
                llm=llm if attempt_fix else None,
                paths=paths,
                use_pylint=use_pylint,
                pytest_args=pytest_args,
                auto_rollback=auto_rollback,
                attempt_fix=attempt_fix,
                commit_message=commit_message,
                max_fix_attempts=cfg.get("max_fix_attempts", 2),
            )

            summary = self._format_summary(result)
            return {"output": summary, "success": result.ok, "details": result.__dict__}
        except Exception as e:
            return {"output": f"CI tool failed: {e}", "success": False, "details": {}}

    def _format_summary(self, res: ValidatorResult) -> str:
        if res.ok:
            return "Checks passed. Linters and tests OK."
        parts = []
        parts.append("Checks failed.")
        if not res.linter_ok:
            parts.append("Linters reported issues.")
        if not res.tests_ok:
            parts.append("Tests failed.")
        if res.applied_patch:
            parts.append("An LLM-generated patch was applied (see details).")
        if res.message:
            parts.append("Message: " + res.message)
        return " ".join(parts)
