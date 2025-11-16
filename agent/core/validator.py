# agent/core/validator.py
"""
CodeValidator: run linters/tests, summarize failures, rollback and request LLM fixes.

Capabilities:
- run_linters(paths): runs flake8 and (optionally) pylint; returns stdout/stderr and exit codes.
- run_tests(path): runs pytest; returns stdout/stderr and exit code.
- git_snapshot(repo_path): returns current HEAD commit hash.
- rollback_to_commit(repo_path, commit): performs `git reset --hard <commit>` and returns output.
- create_branch_for_work(repo_path, branch_name): create & checkout a working branch.
- apply_patch_from_diff(repo_path, diff_text): writes diff to temp file and runs `git apply` then commit.
- validate_and_fix(repo_path, llm=None, apply_patch=True, auto_rollback=True): high-level flow:
    1) run linters & tests
    2) if pass -> return success
    3) if fail -> if auto_rollback True, roll back to HEAD prior to patch
    4) if llm provided -> request patch (LLM should return unified diff) and attempt to apply -> re-run checks
    5) return final status
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from agent.core.patcher import apply_unified_diff

# logging
try:
    from agent.utils.logger import get_logger

    logger = get_logger(__name__)
except Exception:
    import logging

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)


@dataclass
class ValidatorResult:
    ok: bool
    linter_ok: bool
    tests_ok: bool
    linter_output: str
    tests_output: str
    message: str = ""
    applied_patch: Optional[str] = None  # diff text applied (if any)


class CodeValidator:
    def __init__(self, repo_path: str = ".", python_bin: str = "python", env: Optional[Dict[str, str]] = None):
        """
        repo_path: working repo path where git commands are executed
        python_bin: python executable to use for flake8/pytest (can be "python3" or full path)
        env: optional env vars to pass into subprocesses (merged with os.environ)
        """
        self.repo_path = os.path.abspath(repo_path)
        self.python_bin = python_bin
        self.env = os.environ.copy()
        if env:
            self.env.update(env)

    # ---- utilities ----
    def _run(self, cmd: str, timeout: int = 300) -> Tuple[int, str, str]:
        """
        Run shell command in repo_path. Return (rc, stdout, stderr).
        Uses shlex.split for safety.
        """
        logger.debug("Running command: %s (cwd=%s)", cmd, self.repo_path)
        try:
            proc = subprocess.run(
                shlex.split(cmd),
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                timeout=timeout,
            )
            out = proc.stdout.decode("utf-8", errors="replace")
            err = proc.stderr.decode("utf-8", errors="replace")
            logger.debug("Command rc=%s stdout_len=%d stderr_len=%d", proc.returncode, len(out), len(err))
            return proc.returncode, out, err
        except subprocess.TimeoutExpired as e:
            logger.exception("Command timeout: %s", e)
            return 124, "", f"Timeout: {e}"

    # ---- git helpers ----
    def git_head(self) -> Optional[str]:
        rc, out, err = self._run("git rev-parse --verify HEAD")
        if rc != 0:
            logger.debug("git rev-parse failed: %s", err.strip())
            return None
        return out.strip()

    def create_branch(self, branch_name: str) -> Tuple[bool, str]:
        rc, out, err = self._run(f"git checkout -b {shlex.quote(branch_name)}")
        ok = rc == 0
        return ok, out + err

    def checkout(self, ref: str) -> Tuple[bool, str]:
        rc, out, err = self._run(f"git checkout {shlex.quote(ref)}")
        return rc == 0, out + err

    def rollback_to(self, commit: str) -> Tuple[bool, str]:
        """
        Hard reset to commit. WARNING: destructive if uncommitted changes exist.
        This method assumes the workflow used creates commits before large changes.
        """
        rc, out, err = self._run(f"git reset --hard {shlex.quote(commit)}")
        ok = rc == 0
        return ok, out + err

    def commit_all(self, message: str) -> Tuple[bool, str]:
        # Stage all and commit
        rc1, _, err1 = self._run("git add -A")
        if rc1 != 0:
            return False, err1
        rc2, out2, err2 = self._run(f"git commit -m {shlex.quote(message)}")
        ok = rc2 == 0
        return ok, out2 + err2

    # ---- linters/tests ----
    def run_linters(self, paths: Optional[str] = None, use_pylint: bool = False) -> Tuple[bool, str]:
        """
        Run flake8 over repo or paths. Optionally run pylint (slower).
        Returns (ok, output).
        """
        # prefer flake8 if installed
        flake = shutil.which("flake8")
        out_lines = []
        ok = True
        if flake:
            cmd = f"{shlex.quote(flake)} {paths or self.repo_path}"
            rc, out, err = self._run(cmd)
            out_lines.append(out)
            out_lines.append(err)
            if rc != 0:
                ok = False
        else:
            # try module invocation via python -m
            rc, out, err = self._run(f"{shlex.quote(self.python_bin)} -m flake8 {paths or self.repo_path}")
            out_lines.append(out)
            out_lines.append(err)
            if rc != 0:
                ok = False

        if use_pylint:
            pylint = shutil.which("pylint")
            if pylint:
                rc, out, err = self._run(f"{shlex.quote(pylint)} {paths or self.repo_path}")
            else:
                rc, out, err = self._run(f"{shlex.quote(self.python_bin)} -m pylint {paths or self.repo_path}")
            out_lines.append(out)
            out_lines.append(err)
            if rc != 0:
                ok = False

        return ok, "\n".join([ln for ln in out_lines if ln])

    def run_tests(self, pytest_args: Optional[str] = None) -> Tuple[bool, str]:
        """
        Run pytest in repo_path. pytest must be installed.
        Returns (ok, output).
        """
        pytest_bin = shutil.which("pytest") or f"{self.python_bin} -m pytest"
        cmd = f"{pytest_bin} {pytest_args or ''}"
        rc, out, err = self._run(cmd)
        ok = rc == 0
        return ok, out + err

    # ---- apply patch -->
    def apply_diff_patch(self, diff_text: str, commit_message: Optional[str] = "Apply LLM patch") -> Tuple[bool, str]:
        """
        Try to apply a unified diff patch.
        Strategy:
         1) Attempt to apply hunks in-place using our Python patcher (conservative).
         2) If that fails (or partial apply), fall back to `git apply --index` and commit.
         3) If git apply fails, attempt `patch` program. If both fail, return failure and logs.
        """
        # 1) Try python hunk applier (non-destructive and precise)
        try:
            ok, report = apply_unified_diff(diff_text, repo_root=self.repo_path, dry_run=False)
        except Exception as e:
            ok = False
            report = {"error": f"Python patcher raised exception: {e}"}

        if ok:
            # commit the applied changes if git repo present
            try:
                ok_commit, commit_out = self.commit_all(commit_message)
                if ok_commit:
                    return True, "Applied patch hunks and committed: " + commit_out
                else:
                    return True, "Applied patch hunks but commit failed or nothing to commit: " + commit_out
            except Exception as e:
                return True, f"Applied patch hunks but commit failed with exception: {e}"

        # 2) Fallback: write diff to temp and attempt git apply
        tmp = None
        try:
            tmp_fd, tmp = tempfile.mkstemp(prefix="llm_patch_", suffix=".diff")
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(diff_text)
            # Try git apply --whitespace=fix --index
            rc, out, err = self._run(f"git apply --whitespace=fix --index {shlex.quote(tmp)}")
            if rc == 0:
                ok_commit, commit_out = self.commit_all(commit_message)
                if ok_commit:
                    return True, "git apply succeeded and committed: " + commit_out
                return True, "git apply succeeded but commit failed or nothing to commit: " + commit_out
            # Try plain patch tool
            rc2, out2, err2 = self._run(f"patch -p0 < {shlex.quote(tmp)}")
            if rc2 == 0:
                ok_commit, commit_out = self.commit_all(commit_message)
                if ok_commit:
                    return True, "patch succeeded and committed: " + commit_out
                return True, "patch succeeded but commit failed or nothing to commit: " + commit_out

            # both failed
            combined = f"git apply rc={rc} out={out} err={err}\npatch rc={rc2} out={out2} err={err2}\npatcher_report={report}"
            return False, combined
        except Exception as e:
            return False, f"Applying patch failed: {e}\npatcher_report={report}"
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    # ---- parsing helpers for diffs in text (from LLM) ----
    @staticmethod
    def extract_diff(text: str) -> Optional[str]:
        """
        Try to extract unified diff from LLM output. Accepts plain diff or fenced code block with 'diff' or 'patch'.
        """
        if not text:
            return None
        # If starts with 'diff --git' or '*** Begin Patch' or '@@ ', assume diff
        if text.strip().startswith(("diff --git", "*** Begin Patch", "@@ ")):
            return text
        # Find fenced code blocks ```diff ... ```
        m = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # fallback: try to find first contiguous block that looks like a diff (has @@ or diff --git)
        m2 = re.search(r"(diff --git[\s\S]*?$)|(@@[\s\S]*?@@)", text, flags=re.MULTILINE)
        if m2:
            return m2.group(0)
        return None

    # ---- high-level orchestration ----
    def validate(self, paths: Optional[str] = None, use_pylint: bool = False, pytest_args: Optional[str] = None) -> ValidatorResult:
        """Run linters and tests and return structured result."""
        l_ok, l_out = self.run_linters(paths=paths, use_pylint=use_pylint)
        t_ok, t_out = self.run_tests(pytest_args=pytest_args)
        ok = l_ok and t_ok
        msg = "OK" if ok else "Issues found"
        return ValidatorResult(ok=ok, linter_ok=l_ok, tests_ok=t_ok, linter_output=l_out or "", tests_output=t_out or "", message=msg)

    def validate_and_fix(
        self,
        *,
        llm=None,
        paths: Optional[str] = None,
        use_pylint: bool = False,
        pytest_args: Optional[str] = None,
        auto_rollback: bool = True,
        attempt_fix: bool = True,
        llm_patch_prompt_template: Optional[str] = None,
        commit_message: str = "LLM-applied-fix",
        max_fix_attempts: int = 2,
    ) -> ValidatorResult:
        """
        High-level flow:
         - Run current linters/tests
         - If ok -> return success
         - If not ok and auto_rollback True -> compute current HEAD and create a safety branch (HEAD@{0} saved)
         - If llm provided and attempt_fix True -> ask llm to generate a patch; extract diff and apply via git apply
             * After applying, re-run validation; if valid -> return success
             * If still failing and attempts remain -> request another patch
         - If all fails -> optionally rollback to original HEAD (if auto_rollback True)
        """
        logger.info("Starting validate_and_fix (paths=%s)", paths or self.repo_path)
        original_head = self.git_head()
        logger.debug("Original HEAD: %s", original_head)

        res = self.validate(paths=paths, use_pylint=use_pylint, pytest_args=pytest_args)
        if res.ok:
            logger.info("Validation succeeded on first try.")
            return res

        # Validation failed
        logger.warning("Validation failed: linters_ok=%s tests_ok=%s", res.linter_ok, res.tests_ok)

        # Safety: create a branch to preserve current state (if git available)
        timestamp = int(time.time())
        safety_branch = f"ci-safety-{timestamp}"
        try:
            ok_branch, msg_branch = self.create_branch(safety_branch)
            if ok_branch:
                logger.info("Created backup branch %s to preserve current work.", safety_branch)
            else:
                logger.debug("Could not create safety branch: %s", msg_branch)
        except Exception as e:
            logger.debug("Creating safety branch failed: %s", e)

        applied_patch_text = None
        # Try to request fixes via LLM
        if llm and attempt_fix:
            attempts = 0
            while attempts < max_fix_attempts:
                attempts += 1
                logger.info("Requesting patch from LLM (attempt %s/%s)", attempts, max_fix_attempts)
                # Construct prompt summarizing linter and test failures
                prompt_parts = []
                prompt_parts.append("You are an assistant that produces a unified diff patch to fix Python repository issues.")
                prompt_parts.append("Repository path: " + self.repo_path)
                prompt_parts.append("\nLinter output (flake8/pylint):\n" + (res.linter_output or "(no linter output)"))
                prompt_parts.append("\nTest output (pytest):\n" + (res.tests_output or "(no tests output)"))
                if llm_patch_prompt_template:
                    # allow custom template with placeholders
                    prompt = llm_patch_prompt_template.replace("{{LINTER_OUTPUT}}", res.linter_output).replace("{{TEST_OUTPUT}}", res.tests_output).replace("{{REPO_PATH}}", self.repo_path)
                else:
                    prompt = "\n\n".join(prompt_parts) + "\n\nPlease return a unified diff (git-style) that fixes the issues. Wrap the diff in a ```diff ... ``` code block if possible."

                try:
                    llm_resp = llm.generate(prompt, model=getattr(llm, "model", None))
                    llm_text = llm_resp.get("text") if isinstance(llm_resp, dict) else str(llm_resp)
                except Exception as e:
                    logger.exception("LLM call for patch failed: %s", e)
                    llm_text = ""

                diff = self.extract_diff(llm_text or "")
                if not diff:
                    logger.warning("LLM did not return a recognizable diff; returning failure and LLM output.")
                    # attach LLM output to result for debugging
                    res.message = "LLM did not return a valid diff"
                    res.linter_output += "\n\nLLM_OUTPUT:\n" + (llm_text or "")
                    break

                # Try applying diff
                ok_apply, apply_out = self.apply_diff_patch(diff, commit_message=commit_message)
                if ok_apply:
                    logger.info("Applied patch from LLM. Re-running validation...")
                    applied_patch_text = diff
                    # re-run validation
                    res_after = self.validate(paths=paths, use_pylint=use_pylint, pytest_args=pytest_args)
                    if res_after.ok:
                        res_after.applied_patch = diff
                        res_after.message = "Applied LLM patch and validation passed."
                        return res_after
                    else:
                        logger.warning("Patch applied but validation still failing. tests_ok=%s linter_ok=%s", res_after.tests_ok, res_after.linter_ok)
                        # update res to latest outputs and try again if attempts remain
                        res = res_after
                        continue
                else:
                    logger.warning("Failed to apply LLM-provided diff: %s", apply_out)
                    res.message = "Failed to apply LLM diff: " + apply_out
                    res.linter_output += "\n\nLLM_DIFF_APPLY_OUTPUT:\n" + apply_out
                    # do not retry if apply failed
                    break

        # If we reach here, either no llm, or fixes didn't work
        if auto_rollback and original_head:
            logger.info("Attempting rollback to original HEAD %s", original_head)
            ok_rb, rb_out = self.rollback_to(original_head)
            if ok_rb:
                logger.info("Rollback succeeded.")
            else:
                logger.warning("Rollback failed: %s", rb_out)
        res.applied_patch = applied_patch_text
        return res
