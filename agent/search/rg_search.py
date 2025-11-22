# agent/search/rg_search.py
"""
Ripgrep (rg) wrapper and safe fallback search utilities.

Provides:
- rg_search(query, repo_root=".", max_results=50, file_glob=None, flags=None)
    -> list of match dicts: {path, line, col, text, match_text, context_lines, type}

- find_symbol(symbol, repo_root=".", languages=None, max_results=30)
    -> prioritized list of likely definition locations (uses language-aware regexes
       and exact search heuristics). Good for "where is foo defined?" flows.

Behavior:
- If ripgrep (rg) is installed and reachable on PATH, this module will use it
  (fast, robust). It invokes `rg --json` and parses output.
- If rg is not available, it falls back to a Python-based search using os.walk
  and regex matching (slower but doesn't require external deps).
- The functions always return a list of structured hits. They will never raise
  on common user errors; instead they return an empty list and log warnings.

Note:
- This wrapper is intentionally conservative and does not execute shell input
  directly without shlex quoting where appropriate.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Pattern, Tuple

# prefer the project's logger if available
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
class RGHit:
    path: str
    line: int
    col: Optional[int]
    text: str
    match_text: str
    type: str  # 'match' or 'context' or other rg event types
    # optional surrounding context lines (tuple of (before, after))
    context_before: Optional[List[str]] = None
    context_after: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# -------------------------
# Internal helpers
# -------------------------
def _rg_available() -> bool:
    return shutil.which("rg") is not None


def _run_cmd(cmd: Iterable[str], cwd: str = ".", timeout: int = 30) -> Tuple[int, str, str]:
    """
    Run a command list and return (rc, stdout, stderr).
    """
    try:
        proc = subprocess.run(list(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        return proc.returncode, out, err
    except subprocess.TimeoutExpired as e:
        logger.warning("Command timeout: %s", e)
        return 124, "", f"Timeout: {e}"
    except Exception as e:
        logger.exception("Command execution failed: %s", e)
        return 1, "", str(e)


def _parse_rg_json_stream(json_text: str) -> List[RGHit]:
    """
    Parse ripgrep --json output (may contain multiple JSON objects separated by newlines).
    We look for objects with "type": "match" and extract useful fields.
    """
    hits: List[RGHit] = []
    for line in json_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            # Some versions of rg may output non-json lines (warnings) — ignore
            continue
        typ = obj.get("type")
        if typ == "match":
            data = obj.get("data", {})
            path_dict = data.get("path", {})
            path_text = path_dict.get("text") or path_dict.get("path", "")
            for sub in data.get("submatches", []):
                match_text = sub.get("match", {}).get("text", "")
                # position info - handle case where start might be int or dict
                start_info = sub.get("start", {})
                if isinstance(start_info, dict):
                    line_num = start_info.get("line") or data.get("line_number") or start_info.get("line_number")
                    col = start_info.get("column") or None
                else:
                    # start is an int (line number directly)
                    line_num = start_info if isinstance(start_info, int) else data.get("line_number")
                    col = None
                # the full line text is provided in data.get("lines", {}).get("text")
                line_text = (data.get("lines") or {}).get("text") or ""
                # context is not provided by rg json by default; we'll leave None
                hit = RGHit(path=path_text, line=int(line_num) if line_num else 0, col=int(col) if col else None, text=line_text.rstrip("\n"), match_text=match_text, type="match")
                hits.append(hit)
        # We could handle other types if useful (begin, end, summary) but skip for now
    return hits


def _pygrep_search(query_regex: Pattern, repo_root: str, file_glob: Optional[str] = None, max_results: int = 100) -> List[RGHit]:
    """
    Pure-Python fallback that searches files under repo_root for regex matches.
    It attempts to skip typical binary files by extension and skip common VCS dirs.
    """
    hits: List[RGHit] = []
    repo_root = os.path.abspath(repo_root or ".")
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".hg"}
    # optional file_glob not implemented as glob; just use simple suffix match if provided
    file_suffix = None
    if file_glob and "*" not in file_glob:
        # if user passed something like "*.py", handle suffix
        if file_glob.startswith("*."):
            file_suffix = file_glob[1:]
        else:
            # if it's a path, we can match contains
            file_suffix = file_glob

    compiled = query_regex
    results = 0
    for root, dirs, files in os.walk(repo_root):
        # prune skip dirs
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if results >= max_results:
                return hits
            if file_suffix and not fname.endswith(file_suffix):
                continue
            fpath = os.path.join(root, fname)
            # skip large/binary by simple heuristic: size > 1MB => skip
            try:
                if os.path.getsize(fpath) > 2 * 1024 * 1024:
                    continue
            except Exception:
                pass
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    for i, ln in enumerate(fh, start=1):
                        if compiled.search(ln):
                            # extract match group 0
                            m = compiled.search(ln)
                            match_text = m.group(0) if m else ""
                            hit = RGHit(path=os.path.relpath(fpath, repo_root), line=i, col=(m.start() + 1 if m else None), text=ln.rstrip("\n"), match_text=match_text, type="match")
                            hits.append(hit)
                            results += 1
                            if results >= max_results:
                                return hits
            except Exception:
                continue
    return hits


# -------------------------
# Public API
# -------------------------
def rg_search(query: str, repo_root: str = ".", *, max_results: int = 50, file_glob: Optional[str] = None, flags: Optional[List[str]] = None) -> List[Dict]:
    """
    Search for `query` in files under repo_root.

    - If ripgrep is available, uses `rg --json` for robust, fast search.
    - Otherwise falls back to a Python-based file walker + regex.

    Parameters:
        query: literal string or regex pattern (interpreted as raw regex).
        repo_root: folder to search (defaults to current dir).
        max_results: limit number of results returned.
        file_glob: optional file filter like '*.py' (simple handling).
        flags: optional list of extra rg flags (e.g., ["-S"] for smart-case).

    Returns:
        list of dicts derived from RGHit.to_dict()
    """
    repo_root = repo_root or "."
    flags = flags or []
    # sanitize repo_root
    if not os.path.isdir(repo_root):
        logger.warning("rg_search: repo_root not a directory: %s", repo_root)
        return []

    if _rg_available():
        # Build rg command: use --json for stable parsing
        cmd = ["rg", "--json", "--line-number", "--hidden", "--no-ignore-vcs", "--no-ignore"]
        # apply user flags
        if flags:
            cmd.extend(flags)
        # file filter
        if file_glob:
            cmd.extend(["-g", file_glob])
        # query: pass raw (rg interprets as regex)
        cmd.append(query)
        rc, out, err = _run_cmd(cmd, cwd=repo_root, timeout=30)
        if rc not in (0, 1):  # rg returns 0 if match found, 1 if no match, >1 on error
            logger.debug("rg returned rc=%s stderr=%s", rc, err.strip())
            # fallback to python grep
        try:
            hits = _parse_rg_json_stream(out)
            # trim to max_results
            return [h.to_dict() for h in hits[:max_results]]
        except Exception as e:
            logger.exception("Failed to parse rg json output: %s", e)
            # fallback to python grep below

    # fallback
    try:
        regex = re.compile(query, flags=re.MULTILINE)
    except Exception:
        # treat query as escaped literal
        regex = re.compile(re.escape(query), flags=re.MULTILINE)
    hits = _pygrep_search(regex, repo_root, file_glob=file_glob, max_results=max_results)
    return [h.to_dict() for h in hits[:max_results]]


# -------------------------
# Symbol-finding helpers
# -------------------------
# language -> list of definition regex patterns (pattern should contain a capturing group for the symbol name)
_LANGUAGE_SYMBOL_PATTERNS = {
    "python": [
        r"^\s*def\s+{sym}\b",
        r"^\s*class\s+{sym}\b",
        r"^\s*{sym}\s*=",
    ],
    "javascript": [
        r"^\s*function\s+{sym}\b",
        r"^\s*(?:const|let|var)\s+{sym}\s*=",
        r"^\s*export\s+function\s+{sym}\b",
        r"^\s*class\s+{sym}\b",
    ],
    "java": [
        r"^\s*(public|private|protected)?\s*(class|interface)\s+{sym}\b",
        r"^\s*(public|private|protected)?.*\s+{sym}\s*\(",
    ],
    "go": [
        r"^\s*func\s+{sym}\b",
    ],
    "c": [
        r"^\s*[A-Za-z0-9_].*\s+{sym}\s*\(",
    ],
    # fallback generic patterns
    "generic": [
        r"^\s*def\s+{sym}\b",
        r"^\s*class\s+{sym}\b",
        r"^\s*(?:function|const|let|var)\s+{sym}\b",
        r"^\s*{sym}\s*=",
    ],
}


def _build_symbol_regex(symbol: str, languages: Optional[Iterable[str]] = None) -> Pattern:
    """
    Create a compiled regex that tries common definition patterns for the symbol across provided languages.
    """
    languages = list(languages) if languages else ["python", "javascript", "generic"]
    parts = []
    esc = re.escape(symbol)
    for lang in languages:
        pats = _LANGUAGE_SYMBOL_PATTERNS.get(lang, [])
        for p in pats:
            try:
                parts.append(p.format(sym=esc))
            except Exception:
                continue
    if not parts:
        parts = [r"\b" + esc + r"\b"]
    combined = "|".join(f"(?:{p})" for p in parts)
    return re.compile(combined, flags=re.MULTILINE)


def find_symbol(symbol: str, repo_root: str = ".", *, languages: Optional[Iterable[str]] = None, max_results: int = 30) -> List[Dict]:
    """
    Locate likely definition(s) of a symbol (function/class/variable) in the repo.

    Strategy:
      1. If rg available, run targeted regex search for definition patterns.
      2. If rg not available or parsing fails, run Python fallback search.

    Returns:
      list of hits (dicts) ordered by likely relevance.
    """
    if not symbol or not symbol.strip():
        return []

    # heuristics: narrow the search to likely file globs based on symbol name (e.g., .py)
    repo_root = repo_root or "."
    langs = list(languages) if languages else ["python", "javascript", "generic"]
    regex = _build_symbol_regex(symbol, languages=langs)

    # If rg available, run it with the regex string (we want the raw regex)
    try:
        if _rg_available():
            rg_pattern = regex.pattern
            cmd = ["rg", "--json", "--line-number", "--hidden", "--no-ignore-vcs", "--no-ignore", "-S", rg_pattern]
            rc, out, err = _run_cmd(cmd, cwd=repo_root, timeout=20)
            if rc not in (0, 1):
                logger.debug("rg returned rc=%s when finding symbol %s", rc, symbol)
            hits = _parse_rg_json_stream(out)
            if hits:
                # prefer definitions (lines that look like def/class); we already used patterns so good
                return [h.to_dict() for h in hits[:max_results]]
    except Exception:
        logger.exception("rg symbol search failed; falling back to python search")

    # fallback: pure python grep with compiled regex
    py_hits = _pygrep_search(regex, repo_root, max_results=max_results)
    return [h.to_dict() for h in py_hits[:max_results]]


# -------------------------
# Convenience CLI for local debugging
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog="rg_search", description="Ripgrep wrapper / fallback search")
    sub = parser.add_subparsers(dest="cmd")
    p_search = sub.add_parser("search", help="search for regex or text")
    p_search.add_argument("query", help="regex or text to search for")
    p_search.add_argument("--repo", default=".", help="repo root")
    p_search.add_argument("--max", type=int, default=50)
    p_search.add_argument("--glob", default=None)
    p_sym = sub.add_parser("symbol", help="find symbol definition")
    p_sym.add_argument("symbol", help="symbol name")
    p_sym.add_argument("--repo", default=".", help="repo root")

    args = parser.parse_args()
    if args.cmd == "search":
        res = rg_search(args.query, repo_root=args.repo, max_results=args.max, file_glob=args.glob)
        print(json.dumps(res, indent=2))
    elif args.cmd == "symbol":
        res = find_symbol(args.symbol, repo_root=args.repo)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()
