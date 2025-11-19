# agent/utils/file_ref.py
"""
File reference resolver.

Supports inline file references using a leading '@' token.

Reference grammar (friendly):
  @path/to/file.ext
  @./relative/path.py
  @../up/one/level.txt
  @path/to/file.py:10        -> line 10
  @path/to/file.py:10-20     -> lines 10..20 (inclusive)

Behavior:
- Paths are resolved relative to repo_root (default '.').
- If the file is missing, resolves to a short error string (so prompts remain valid).
- Truncates very large files by `max_chars` when expanding into prompt to avoid huge context.
- The expansion wraps content in a fenced block:
    <FILE: path/to/file.py lines 10-20>
    ```file
    <file contents...>
    ```
- Use `expand_file_refs()` before sending prompts to the planner/LLM so the LLM can "see" referenced files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Simple logger fallback
try:
    from agent.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Regex to capture file references:
# @path/to/file.py OR @./file.py OR @../file.txt
# optional :start or :start-end (1-indexed lines)
_FILE_REF_RE = re.compile(
    r"""@(?P<path>(?:\./|\.\./|/)?[A-Za-z0-9_\-./\\]+?)(?::(?P<start>\d+)(?:-(?P<end>\d+))?)?(?=$|\s)"""
)

@dataclass
class FileRef:
    raw: str           # original matched token e.g. "@src/foo.py:10-20"
    path: str          # path string as written
    start: Optional[int] = None  # 1-based line start (inclusive)
    end: Optional[int] = None    # 1-based line end (inclusive)


def find_file_refs(text: str) -> List[FileRef]:
    """Return list of FileRef objects found in text, in order of appearance."""
    out: List[FileRef] = []
    for m in _FILE_REF_RE.finditer(text):
        path = m.group("path")
        start_s = m.group("start")
        end_s = m.group("end")
        start = int(start_s) if start_s else None
        end = int(end_s) if end_s else None
        out.append(FileRef(raw=m.group(0), path=path, start=start, end=end))
    return out


def _read_file_segment(abs_path: str, start: Optional[int], end: Optional[int]) -> Tuple[bool, str]:
    """
    Return (ok, content_or_error).
    Lines are 1-indexed. If start/end are None, return entire file (but may be truncated later).
    """
    if not os.path.exists(abs_path):
        return False, f"[MISSING FILE: {abs_path}]"
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except Exception as e:
        return False, f"[ERROR READING FILE {abs_path}: {e}]"
    # apply line slicing
    if start is None and end is None:
        # return full content joined with \n
        return True, "\n".join(lines)
    # convert to 0-index slice; clamp bounds
    n = len(lines)
    s = max(1, start or 1)
    e = min(n, end or n)
    if s > n:
        return False, f"[LINE RANGE OUT OF BOUNDS: file has {n} lines, requested start={s}]"
    segment = lines[s - 1 : e]
    return True, "\n".join(segment)


def resolve_file_ref(ref: FileRef, repo_root: str = ".") -> Tuple[bool, str]:
    """
    Resolve a FileRef to its content relative to repo_root.
    Returns (ok, content_or_error_string).
    """
    # normalize path: remove leading '/' only if user provided absolute? We'll support both.
    path = ref.path
    # Prevent path traversal outside repo_root if path is absolute: join and normpath and ensure within repo_root.
    candidate = os.path.normpath(os.path.join(repo_root, path))
    # Ensure candidate is inside repo_root (avoid leaking system files)
    repo_root_abs = os.path.abspath(repo_root)
    cand_abs = os.path.abspath(candidate)
    if not (cand_abs == repo_root_abs or cand_abs.startswith(repo_root_abs + os.sep)):
        # Path appears outside repo root — deny for safety
        return False, f"[REFERS OUTSIDE REPO: {ref.raw}]"
    return _read_file_segment(cand_abs, ref.start, ref.end)


def _truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    """Return (possibly_truncated_text, was_truncated)."""
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 200] + "\n\n...[TRUNCATED]...", True
    return text, False


def expand_file_refs(text: str, repo_root: str = ".", max_chars: int = 4000) -> str:
    """
    Replace all @file references in text with an explicit file block containing file excerpt.

    Example replacement:
      "@src/foo.py:10-20" -> "<FILE: src/foo.py lines 10-20>\n```file\n<contents>\n```\n"

    max_chars: max characters for each expanded file excerpt (prevents huge prompts).
    """
    refs = find_file_refs(text)
    if not refs:
        return text

    # Build replacement map preserving order — avoid overlapping replacements by using re.sub with a function
    def _repl(match: re.Match) -> str:
        path = match.group("path")
        start_s = match.group("start")
        end_s = match.group("end")
        ref = FileRef(raw=match.group(0), path=path, start=(int(start_s) if start_s else None), end=(int(end_s) if end_s else None))
        ok, content = resolve_file_ref(ref, repo_root=repo_root)
        if not ok:
            # Insert an inline marker so planner/LLM knows the file was referenced but missing
            return f"<FILE_REF: {ref.path} lines {ref.start or 1}-{ref.end or 'EOF'}>\\n```\n{content}\n```\\n"
        # truncate if needed
        excerpt, truncated = _truncate(content, max_chars)
        header = f"<FILE: {ref.path}"
        if ref.start or ref.end:
            header += f" lines {ref.start or 1}-{ref.end or 'EOF'}"
        header += ">"
        note = " [TRUNCATED]" if truncated else ""
        return f"{header}{note}\n```file\n{excerpt}\n```\n"

    # use sub to handle repeated patterns safely
    expanded = _FILE_REF_RE.sub(_repl, text)
    return expanded
