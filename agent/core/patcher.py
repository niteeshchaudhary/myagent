# agent/core/patcher.py
"""
Simple unified-diff patch applier.

Features
- Parses unified diff strings (git-style) and applies hunks in-place to files.
- Applies hunks conservatively by matching hunk context in the target file.
- Atomic write: writes to a temp file then replaces original.
- Returns (ok: bool, report: dict) with per-file results and helpful messages.

Limitations
- Not as feature-complete as GNU patch/git-apply.
- If a hunk context can't be matched, the function will fail for that file (safe).
- Always keep a safety backup (this utility does not create git commits).
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Dict, List, Tuple

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
    logger.setLevel(logging.DEBUG)

# Change tracking
try:
    from agent.utils.change_tracker import get_change_tracker
    _has_change_tracker = True
except Exception:
    _has_change_tracker = False
    logger.debug("Change tracker not available")


# --- Parsing helpers ---
RE_FILE_HEADER = re.compile(r'^(?:diff --git a/)?(?P<a>[^ ]+)\s+(?:b/)?(?P<b>.+)$')
RE_OLD = re.compile(r'^---\s+(?P<path>.+)$')
RE_NEW = re.compile(r'^\+\+\+\s+(?P<path>.+)$')
RE_HUNK = re.compile(r'^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@')

def _split_lines(text: str) -> List[str]:
    # keep line endings trimmed but preserve '\n' context for matching
    return text.splitlines(keepends=False)

def _apply_single_hunk(orig_lines: List[str], hunk_lines: List[str], hunk_ctx_before: List[str], hunk_ctx_after: List[str]) -> Tuple[bool, List[str], str]:
    """
    Apply a single hunk to orig_lines.
    - hunk_lines: lines beginning with ' ', '+' or '-'.
    - hunk_ctx_before/after are sequences of context lines used to locate hunk position (optional).
    Returns (ok, new_lines, message)
    """
    # Build the 'target' replacement from hunk_lines (skip '-' lines)
    new_segment = [ln[1:] for ln in hunk_lines if ln.startswith('+') or ln.startswith(' ')]
    # Build the old segment pattern to find: lines marked ' ' or '-' (we use ' ' and '-' combined)
    old_segment = [ln[1:] for ln in hunk_lines if ln.startswith(' ') or ln.startswith('-')]

    # Strategy: locate a place in orig_lines where old_segment occurs (exact match)
    # Prefer to search for full old_segment; if not found, try matching using context before/after.
    def find_subsequence(haystack: List[str], needle: List[str]) -> int:
        if not needle:
            return 0
        n = len(needle)
        for i in range(len(haystack) - n + 1):
            match = True
            for j in range(n):
                if haystack[i + j] != needle[j]:
                    match = False
                    break
            if match:
                return i
        return -1

    idx = find_subsequence(orig_lines, old_segment)
    if idx >= 0:
        # Replace old_segment at idx with new_segment
        new_lines = orig_lines[:idx] + new_segment + orig_lines[idx + len(old_segment):]
        return True, new_lines, f"Applied at index {idx}"
    # If not found, try context-based locate
    if hunk_ctx_before or hunk_ctx_after:
        # find an index where context before is found and context after follows appropriately
        for i in range(len(orig_lines)):
            ok_before = True
            if hunk_ctx_before:
                b = hunk_ctx_before
                if i - len(b) < 0:
                    ok_before = False
                else:
                    ok_before = orig_lines[i - len(b):i] == b
            # check after location where old_segment would be
            if not ok_before:
                continue
            # check if old_segment matches from i to i+len(old_segment)
            if orig_lines[i:i + len(old_segment)] == old_segment:
                new_lines = orig_lines[:i] + new_segment + orig_lines[i + len(old_segment):]
                return True, new_lines, f"Applied at context-located index {i}"
        # fallback: try fuzzy search: search for first line of old_segment
        if old_segment:
            first = old_segment[0]
            for i, line in enumerate(orig_lines):
                if line == first:
                    # try to patch optimistically
                    seg_len = len(old_segment)
                    if orig_lines[i:i + seg_len] == old_segment:
                        new_lines = orig_lines[:i] + new_segment + orig_lines[i + seg_len:]
                        return True, new_lines, f"Applied optimistic at index {i}"
    return False, orig_lines, "Could not find matching old segment"

def _parse_unified_diff(diff_text: str) -> List[Dict]:
    """
    Parse a unified diff into a list of file change dicts:
    { "old_path": str, "new_path": str, "hunks": [ { "header": str, "lines": [...] } ] }
    """
    lines = _split_lines(diff_text)
    i = 0
    parsed = []
    current = None

    while i < len(lines):
        ln = lines[i]
        # file header lines: '--- a/path' and '+++ b/path' or diff --git a/path b/path
        m_old = RE_OLD.match(ln)
        if m_old:
            old_path = m_old.group("path").strip()
            # next line should be +++
            i += 1
            if i < len(lines):
                m_new = RE_NEW.match(lines[i])
                new_path = m_new.group("path").strip() if m_new else None
            else:
                new_path = None
            current = {"old_path": old_path, "new_path": new_path or old_path, "hunks": []}
            parsed.append(current)
            i += 1
            continue

        # sometimes diffs include diff --git header with a/ b/
        m_file = RE_FILE_HEADER.match(ln)
        if m_file:
            # advance and look for --- +++
            # try to detect following --- +++ pair
            # create a tentative current if none
            if current is None:
                current = {"old_path": None, "new_path": None, "hunks": []}
                parsed.append(current)
            i += 1
            continue

        # hunk header
        m_h = RE_HUNK.match(ln)
        if m_h:
            # collect hunk lines until next @@ or next file header
            header = ln
            hunk_lines = []
            i += 1
            while i < len(lines) and not RE_HUNK.match(lines[i]) and not RE_OLD.match(lines[i]) and not RE_NEW.match(lines[i]) and not RE_FILE_HEADER.match(lines[i]):
                # include even context lines and +/- lines
                if lines[i].startswith(("+", "-", " ")):
                    hunk_lines.append(lines[i])
                else:
                    # lines like \ No newline at end of file -> treat as context
                    hunk_lines.append(lines[i])
                i += 1
            if current is None:
                # create anonymous file entry (rare)
                current = {"old_path": None, "new_path": None, "hunks": []}
                parsed.append(current)
            current["hunks"].append({"header": header, "lines": hunk_lines})
            continue

        # otherwise proceed
        i += 1

    return parsed


# --- Public API ---
def apply_unified_diff(diff_text: str, repo_root: str = ".", dry_run: bool = False) -> Tuple[bool, Dict]:
    """
    Attempt to apply unified diff to files under repo_root.
    Returns (ok, report) where report contains per-file status.

    If dry_run True, do not write files; only report whether application would succeed.
    """
    parsed = _parse_unified_diff(diff_text)
    if not parsed:
        return False, {"error": "No file changes parsed from diff."}

    report = {"files": []}
    for file_change in parsed:
        old_path = file_change.get("old_path") or file_change.get("new_path")
        new_path = file_change.get("new_path") or file_change.get("old_path")
        # Normalize paths: remove a/ or b/ prefixes if present
        def norm(p):
            if not p:
                return None
            return p[2:] if p.startswith(("a/", "b/")) else p
        oldp = norm(old_path) or norm(new_path)
        newp = norm(new_path) or norm(old_path)

        target_path = os.path.join(repo_root, newp) if newp else os.path.join(repo_root, oldp)
        file_report = {"file": target_path, "applied": False, "hunks": []}

        if not os.path.exists(target_path):
            file_report["error"] = "Target file does not exist"
            report["files"].append(file_report)
            continue

        # Track changes: snapshot before modification
        if _has_change_tracker and not dry_run:
            try:
                tracker = get_change_tracker()
                tracker.snapshot_file(target_path)
            except Exception:
                pass

        # Read original content
        with open(target_path, "r", encoding="utf-8", errors="replace") as fh:
            orig_lines = fh.read().splitlines(keepends=False)

        # Apply hunks sequentially
        working_lines = orig_lines
        all_ok = True
        for h in file_change.get("hunks", []):
            hlines = h.get("lines", [])
            # determine context before/after from the hunk: collect up to 3 context lines before/after
            ctx_before = [ln[1:] for ln in hlines if ln.startswith(" ")][:3]
            ctx_after = [ln[1:] for ln in hlines if ln.startswith(" ")][-3:]
            ok, new_working, msg = _apply_single_hunk(working_lines, hlines, ctx_before, ctx_after)
            file_report["hunks"].append({"ok": ok, "msg": msg})
            if not ok:
                all_ok = False
                break
            working_lines = new_working

        if all_ok:
            file_report["applied"] = True
            if not dry_run:
                # write atomic
                dirpath = os.path.dirname(target_path)
                os.makedirs(dirpath or ".", exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(prefix=".patchtmp_", dir=dirpath)
                try:
                    new_content = "\n".join(working_lines) + ("\n" if working_lines and not working_lines[-1].endswith("\n") else "")
                    with os.fdopen(fd, "w", encoding="utf-8") as outfh:
                        outfh.write(new_content)
                    os.replace(tmp_path, target_path)
                    
                    # Track changes: record after modification
                    if _has_change_tracker:
                        try:
                            tracker = get_change_tracker()
                            tracker.record_change(target_path, new_content=new_content)
                        except Exception:
                            pass
                except Exception as e:
                    file_report["applied"] = False
                    file_report["error"] = f"Writing failed: {e}"
                    # cleanup tmp
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
        else:
            file_report["applied"] = False

        report["files"].append(file_report)

    # overall ok only if all file appliers succeeded
    overall_ok = all(f.get("applied") for f in report["files"])
    report["overall_ok"] = overall_ok
    return overall_ok, report
