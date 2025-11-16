# agent/search/tags_client.py
"""
Tags client using Universal Ctags (fallback: tags file parser).

Provides:
- TagsClient(repo_root=".") class that can:
    - build_tags(recurse=True)  -> run ctags to generate tags (in-memory)
    - load_tags_from_file(path) -> parse an existing tags file (traditional or JSON lines)
    - find_definitions(symbol, kinds=None, max_results=20) -> list of probable definitions
    - find_references(symbol, max_results=50) -> list of tag hits where symbol appears
    - list_symbols() -> all symbol names indexed
    - get_tag_entries(name) -> raw entries for a symbol

Notes:
- This attempts to use `ctags` (universal-ctags) with JSON output if available:
    ctags -R --fields=+n --extras=+q --output-format=json -f - .
  If that fails, it falls back to parsing a plain 'tags' file or returns best-effort results.
- The outputs are normalized dicts with keys:
    { "name", "path", "line", "pattern", "kind", "scope", "language", "raw" }
- The module is defensive: missing ctags or unexpected output will not raise (it logs and returns empty lists).
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
from typing import Dict, Iterable, List, Optional, Tuple

# prefer project logger if available
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
class TagEntry:
    name: str
    path: str
    line: Optional[int] = None
    pattern: Optional[str] = None
    kind: Optional[str] = None
    scope: Optional[str] = None
    language: Optional[str] = None
    raw: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class TagsClient:
    """
    A simple tags client backed by universal-ctags (preferred) or tags file parse (fallback).

    Usage:
        tc = TagsClient(repo_root=".")
        tc.build_tags()  # runs ctags and populates in-memory index
        defs = tc.find_definitions("my_function")
    """

    def __init__(self, repo_root: str = "."):
        self.repo_root = os.path.abspath(repo_root or ".")
        self._entries_by_name: Dict[str, List[TagEntry]] = {}
        self._all_entries: List[TagEntry] = []
        self._ctags_cmd = shutil.which("ctags") or shutil.which("uctags") or shutil.which("universal-ctags")
        self._last_built_at: Optional[float] = None

    # -------------------------
    # Low-level: run ctags & parse
    # -------------------------
    def _run_cmd(self, cmd: Iterable[str], cwd: Optional[str] = None, timeout: int = 60) -> Tuple[int, str, str]:
        try:
            proc = subprocess.run(list(cmd), cwd=cwd or self.repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            out = proc.stdout.decode("utf-8", errors="replace")
            err = proc.stderr.decode("utf-8", errors="replace")
            return proc.returncode, out, err
        except subprocess.TimeoutExpired as e:
            logger.warning("ctags command timeout: %s", e)
            return 124, "", f"Timeout: {e}"
        except Exception as e:
            logger.exception("ctags command failed: %s", e)
            return 1, "", str(e)

    def build_tags(self, recurse: bool = True) -> int:
        """
        Run ctags over repo_root and load tags into memory.

        Returns number of entries loaded.
        """
        if not self._ctags_cmd:
            logger.warning("ctags executable not found on PATH. Install universal-ctags for best results.")
            # attempt to load an existing tags file if present
            tags_path = os.path.join(self.repo_root, "tags")
            if os.path.exists(tags_path):
                self.load_tags_from_file(tags_path)
                return len(self._all_entries)
            return 0

        # Try to use JSON output (universal-ctags)
        cmd = [self._ctags_cmd, "-R", "--fields=+nS", "--extras=+q", "--output-format=json", "-f", "-", "."]
        rc, out, err = self._run_cmd(cmd)
        if rc == 0 and out:
            try:
                entries = self._parse_ctags_json_lines(out)
                self._populate(entries)
                self._last_built_at = os.path.getmtime(self.repo_root) if os.path.exists(self.repo_root) else None
                return len(self._all_entries)
            except Exception as e:
                logger.exception("Failed to parse ctags JSON output: %s", e)

        # Fallback: generate a tags file and parse it
        try:
            # write to temp tags file to avoid overwriting user's tags
            tags_tmp = os.path.join(self.repo_root, ".ctags_tmp.tags")
            cmd2 = [self._ctags_cmd, "-R", "-f", tags_tmp, "."]
            rc2, out2, err2 = self._run_cmd(cmd2)
            if rc2 == 0 and os.path.exists(tags_tmp):
                self.load_tags_from_file(tags_tmp)
                try:
                    os.remove(tags_tmp)
                except Exception:
                    pass
                self._last_built_at = os.path.getmtime(tags_tmp)
                return len(self._all_entries)
            else:
                logger.warning("ctags fallback returned rc=%s stderr=%s", rc2, err2)
        except Exception as e:
            logger.exception("ctags fallback failed: %s", e)

        # nothing worked
        logger.warning("build_tags: unable to produce tags.")
        return 0

    def _parse_ctags_json_lines(self, text: str) -> List[Dict]:
        """
        Universal-ctags JSON output typically writes one JSON object per line.
        Each object contains 'name' and 'path' and 'pattern' or 'line'.
        We'll collect relevant ones.
        """
        entries: List[Dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # ignore invalid json lines
                continue
            # a JSON entry with "kind" and "name" typically represents a tag
            if isinstance(obj, dict) and "name" in obj and "path" in obj:
                entries.append(obj)
        return entries

    # -------------------------
    # Fallback parser for traditional 'tags' format
    # -------------------------
    _TAGS_LINE_RE = re.compile(r"^(?P<name>[^\t]+)\t(?P<file>[^\t]+)\t(?P<ex>.+)$")

    def load_tags_from_file(self, path: str) -> int:
        """
        Parse a tags file (either JSON-lines produced by universal-ctags or classic tags format).
        Populates in-memory index and returns number of entries.
        """
        if not os.path.exists(path):
            logger.warning("tags file not found: %s", path)
            return 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except Exception as e:
            logger.exception("Failed to read tags file %s: %s", path, e)
            return 0

        # Heuristic: if file looks like JSON lines (starts with '{' on lines), parse JSON-lines
        lines = text.splitlines()
        json_like = all(line.strip().startswith("{") or not line.strip() for line in lines[:20])
        parsed: List[Dict] = []
        if json_like:
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if isinstance(obj, dict) and "name" in obj and "path" in obj:
                    parsed.append(obj)
        else:
            # parse classic tags file
            for ln in lines:
                if not ln or ln.startswith("!_TAG_"):
                    continue
                m = self._TAGS_LINE_RE.match(ln)
                if not m:
                    continue
                name = m.group("name")
                filep = m.group("file")
                ex = m.group("ex")
                # ex can be /^...$/;"\tkind\t...
                # try to extract kind and line
                kind = None
                line_no = None
                # extract ;"\t fields
                parts = ex.split(";\t")
                pattern = parts[0] if parts else ex
                # attempt to find 'line:123' or a number in pattern
                mline = re.search(r'line:(\d+)', ex)
                if mline:
                    line_no = int(mline.group(1))
                else:
                    # attempt to extract digits at end of pattern
                    mnum = re.search(r'(\d+)$', ex.strip())
                    if mnum:
                        line_no = int(mnum.group(1))
                # extract kind if present in parts
                if len(parts) > 1:
                    for p in parts[1:]:
                        if p.startswith("kind:") or p.startswith("kind\t"):
                            kind = p.split(":", 1)[-1].strip()
                        else:
                            # sometimes it's just a single letter kind field
                            # pattern ;"\tkind\t...
                            fields = p.split("\t")
                            if fields:
                                kind = fields[0]
                parsed.append({"name": name, "path": filep, "pattern": pattern, "line": line_no, "kind": kind})

        # normalize and populate
        self._populate(parsed)
        return len(self._all_entries)

    # -------------------------
    # Normalize & populate internal index
    # -------------------------
    def _populate(self, raw_entries: Iterable[Dict]) -> None:
        self._entries_by_name.clear()
        self._all_entries.clear()
        for obj in raw_entries:
            try:
                name = obj.get("name")
                path = obj.get("path") or obj.get("file") or obj.get("filename") or ""
                # If path is relative, make relative to repo_root
                if path and not os.path.isabs(path):
                    path = os.path.normpath(os.path.join(self.repo_root, path))
                line = None
                if "line" in obj and obj.get("line") is not None:
                    try:
                        line = int(obj.get("line"))
                    except Exception:
                        line = None
                # try pattern from obj
                pattern = obj.get("pattern") or obj.get("ex") or None
                kind = obj.get("kind") or obj.get("type") or None
                scope = obj.get("scope") or obj.get("scopeKind") or None
                language = obj.get("language") or obj.get("lang") or None

                entry = TagEntry(name=name, path=path, line=line, pattern=pattern, kind=kind, scope=scope, language=language, raw=obj)
                self._all_entries.append(entry)
                self._entries_by_name.setdefault(name, []).append(entry)
            except Exception:
                logger.exception("Failed to normalize tag entry: %s", obj)
                continue

    # -------------------------
    # Public query methods
    # -------------------------
    def get_tag_entries(self, name: str) -> List[Dict]:
        """
        Return raw tag entry dicts for a given symbol name (if any).
        """
        entries = self._entries_by_name.get(name) or []
        return [e.to_dict() for e in entries]

    def list_symbols(self) -> List[str]:
        """Return a sorted list of all symbol names indexed."""
        return sorted(self._entries_by_name.keys())

    def find_definitions(self, symbol: str, kinds: Optional[Iterable[str]] = None, max_results: int = 20) -> List[Dict]:
        """
        Return the most likely definition locations for a symbol.

        Strategy:
          - Prefer entries whose kind looks like function/class/def (kinds filter).
          - Prefer entries with an explicit line number.
          - Return normalized dicts.
        """
        if not symbol:
            return []
        entries = self._entries_by_name.get(symbol, [])[:]
        if not entries:
            # fallback: case-insensitive search in names
            names = [n for n in self._entries_by_name.keys() if n.lower() == symbol.lower() or symbol.lower() in n.lower()]
            for n in names:
                entries.extend(self._entries_by_name.get(n, []))
        # filter by kinds if provided
        if kinds:
            kinds_set = set(kinds)
            entries = [e for e in entries if (e.kind and e.kind in kinds_set)]
            if not entries:
                # if filtering removed all, ignore kinds
                entries = self._entries_by_name.get(symbol, [])[:]

        # sort: entries with line numbers first, then by path
        entries.sort(key=lambda e: (0 if e.line else 1, e.path or "", e.name or ""))
        return [e.to_dict() for e in entries[:max_results]]

    def find_references(self, symbol: str, max_results: int = 50) -> List[Dict]:
        """
        Find tag entries that reference the symbol. For ctags, this often equals definitions.
        This is a best-effort: we look for names equal to symbol and also substring matches.
        """
        if not symbol:
            return []
        exact = self._entries_by_name.get(symbol, [])[:]
        results = exact[:]
        # add substring matches
        for name, entries in self._entries_by_name.items():
            if symbol in name and name != symbol:
                results.extend(entries)
        # dedupe by path+line
        seen = set()
        out = []
        for e in results:
            key = (e.path, e.line, e.name)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
            if len(out) >= max_results:
                break
        return [e.to_dict() for e in out]

    # -------------------------
    # Utility / CLI helpers
    # -------------------------
    def refresh_if_stale(self) -> None:
        """
        If tags appear empty or repo has changed, optionally rebuild tags.
        (Very small heuristic: if we have no entries, try to build).
        """
        if not self._all_entries:
            try:
                self.build_tags()
            except Exception:
                pass

    # -------------------------
    # CLI for debug
    # -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="tags_client", description="ctags wrapper for symbol lookup")
    sub = parser.add_subparsers(dest="cmd")
    p_build = sub.add_parser("build", help="build tags for repo")
    p_build.add_argument("--repo", default=".", help="repo root")

    p_defs = sub.add_parser("defs", help="find definitions for symbol")
    p_defs.add_argument("symbol", help="symbol name")
    p_defs.add_argument("--repo", default=".", help="repo root")

    p_list = sub.add_parser("list", help="list symbols")
    p_list.add_argument("--repo", default=".", help="repo root")

    args = parser.parse_args()
    if args.cmd == "build":
        tc = TagsClient(repo_root=args.repo)
        n = tc.build_tags()
        print(f"Loaded {n} tags.")
    elif args.cmd == "defs":
        tc = TagsClient(repo_root=args.repo)
        tc.build_tags()
        res = tc.find_definitions(args.symbol)
        print(json.dumps(res, indent=2))
    elif args.cmd == "list":
        tc = TagsClient(repo_root=args.repo)
        tc.build_tags()
        syms = tc.list_symbols()
        print("\n".join(syms[:500]))
    else:
        parser.print_help()
