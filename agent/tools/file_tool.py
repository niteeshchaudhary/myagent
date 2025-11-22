# file_tool.py
"""
Robust file tool with diff-first behavior.

Provides:
- read(path)
- write(path, content, author=None, dry_run=False, make_dirs=True)
- delete(path, dry_run=False)
- list_dir(path)
- exists(path)
- handle(request_dict)  # convenience entrypoint used by tool managers

Behavior:
- If writing to an existing file, compute a unified diff between current and new content.
  Attempt to apply the unified diff via the repo patcher (apply_unified_diff).
  If that succeeds, report "patched". If it fails, fall back to an atomic file write and report the reason.
- Uses UTF-8 everywhere and atomic replace (write to tmp -> os.replace).
- Returns structured dicts: {"ok": bool, "action": str, "path": str, "message": str, "meta": {...}}
"""

from __future__ import annotations
import os
import io
import difflib
import tempfile
import shutil
import logging
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger("file_tool")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)


# Try to import the project's patcher (preferred). Fall back gracefully.
_apply_unified_diff = None  # type: Optional[callable]
try:
    # project layout may differ; prefer canonical import if available
    from agent.core.patcher import apply_unified_diff as _apply_unified_diff  # type: ignore
    logger.debug("Imported agent.core.patcher.apply_unified_diff")
except Exception:
    try:
        # fallback to local module if present at top-level (e.g., patcher.py)
        from patcher import apply_unified_diff as _apply_unified_diff  # type: ignore
        logger.debug("Imported local patcher.apply_unified_diff")
    except Exception:
        _apply_unified_diff = None
        logger.debug("No patcher.apply_unified_diff found; will fallback to atomic writes.")


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _atomic_write(path: str, content: str) -> None:
    dirpath = os.path.dirname(path) or "."
    # ensure directory exists
    os.makedirs(dirpath, exist_ok=True)
    # write to temp and replace
    fd, tmpname = tempfile.mkstemp(dir=dirpath, prefix=".tmp_filetool_", text=True)
    os.close(fd)  # we'll open using python's io
    try:
        with io.open(tmpname, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        os.replace(tmpname, path)
    finally:
        # cleanup if tmp remains
        if os.path.exists(tmpname):
            try:
                os.remove(tmpname)
            except Exception:
                pass


def _compute_unified_diff(path: str, new_content: str) -> str:
    """
    Compute a git-style unified diff string (no index headers).
    fromfile/tofile use a/<path> and b/<path> to match git patch style.
    """
    try:
        old_text = _read_text_file(path)
    except FileNotFoundError:
        old_lines: List[str] = []
    else:
        old_lines = old_text.splitlines(keepends=False)
    new_lines = new_content.splitlines(keepends=False)
    diff_lines = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm=""
    ))
    return "\n".join(diff_lines)


def read(path: str) -> Dict[str, Any]:
    try:
        content = _read_text_file(path)
        return {"ok": True, "action": "read", "path": path, "content": content}
    except FileNotFoundError:
        return {"ok": False, "action": "read", "path": path, "message": "file not found"}
    except Exception as e:
        logger.exception("Error reading file %s", path)
        return {"ok": False, "action": "read", "path": path, "message": str(e)}


def exists(path: str) -> bool:
    return os.path.exists(path)


def list_dir(path: str) -> Dict[str, Any]:
    try:
        entries = os.listdir(path)
        return {"ok": True, "action": "list", "path": path, "entries": entries}
    except FileNotFoundError:
        return {"ok": False, "action": "list", "path": path, "message": "dir not found"}
    except Exception as e:
        logger.exception("Error listing dir %s", path)
        return {"ok": False, "action": "list", "path": path, "message": str(e)}


def delete(path: str, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"ok": True, "action": "delete", "path": path, "message": "dry_run - not deleting"}
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        return {"ok": True, "action": "delete", "path": path, "message": "deleted"}
    except FileNotFoundError:
        return {"ok": False, "action": "delete", "path": path, "message": "file not found"}
    except Exception as e:
        logger.exception("Error deleting %s", path)
        return {"ok": False, "action": "delete", "path": path, "message": str(e)}


def write(
    path: str,
    content: str,
    *,
    author: Optional[str] = None,
    dry_run: bool = False,
    make_dirs: bool = True,
    prefer_patch: bool = True,
    repo_root: Optional[str] = None
) -> Dict[str, Any]:
    """
    Write content to path using diff-first approach.

    Args:
      path: filesystem path (relative or absolute)
      content: the new file contents (UTF-8)
      author: optional metadata string
      dry_run: if True, do not modify filesystem; report what would be done
      make_dirs: create parent directories if needed
      prefer_patch: if True and file exists, attempt to apply unified diff via patcher
      repo_root: optional repo root to pass to patcher.apply_unified_diff

    Returns:
      dict with keys: ok (bool), action (patched/wrote/no-op/failed), path, message, meta
    """
    meta: Dict[str, Any] = {"author": author}
    path_exists = os.path.exists(path)

    if path_exists:
        # compare contents
        try:
            old_text = _read_text_file(path)
        except Exception as e:
            logger.exception("Failed to read existing file %s", path)
            old_text = None

        if old_text is not None and old_text == content:
            return {"ok": True, "action": "no-op", "path": path, "message": "content identical", "meta": meta}

        # produce unified diff
        diff_text = _compute_unified_diff(path, content)
        meta["diff"] = diff_text

        if dry_run:
            return {"ok": True, "action": "dry-run", "path": path, "message": "would patch", "meta": meta}

        # attempt to apply unified diff via patcher if available and prefer_patch True
        if prefer_patch and _apply_unified_diff is not None:
            try:
                # apply_unified_diff is expected to return (ok: bool, report: str) or raise
                # adapt to both styles
                try:
                    result = _apply_unified_diff(diff_text, repo_root=repo_root or ".")
                    if isinstance(result, tuple):
                        ok, report = result
                    elif isinstance(result, dict):
                        ok = result.get("ok", False)
                        report = result.get("report", str(result))
                    else:
                        # unknown but truthy
                        ok, report = (True, str(result))
                except TypeError:
                    # maybe function signature does not accept repo_root
                    result = _apply_unified_diff(diff_text)
                    if isinstance(result, tuple):
                        ok, report = result
                    elif isinstance(result, dict):
                        ok = result.get("ok", False)
                        report = result.get("report", str(result))
                    else:
                        ok, report = (True, str(result))

                if ok:
                    return {"ok": True, "action": "patched", "path": path, "message": "applied unified diff", "meta": meta}
                else:
                    # patcher reported failure; fall through to fallback write
                    meta["patcher_report"] = report
                    logger.warning("Patcher failed for %s: %s", path, report)
            except Exception as e:
                logger.exception("Exception while applying unified diff for %s", path)
                meta["patcher_exception"] = str(e)
                # fall through to fallback atomic write

        # fallback: atomic write (create backup first)
        try:
            # backup existing file
            backup_path = f"{path}.bak"
            try:
                shutil.copyfile(path, backup_path)
                meta["backup"] = backup_path
            except Exception:
                # non-fatal if backup fails; continue
                meta["backup_error"] = "backup failed"

            if make_dirs:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

            _atomic_write(path, content)
            return {"ok": True, "action": "wrote", "path": path, "message": "wrote file (fallback after patch failure)", "meta": meta}
        except Exception as e:
            logger.exception("Failed to write file %s", path)
            return {"ok": False, "action": "failed", "path": path, "message": str(e), "meta": meta}

    else:
        # file does not exist -> create new file
        if dry_run:
            diff_text = _compute_unified_diff(path, content)
            meta["diff"] = diff_text
            return {"ok": True, "action": "dry-run-create", "path": path, "message": "would create new file", "meta": meta}

        try:
            if make_dirs:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _atomic_write(path, content)
            return {"ok": True, "action": "created", "path": path, "message": "created new file", "meta": meta}
        except Exception as e:
            logger.exception("Failed to create file %s", path)
            return {"ok": False, "action": "failed", "path": path, "message": str(e), "meta": meta}


def handle(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience entrypoint. Expected request dict examples:
      {"action":"write","path":"src/foo.py","content":"...","dry_run":False}
      {"action":"read","path":"src/foo.py"}
      {"action":"list","path":"src"}
      {"action":"delete","path":"tmp","dry_run":True}
      {"action":"exists","path":"src/foo.py"}

    Returns structured dict result.
    """
    action = request.get("action")
    path = request.get("path")
    if not action or not path:
        return {"ok": False, "message": "missing action or path", "request": request}

    action = action.lower()
    if action == "read":
        return read(path)
    if action == "exists":
        return {"ok": True, "action": "exists", "path": path, "exists": exists(path)}
    if action == "list":
        return list_dir(path)
    if action == "delete":
        return delete(path, dry_run=bool(request.get("dry_run", False)))
    if action == "write":
        content = request.get("content", "")
        return write(
            path,
            content,
            author=request.get("author"),
            dry_run=bool(request.get("dry_run", False)),
            make_dirs=bool(request.get("make_dirs", True)),
            prefer_patch=bool(request.get("prefer_patch", True)),
            repo_root=request.get("repo_root"),
        )

    return {"ok": False, "message": f"unknown action: {action}", "request": request}


# If run as a script, provide a tiny CLI for ad-hoc testing
if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["read", "write", "list", "delete", "exists"])
    ap.add_argument("path")
    ap.add_argument("--content", help="content for writes", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true", help="print json result")
    args = ap.parse_args()

    req = {"action": args.action, "path": args.path, "dry_run": args.dry_run}
    if args.content is not None:
        req["content"] = args.content

    result = handle(req)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result.get("message") or result)
