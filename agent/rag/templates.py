# agent/rag/templates.py
"""
RAG prompt templates and small helpers.

This module centralizes the prompt templates used when composing grounded prompts
for the LLM in RAG flows: answering questions, requesting patches, and asking
for todo lists. Keeping templates in one place makes it easy to tune prompts
and ensures consistent instruction style (cite sources, avoid hallucination).

Public helpers:
- grounding_prompt(question, chunks, config=None) -> str
- patch_request_prompt(linter_output, test_output, repo_path, extra_instructions=None) -> str
- todo_list_prompt(user_request, memory_snippet=None, max_items=12) -> str
- answer_with_sources_contraints() -> small constant with common constraints
"""

from __future__ import annotations

import textwrap
from typing import Iterable, List, Dict, Optional

# prefer project logger if available
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


# ---- small reusable pieces ----
_DEFAULT_INSTRUCTIONS = [
    "You are a precise coding assistant with knowledge of the user's repository.",
    "Use ONLY the provided source excerpts to answer questions or propose changes.",
    "If the answer cannot be determined from the sources, say you don't know and list which files to inspect.",
    "Cite the files and (if available) the line ranges you used for each factual claim.",
]

_COMMON_CONSTRAINTS = (
    "- Answer concisely and accurately.\n"
    "- Do not invent facts or code that is not grounded in the provided sources.\n"
    "- When giving code snippets, include only the minimal necessary edits and keep surrounding context short.\n"
    "- At the end, include a SOURCES: section listing file paths and line ranges you relied on."
)


def _format_header(title: str) -> str:
    return f"=== {title} ===\n"


def _truncate_text(s: str, max_chars: int) -> str:
    if max_chars and len(s) > max_chars:
        return s[: max_chars - 120] + "\n\n...[TRUNCATED]..."
    return s


# ---- public template helpers ----
def grounding_prompt(
    question: str,
    chunks: Iterable[Dict],
    *,
    max_chars_per_chunk: int = 3000,
    instructions: Optional[List[str]] = None,
) -> str:
    """
    Compose a grounded prompt for answering a question about the repo.

    `chunks` is an iterable of dicts with keys:
      - source_type: "exact"|"semantic"|"file_expansion"
      - path, start_line, end_line, text, score (optional)

    Returns a single string prompt suitable to pass to the LLM.
    """
    instructions = instructions or _DEFAULT_INSTRUCTIONS
    lines: List[str] = []
    lines.append("\n".join(instructions))
    lines.append("")  # spacer

    # Group chunks by source_type for readability
    exacts = [c for c in chunks if c.get("source_type") == "exact"]
    sems = [c for c in chunks if c.get("source_type") == "semantic"]
    files = [c for c in chunks if c.get("source_type") == "file_expansion"]

    if exacts:
        lines.append(_format_header("EXACT SOURCES"))
        for i, c in enumerate(exacts, 1):
            path = c.get("path") or "<unknown>"
            sl = c.get("start_line")
            el = c.get("end_line")
            hdr = f"{i}) <FILE: {path}"
            if sl or el:
                hdr += f" lines {sl or '?'}-{el or '?'}"
            hdr += ">"
            lines.append(hdr)
            txt = _truncate_text(c.get("text") or "", max_chars_per_chunk)
            lines.append("```file")
            lines.append(txt)
            lines.append("```")
            lines.append("")

    if sems:
        lines.append(_format_header("RETRIEVED SOURCES (semantic)"))
        for i, c in enumerate(sems, 1):
            path = c.get("path") or "<unknown>"
            sl = c.get("start_line")
            el = c.get("end_line")
            hdr = f"{i}) <FILE: {path}"
            if sl or el:
                hdr += f" lines {sl or '?'}-{el or '?'}"
            hdr += f"> (score={c.get('score', 0):.3f})"
            lines.append(hdr)
            txt = _truncate_text(c.get("text") or "", max_chars_per_chunk)
            lines.append("```file")
            lines.append(txt)
            lines.append("```")
            lines.append("")

    if files:
        lines.append(_format_header("EXPLICIT FILE EXPANSIONS"))
        for i, c in enumerate(files, 1):
            lines.append(f"{i}) <EXPANDED FILE BLOCK>")
            txt = _truncate_text(c.get("text") or "", max_chars_per_chunk)
            lines.append("```file")
            lines.append(txt)
            lines.append("```")
            lines.append("")

    # Question and constraints
    lines.append(_format_header("QUESTION"))
    lines.append(question)
    lines.append("")
    lines.append(_format_header("CONSTRAINTS"))
    lines.append(_COMMON_CONSTRAINTS)
    return "\n".join(lines)


def patch_request_prompt(
    linter_output: str,
    test_output: str,
    repo_path: str,
    *,
    extra_instructions: Optional[str] = None,
    require_unified_diff: bool = True,
    minimal_changes: bool = True,
) -> str:
    """
    Compose a prompt asking the LLM to produce a unified diff patch that fixes issues.
    The prompt encourages minimal changes and asks specifically for a git-style unified diff
    wrapped in a ```diff code fence.

    Returns a single prompt string.
    """
    parts: List[str] = []
    parts.append("You are an assistant that generates a **git-style unified diff** to fix issues in a repository.")
    parts.append(f"Repository path: {repo_path}")
    parts.append("")
    if linter_output:
        parts.append("LINTER OUTPUT (flake8 / pylint):")
        parts.append("```")
        parts.append(linter_output.strip() or "(none)")
        parts.append("```")
        parts.append("")
    if test_output:
        parts.append("TEST OUTPUT (pytest):")
        parts.append("```")
        parts.append(test_output.strip() or "(none)")
        parts.append("```")
        parts.append("")

    # directives for patch quality
    directives = [
        "Guidelines for the patch:",
        "- Produce a git-style unified diff (use `diff --git a/... b/...` format if possible).",
        "- Wrap the diff in a ```diff ... ``` fenced code block with no extra commentary outside the fence.",
    ]
    if minimal_changes:
        directives.append("- Make the minimal changes necessary to fix the linter/tests. Do not refactor unrelated code.")
    directives.append("- If the fix requires adding tests, include them in the diff.")
    directives.append("- Do not include long explanations; only return the diff.")
    parts.extend(directives)
    if extra_instructions:
        parts.append("")
        parts.append("Additional instructions:")
        parts.append(extra_instructions.strip())

    if require_unified_diff:
        parts.append("")
        parts.append("IMPORTANT: Return only the unified diff inside a ```diff ... ``` code block. Do not provide any other text.")
    else:
        parts.append("")
        parts.append("You may return a unified diff inside a ```diff ... ``` block. If unsure, return a diff and a short summary after the block.")

    return "\n".join(parts)


def todo_list_prompt(user_request: str, *, memory_snippet: Optional[str] = None, max_items: int = 12) -> str:
    """
    Ask the LLM (planner) to generate a concise JSON todo list from the user's request.

    Returns a prompt that requests JSON only, e.g. {"todos": ["step 1", ...]}
    """
    base = textwrap.dedent(
        f"""
        You are an assistant that turns user requests into an actionable to-do list of implementation steps.
        Produce a short, ordered list of concrete, implementable steps (tasks) that the agent can execute one-by-one.
        Each todo must be a single sentence and describe **one** action (e.g., 'Create CLI command `hello` that prints hello world', 'Add tests in tests/test_cli.py', 'Run flake8 and fix style issues').
        Return ONLY valid JSON in the following form:

        {{ "todos": ["step 1", "step 2", "..."] }}

        Constraints:
        - Produce no more than {max_items} items.
        - Prefer small, verifiable steps that can be executed automatically.
        - Use imperative phrasing.
        - If the task requires human judgement or privileged access, include a step that requests confirmation.

        User request:
        """
    ).strip()
    if memory_snippet:
        base = base + "\n\nUseful context / memory:\n" + memory_snippet.strip() + "\n\n"
    base = base + "\n" + user_request.strip() + "\n\nReturn JSON only."
    return base


def answer_with_sources_constraints() -> str:
    """
    Return a short, reusable constraints string ensuring the LLM cites sources and avoids hallucination.
    """
    return _COMMON_CONSTRAINTS


# ---- tiny utilities helpful to other modules ----
def format_sources_list_for_output(chunks: Iterable[Dict]) -> str:
    """
    Given retrieved chunks, produce a short numbered sources list suitable to append to an LLM answer.
    Each line looks like: "- path: start-end (semantic/exact)"
    """
    out = []
    for c in chunks:
        p = c.get("path") or "<unknown>"
        sl = c.get("start_line")
        el = c.get("end_line")
        st = c.get("source_type") or "semantic"
        if sl or el:
            out.append(f"- {p} lines {sl or '?'}-{el or '?'} ({st})")
        else:
            out.append(f"- {p} ({st})")
    return "\n".join(out)
