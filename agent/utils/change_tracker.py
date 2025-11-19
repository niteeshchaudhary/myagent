"""
Change tracker for highlighting file modifications made by the agent.

Features:
- Tracks file state before and after modifications
- Generates colored diffs showing additions (green) and removals (red)
- Displays changes in a readable format similar to IDE diff views
"""
from __future__ import annotations

import os
import sys
import difflib
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# ANSI color codes for terminal output
GREEN = '\033[92m'  # Green for additions
RED = '\033[91m'    # Red for removals
YELLOW = '\033[93m' # Yellow for context
BLUE = '\033[94m'   # Blue for file headers
RESET = '\033[0m'   # Reset color
BOLD = '\033[1m'    # Bold text
DIM = '\033[2m'     # Dim text

# Try to detect if terminal supports colors
def _supports_color() -> bool:
    """Check if terminal supports ANSI color codes."""
    try:
        import sys
        # Check if we're in a TTY
        if not sys.stdout.isatty():
            return False
        # Check environment variable
        if os.environ.get('NO_COLOR') or os.environ.get('TERM') == 'dumb':
            return False
        return True
    except:
        return False

SUPPORTS_COLOR = _supports_color()


@dataclass
class FileChange:
    """Represents a change to a single file."""
    file_path: str
    old_content: Optional[str]
    new_content: Optional[str]
    is_new_file: bool
    is_deleted: bool
    
    def get_diff(self, context_lines: int = 3) -> List[str]:
        """Generate unified diff lines for this change."""
        if self.is_deleted:
            old_lines = (self.old_content or "").splitlines(keepends=True)
            return [f"{RED}-{line}{RESET}" for line in old_lines]
        
        if self.is_new_file:
            new_lines = (self.new_content or "").splitlines(keepends=True)
            return [f"{GREEN}+{line}{RESET}" for line in new_lines]
        
        # Modified file - generate unified diff
        old_lines = (self.old_content or "").splitlines(keepends=True)
        new_lines = (self.new_content or "").splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}",
            lineterm='',
            n=context_lines
        )
        return list(diff)


class ChangeTracker:
    """
    Tracks file changes made by the agent and provides colored diff output.
    Supports review mode where changes can be accepted or rejected before applying.
    """
    
    def __init__(self, review_mode: bool = False):
        """
        Args:
            review_mode: If True, changes are stored as pending and require approval
        """
        self._file_snapshots: Dict[str, Optional[str]] = {}
        self._changes: List[FileChange] = []
        self._pending_changes: List[FileChange] = []
        self._review_mode = review_mode
    
    def snapshot_file(self, file_path: str) -> None:
        """
        Capture the current state of a file before modification.
        Call this before writing to a file.
        """
        if file_path in self._file_snapshots:
            # Already tracked, don't overwrite
            return
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    self._file_snapshots[file_path] = f.read()
            except Exception:
                self._file_snapshots[file_path] = None
        else:
            # File doesn't exist yet - it's a new file
            self._file_snapshots[file_path] = None
    
    def record_change(self, file_path: str, new_content: Optional[str] = None, pending: bool = None) -> None:
        """
        Record a change to a file after modification.
        Call this after writing to a file.
        
        Args:
            file_path: Path to the modified file
            new_content: The new content (if None, reads from file)
            pending: If True, store as pending change (overrides review_mode)
        """
        old_content = self._file_snapshots.get(file_path)
        
        if new_content is None:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        new_content = f.read()
                except Exception:
                    new_content = None
            else:
                new_content = None
        
        is_new_file = old_content is None and new_content is not None
        is_deleted = old_content is not None and new_content is None
        
        change = FileChange(
            file_path=file_path,
            old_content=old_content,
            new_content=new_content,
            is_new_file=is_new_file,
            is_deleted=is_deleted
        )
        
        # Store as pending if in review mode or explicitly requested
        if pending is None:
            pending = self._review_mode
        
        if pending:
            self._pending_changes.append(change)
        else:
            self._changes.append(change)
    
    def get_changes(self) -> List[FileChange]:
        """Get all recorded changes."""
        return self._changes.copy()
    
    def clear(self) -> None:
        """Clear all tracked changes and snapshots."""
        self._file_snapshots.clear()
        self._changes.clear()
        self._pending_changes.clear()
    
    def has_changes(self) -> bool:
        """Check if any changes have been recorded."""
        return len(self._changes) > 0 or len(self._pending_changes) > 0
    
    def has_pending_changes(self) -> bool:
        """Check if there are pending changes awaiting review."""
        return len(self._pending_changes) > 0
    
    def get_pending_changes(self) -> List[FileChange]:
        """Get all pending changes awaiting review."""
        return self._pending_changes.copy()
    
    def set_review_mode(self, enabled: bool) -> None:
        """Enable or disable review mode."""
        self._review_mode = enabled
    
    def apply_change(self, change: FileChange) -> bool:
        """
        Apply a single pending change to the file system.
        
        Returns:
            True if change was applied successfully, False otherwise
        """
        try:
            if change.is_deleted:
                if os.path.exists(change.file_path):
                    os.remove(change.file_path)
            else:
                # Create directory if needed
                dirpath = os.path.dirname(change.file_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                
                # Write the new content
                with open(change.file_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(change.new_content or "")
            
            # Move from pending to applied
            if change in self._pending_changes:
                self._pending_changes.remove(change)
            self._changes.append(change)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to apply change to {change.file_path}: {e}")
            return False
    
    def reject_change(self, change: FileChange) -> bool:
        """
        Reject a pending change (remove it from pending list).
        
        Returns:
            True if change was rejected successfully
        """
        if change in self._pending_changes:
            self._pending_changes.remove(change)
            return True
        return False
    
    def apply_all_pending(self) -> Dict[str, int]:
        """
        Apply all pending changes.
        
        Returns:
            Dict with 'applied' and 'failed' counts
        """
        results = {"applied": 0, "failed": 0}
        pending = self._pending_changes.copy()
        
        for change in pending:
            if self.apply_change(change):
                results["applied"] += 1
            else:
                results["failed"] += 1
        
        return results
    
    def reject_all_pending(self) -> int:
        """
        Reject all pending changes.
        
        Returns:
            Number of changes rejected
        """
        count = len(self._pending_changes)
        self._pending_changes.clear()
        return count
    
    def format_diff(self, change: FileChange, use_color: bool = None) -> str:
        """
        Format a single file change as a colored diff string.
        
        Args:
            change: The FileChange to format
            use_color: Whether to use colors (defaults to terminal support)
        """
        if use_color is None:
            use_color = SUPPORTS_COLOR
        
        if not use_color:
            # Plain text diff without colors
            if change.is_deleted:
                return f"--- {change.file_path} (deleted)\n" + \
                       "\n".join(f"-{line.rstrip()}" for line in (change.old_content or "").splitlines())
            if change.is_new_file:
                return f"+++ {change.file_path} (new file)\n" + \
                       "\n".join(f"+{line.rstrip()}" for line in (change.new_content or "").splitlines())
            
            # Modified file - use unified diff
            old_lines = (change.old_content or "").splitlines(keepends=True)
            new_lines = (change.new_content or "").splitlines(keepends=True)
            diff = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{change.file_path}",
                tofile=f"b/{change.file_path}",
                lineterm=''
            )
            return "\n".join(diff)
        
        # Colored diff
        lines = []
        
        # File header
        if change.is_new_file:
            lines.append(f"{BOLD}{GREEN}+++ {change.file_path} (new file){RESET}")
        elif change.is_deleted:
            lines.append(f"{BOLD}{RED}--- {change.file_path} (deleted){RESET}")
        else:
            lines.append(f"{BOLD}{BLUE}--- a/{change.file_path}{RESET}")
            lines.append(f"{BOLD}{BLUE}+++ b/{change.file_path}{RESET}")
        
        if change.is_deleted:
            old_lines = (change.old_content or "").splitlines()
            for line in old_lines:
                lines.append(f"{RED}-{line}{RESET}")
            return "\n".join(lines)
        
        if change.is_new_file:
            new_lines = (change.new_content or "").splitlines()
            for line in new_lines:
                lines.append(f"{GREEN}+{line}{RESET}")
            return "\n".join(lines)
        
        # Modified file - generate colored unified diff
        old_lines = (change.old_content or "").splitlines(keepends=True)
        new_lines = (change.new_content or "").splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{change.file_path}",
            tofile=f"b/{change.file_path}",
            lineterm='',
            n=3
        )
        
        for line in diff:
            if line.startswith('---'):
                lines.append(f"{BOLD}{BLUE}{line}{RESET}")
            elif line.startswith('+++'):
                lines.append(f"{BOLD}{BLUE}{line}{RESET}")
            elif line.startswith('@@'):
                lines.append(f"{DIM}{YELLOW}{line}{RESET}")
            elif line.startswith('-'):
                lines.append(f"{RED}{line}{RESET}")
            elif line.startswith('+'):
                lines.append(f"{GREEN}{line}{RESET}")
            else:
                lines.append(line.rstrip())
        
        return "\n".join(lines)
    
    def display_changes(self, use_color: bool = None, show_pending: bool = True) -> str:
        """
        Display all recorded changes as a formatted diff.
        
        Args:
            use_color: Whether to use colors (defaults to terminal support)
            show_pending: If True, show pending changes; if False, only show applied changes
        
        Returns:
            Formatted string showing all changes
        """
        changes_to_show = []
        if show_pending:
            changes_to_show.extend(self._pending_changes)
        changes_to_show.extend(self._changes)
        
        if not changes_to_show:
            return ""
        
        if use_color is None:
            use_color = SUPPORTS_COLOR
        
        output_lines = []
        
        if use_color:
            output_lines.append(f"\n{BOLD}{'='*80}{RESET}")
            if self.has_pending_changes():
                output_lines.append(f"{BOLD}Agent Changes Summary (PENDING REVIEW){RESET}")
            else:
                output_lines.append(f"{BOLD}Agent Changes Summary{RESET}")
            output_lines.append(f"{BOLD}{'='*80}{RESET}\n")
        else:
            output_lines.append("\n" + "="*80)
            if self.has_pending_changes():
                output_lines.append("Agent Changes Summary (PENDING REVIEW)")
            else:
                output_lines.append("Agent Changes Summary")
            output_lines.append("="*80 + "\n")
        
        for i, change in enumerate(changes_to_show, 1):
            is_pending = change in self._pending_changes
            status = " [PENDING]" if is_pending else " [APPLIED]"
            
            if use_color:
                status_color = YELLOW if is_pending else GREEN
                output_lines.append(f"{BOLD}Change {i}: {change.file_path}{status_color}{status}{RESET}")
            else:
                output_lines.append(f"Change {i}: {change.file_path}{status}")
            
            output_lines.append(self.format_diff(change, use_color=use_color))
            output_lines.append("")  # Empty line between changes
        
        return "\n".join(output_lines)
    
    def print_changes(self, use_color: bool = None) -> None:
        """Print all recorded changes to stdout."""
        output = self.display_changes(use_color=use_color)
        if output:
            print(output)
    
    def review_changes(self, use_color: bool = None) -> Dict[str, int]:
        """
        Interactive review of pending changes with accept/reject options.
        Similar to IDE diff review interfaces.
        
        Returns:
            Dict with 'accepted', 'rejected', and 'cancelled' counts
        """
        if not self.has_pending_changes():
            return {"accepted": 0, "rejected": 0, "cancelled": 0}
        
        if use_color is None:
            use_color = SUPPORTS_COLOR
        
        results = {"accepted": 0, "rejected": 0, "cancelled": 0}
        pending = self._pending_changes.copy()
        
        # Check if we're in an interactive terminal
        if not (sys.stdin and sys.stdin.isatty()):
            # Not interactive - auto-apply all pending changes
            print("\n[Non-interactive mode] Auto-applying all pending changes...")
            apply_results = self.apply_all_pending()
            results["accepted"] = apply_results["applied"]
            return results
        
        # Show all changes first
        print(self.display_changes(use_color=use_color, show_pending=True))
        
        # Ask for bulk action first
        print(f"\n{BOLD if use_color else ''}{'='*80}{RESET if use_color else ''}")
        print(f"{BOLD if use_color else ''}Review Options:{RESET if use_color else ''}")
        print("  [A]ccept all changes")
        print("  [R]eject all changes")
        print("  [I]nteractive (review each change individually)")
        print("  [C]ancel (keep changes pending)")
        print(f"{BOLD if use_color else ''}{'='*80}{RESET if use_color else ''}\n")
        
        try:
            choice = input("Your choice (A/R/I/C): ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\nReview cancelled.")
            return results
        
        if choice == 'A':
            # Accept all
            apply_results = self.apply_all_pending()
            results["accepted"] = apply_results["applied"]
            results["rejected"] = apply_results["failed"]
            print(f"\n✅ Accepted {results['accepted']} change(s)")
            if results["rejected"] > 0:
                print(f"⚠️  Failed to apply {results['rejected']} change(s)")
            return results
        
        elif choice == 'R':
            # Reject all
            count = self.reject_all_pending()
            results["rejected"] = count
            print(f"\n❌ Rejected {count} change(s)")
            return results
        
        elif choice == 'C':
            # Cancel - keep pending
            results["cancelled"] = len(pending)
            print(f"\n⏸️  Kept {len(pending)} change(s) pending")
            return results
        
        elif choice == 'I':
            # Interactive review
            for i, change in enumerate(pending, 1):
                print(f"\n{BOLD if use_color else ''}{'='*80}{RESET if use_color else ''}")
                print(f"{BOLD if use_color else ''}Change {i}/{len(pending)}: {change.file_path}{RESET if use_color else ''}")
                print(self.format_diff(change, use_color=use_color))
                print(f"{BOLD if use_color else ''}{'='*80}{RESET if use_color else ''}")
                
                while True:
                    try:
                        action = input("\n[A]ccept, [R]eject, [S]kip, [Q]uit review: ").strip().upper()
                    except (KeyboardInterrupt, EOFError):
                        print("\nReview cancelled.")
                        return results
                    
                    if action == 'A':
                        if self.apply_change(change):
                            results["accepted"] += 1
                            print(f"✅ Accepted change to {change.file_path}")
                        else:
                            results["rejected"] += 1
                            print(f"❌ Failed to apply change to {change.file_path}")
                        break
                    elif action == 'R':
                        if self.reject_change(change):
                            results["rejected"] += 1
                            print(f"❌ Rejected change to {change.file_path}")
                        break
                    elif action == 'S':
                        print(f"⏭️  Skipped change to {change.file_path}")
                        break
                    elif action == 'Q':
                        print("\nReview cancelled.")
                        return results
                    else:
                        print("Invalid choice. Please enter A, R, S, or Q.")
            
            return results
        
        else:
            print("Invalid choice. Changes remain pending.")
            results["cancelled"] = len(pending)
            return results


# Global change tracker instance
_global_tracker: Optional[ChangeTracker] = None


def get_change_tracker() -> ChangeTracker:
    """Get the global change tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ChangeTracker()
    return _global_tracker


def reset_change_tracker() -> None:
    """Reset the global change tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.clear()
    _global_tracker = None

