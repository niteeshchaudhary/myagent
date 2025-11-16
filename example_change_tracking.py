#!/usr/bin/env python3
"""
Example script demonstrating the change tracking and review functionality.

This shows how the agent highlights file changes with green for additions
and red for removals, similar to IDE diff views. It also demonstrates
the review mode where changes can be accepted or rejected before applying.
"""

import os
import tempfile
from agent.utils.change_tracker import get_change_tracker, reset_change_tracker

def example_review_mode():
    """Demonstrate review mode with accept/reject functionality."""
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        test_file = f.name
        initial_content = """def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
"""
        f.write(initial_content)
    
    try:
        tracker = get_change_tracker()
        
        # Enable review mode
        tracker.set_review_mode(True)
        
        # Snapshot the file before modification
        tracker.snapshot_file(test_file)
        
        # Create a pending change (simulating what happens in review mode)
        modified_content = """def hello():
    print("Hello, World!")
    print("Modified by agent")

def goodbye():
    print("Goodbye!")

def new_function():
    print("This is a new function")
"""
        # Record as pending change (not applied yet)
        tracker.record_change(test_file, new_content=modified_content, pending=True)
        
        print("=" * 80)
        print("Example: Review Mode with Accept/Reject")
        print("=" * 80)
        print("\nIn review mode, changes are stored as pending and require approval.")
        print("The file has NOT been modified yet.\n")
        
        # Show pending changes
        tracker.print_changes()
        
        print("\n" + "=" * 80)
        print("To use review mode in your agent:")
        print("1. Set AgentLoopConfig(review_mode=True)")
        print("2. When the agent makes file changes, they'll be stored as pending")
        print("3. The review_changes() method will be called automatically")
        print("4. You can accept all, reject all, or review each change individually")
        print("=" * 80)
        
        # Clean up - reject the pending change
        tracker.reject_all_pending()
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
        reset_change_tracker()

def example_file_changes():
    """Demonstrate change tracking with file modifications."""
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
        test_file = f.name
        initial_content = """def hello():
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
"""
        f.write(initial_content)
    
    try:
        tracker = get_change_tracker()
        
        # Snapshot the file before modification
        tracker.snapshot_file(test_file)
        
        # Modify the file
        modified_content = """def hello():
    print("Hello, World!")
    print("Modified by agent")

def goodbye():
    print("Goodbye!")

def new_function():
    print("This is a new function")
"""
        with open(test_file, 'w') as f:
            f.write(modified_content)
        
        # Record the change
        tracker.record_change(test_file, new_content=modified_content)
        
        # Display the changes
        print("=" * 80)
        print("Example: File Changes Highlighted")
        print("=" * 80)
        tracker.print_changes()
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        reset_change_tracker()

def example_new_file():
    """Demonstrate tracking a new file creation."""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        test_file = f.name
    
    try:
        tracker = get_change_tracker()
        
        # For a new file, snapshot will record None
        tracker.snapshot_file(test_file)
        
        # Write new content
        new_content = """This is a new file created by the agent.
It contains some content.
And more content here.
"""
        with open(test_file, 'w') as f:
            f.write(new_content)
        
        # Record the change
        tracker.record_change(test_file, new_content=new_content)
        
        # Display the changes
        print("\n" + "=" * 80)
        print("Example: New File Creation")
        print("=" * 80)
        tracker.print_changes()
        
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
        reset_change_tracker()

if __name__ == "__main__":
    print("\nChange Tracking & Review Demo")
    print("=" * 80)
    print("This demonstrates how the agent highlights file changes:")
    print("- Green lines show additions")
    print("- Red lines show removals")
    print("- Context lines show unchanged code")
    print("- Review mode allows accept/reject before applying changes")
    print("=" * 80 + "\n")
    
    example_file_changes()
    example_new_file()
    example_review_mode()
    
    print("\n" + "=" * 80)
    print("Note: Colors will only appear in terminals that support ANSI codes.")
    print("=" * 80)

