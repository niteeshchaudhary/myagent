import os
from agent.utils.logger import log
from agent.utils.change_tracker import get_change_tracker


class FileTool:
    """
    File operations tool for reading, writing, and managing files.
    
    Supports:
    - read: Read file contents
    - write: Write content to file
    - append: Append content to file
    - delete: Delete a file
    - exists: Check if file exists
    - list: List directory contents
    - search: Search for keywords in file
    """
    
    name = "file"
    description = "File operations tool for reading, writing, and managing files"
    
    def run(self, input_data, **kwargs):
        """
        Execute file operation.
        
        Input can be:
        - A dict with 'action' and other fields
        - A string that will be parsed as action or treated as write action with content
        """
        # Check if previous output is available in kwargs (from agent loop)
        previous_output = kwargs.get("previous_output") or kwargs.get("last_output")
        
        # Handle string input - could be a simple write command or JSON
        if isinstance(input_data, str):
            # If it looks like a write command, parse it
            if "Write" in input_data or "write" in input_data or "into" in input_data.lower():
                # Try to extract file path and content from natural language
                # Example: "Write the generated C++ code into abc.cpp"
                import re
                # Look for "into <filename>" pattern
                match = re.search(r'into\s+([^\s]+)', input_data, re.IGNORECASE)
                if match:
                    filename = match.group(1)
                    
                    # Try to extract code from markdown code blocks in the input or previous output
                    content = ""
                    
                    # First, try to extract from markdown code blocks in input_data
                    code_block_pattern = r'```(?:cpp|c\+\+|c|python|py|javascript|js|java|html|css|json|xml|bash|sh)?\s*\n(.*?)```'
                    code_matches = re.findall(code_block_pattern, input_data, re.DOTALL)
                    if code_matches:
                        # Use the last/largest code block found
                        content = max(code_matches, key=len).strip()
                    
                    # If no code block in input, check previous_output
                    if not content and previous_output:
                        if isinstance(previous_output, str):
                            code_matches = re.findall(code_block_pattern, previous_output, re.DOTALL)
                            if code_matches:
                                content = max(code_matches, key=len).strip()
                        elif isinstance(previous_output, dict):
                            # Check common output fields
                            output_text = previous_output.get("output") or previous_output.get("text") or str(previous_output)
                            code_matches = re.findall(code_block_pattern, output_text, re.DOTALL)
                            if code_matches:
                                content = max(code_matches, key=len).strip()
                    
                    # If still no content and input mentions "generated code", use previous_output directly
                    if not content and ("generated" in input_data.lower() or "code" in input_data.lower()):
                        if previous_output:
                            if isinstance(previous_output, str):
                                content = previous_output
                            elif isinstance(previous_output, dict):
                                content = previous_output.get("output") or previous_output.get("text") or str(previous_output)
                    
                    # If still no content, try to extract from the input string itself
                    if not content:
                        # Remove the command part and see if there's any code left
                        remaining = input_data.replace(f"into {filename}", "").replace("Write", "").replace("write", "").replace("the generated", "").replace("generated", "").strip()
                        if remaining and len(remaining) > 20:  # Only use if substantial content
                            content = remaining
                    
                    input_data = {
                        "action": "write",
                        "path": filename,
                        "content": content or ""
                    }
                else:
                    # Fallback: treat as action name
                    input_data = {"action": input_data}
            else:
                # Try to parse as JSON
                try:
                    import json
                    input_data = json.loads(input_data)
                except:
                    # If not JSON, treat as action name
                    input_data = {"action": input_data}
        
        if not isinstance(input_data, dict):
            raise ValueError("File tool expects a dict input.")
        
        return file_tool(input_data)


# Also create an alias class for file_editor
class FileEditorTool(FileTool):
    """Alias for FileTool to support 'file_editor' tool name."""
    name = "file_editor"


def file_tool(action: dict):
    """
    General file tool.
    
    Input:
        {
            "action": "read",
            "path": "test.txt"
        }
    
    Supported actions:
        - read
        - write
        - append
        - delete
        - exists
        - list
        - search
    """

    if not isinstance(action, dict):
        raise ValueError("File tool expects a dict input.")

    act = action.get("action")
    path = action.get("path")

    if act in ["read", "write", "append", "delete", "search"] and not path:
        raise ValueError("Path is required for this action.")

    log.info(f"[file] Action: {act}, Path: {path}")

    # ---------------------------------------------------------
    # READ FILE
    # ---------------------------------------------------------
    if act == "read":
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ---------------------------------------------------------
    # WRITE FILE
    # ---------------------------------------------------------
    if act == "write":
        content = action.get("content", "")
        # Track changes: snapshot before modification
        tracker = get_change_tracker()
        tracker.snapshot_file(path)
        
        # Check if in review mode
        is_pending = tracker._review_mode if hasattr(tracker, '_review_mode') else False
        
        if is_pending:
            # Store as pending change (don't write yet)
            tracker.record_change(path, new_content=content, pending=True)
            return f"Change to {path} stored as pending (awaiting review)."
        else:
            # Create parent directories if they don't exist
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                log.info(f"[file] Created directory: {dir_path}")
            
            # Write immediately
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Record the change
            tracker.record_change(path, new_content=content, pending=False)
            
            return "File written successfully."

    # ---------------------------------------------------------
    # APPEND FILE
    # ---------------------------------------------------------
    if act == "append":
        content = action.get("content", "")
        # Track changes: snapshot before modification
        tracker = get_change_tracker()
        tracker.snapshot_file(path)
        
        # Check if in review mode
        is_pending = tracker._review_mode if hasattr(tracker, '_review_mode') else False
        
        if is_pending:
            # For append, we need to read current content and append to it
            old_content = tracker._file_snapshots.get(path, "")
            new_content = (old_content or "") + content
            tracker.record_change(path, new_content=new_content, pending=True)
            return f"Append to {path} stored as pending (awaiting review)."
        else:
            # Create parent directories if they don't exist
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                log.info(f"[file] Created directory: {dir_path}")
            
            # Write immediately
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            
            # Record the change (read full file to get new content)
            tracker.record_change(path, pending=False)
            
            return "Content appended successfully."

    # ---------------------------------------------------------
    # DELETE FILE
    # ---------------------------------------------------------
    if act == "delete":
        # Track changes: snapshot before deletion
        tracker = get_change_tracker()
        tracker.snapshot_file(path)
        
        # Check if in review mode
        is_pending = tracker._review_mode if hasattr(tracker, '_review_mode') else False
        
        if is_pending:
            # Store as pending deletion
            tracker.record_change(path, new_content=None, pending=True)
            return f"Deletion of {path} stored as pending (awaiting review)."
        else:
            if os.path.exists(path):
                os.remove(path)
                # Record deletion
                tracker.record_change(path, new_content=None, pending=False)
                return "File deleted."
            return "File does not exist."

    # ---------------------------------------------------------
    # EXISTS
    # ---------------------------------------------------------
    if act == "exists":
        return os.path.exists(path)

    # ---------------------------------------------------------
    # LIST DIRECTORY
    # ---------------------------------------------------------
    if act == "list":
        directory = action.get("directory", ".")
        return os.listdir(directory)

    # ---------------------------------------------------------
    # SEARCH INSIDE FILE
    # ---------------------------------------------------------
    if act == "search":
        keyword = action.get("keyword")
        if not keyword:
            raise ValueError("Missing 'keyword' for search action.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        matches = [line.strip() for line in lines if keyword in line]
        return matches

    raise ValueError(f"Unsupported file action: {act}")
