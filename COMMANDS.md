# Complete Command Reference

This document lists all commands available in the codebase, organized by category.

## Table of Contents
1. [CLI Commands](#cli-commands)
2. [Tool Commands](#tool-commands)
3. [MCP Tools](#mcp-tools)

---

## CLI Commands

These are the main commands available through the CLI interface (`cli/main.py`).

### 1. `run`
Run a single prompt through the agent loop.

**Usage:**
```bash
coding-agent run [OPTIONS] --prompt TEXT
```

**Options:**
- `--prompt, -p TEXT` - Prompt to run through the agent loop (required)
- `--provider, -p TEXT` - LLM provider name (e.g. 'openai', 'ollama', 'groq')
- `--model, -m TEXT` - Model name to use for the LLM
- `--persist-path TEXT` - Path to persist memory JSON
- `--persist` - Enable memory persistence to file
- `--max-memory INTEGER` - Cap memory to this many items
- `--verbose / --no-verbose` - Show streaming output when available (default: True)
- `--review / --no-review` - Enable review mode: changes require approval before applying

**Example:**
```bash
coding-agent run --provider openai --model gpt-4o "Build a flask server with 2 endpoints"
```

---

### 2. `repl`
Start an interactive REPL agent session.

**Usage:**
```bash
coding-agent repl [OPTIONS]
```

**Options:**
- `--provider, -p TEXT` - LLM provider name
- `--model, -m TEXT` - Model name
- `--persist-path TEXT` - Path to persist memory JSON
- `--persist` - Enable memory persistence to file
- `--max-memory INTEGER` - Cap memory to this many items
- `--review / --no-review` - Enable review mode

**Example:**
```bash
coding-agent repl -p ollama -m llama3
```

---

### 3. `probe`
Probe LLM providers and show which are available.

**Usage:**
```bash
coding-agent probe [OPTIONS]
```

**Options:**
- `--provider, -p TEXT` - Provider name to probe. If omitted, probes all supported providers

**Example:**
```bash
coding-agent probe
coding-agent probe -p groq
```

---

### 4. `list-tools`
Discover and list available tools from agent.tools package and MCP servers.

**Usage:**
```bash
coding-agent list-tools [OPTIONS]
```

**Options:**
- `--mcp / --no-mcp` - Include MCP tools in the list (default: True)

**Example:**
```bash
coding-agent list-tools
coding-agent list-tools --no-mcp
```

---

### 5. `list-mcp`
List connected MCP servers and their available tools.

**Usage:**
```bash
coding-agent list-mcp
```

**Example:**
```bash
coding-agent list-mcp
```

---

### 6. `memory`
Inspect or clear the agent memory.

**Usage:**
```bash
coding-agent memory [OPTIONS]
```

**Options:**
- `--show / --no-show` - Show memory contents (default: True)
- `--clear` - Clear memory
- `--persist-path TEXT` - Path used for memory persistence
- `--persist` - Enable persistence for the memory backend
- `--limit INTEGER` - Max records to display when showing memory (default: 25)

**Example:**
```bash
coding-agent memory --show
coding-agent memory --clear
coding-agent memory --show --limit 10
```

---

### 7. `ask`
Ask a question about the codebase using RAG (Retrieval-Augmented Generation).

**Usage:**
```bash
coding-agent ask [OPTIONS] QUESTION
```

**Options:**
- `QUESTION` - Question to ask about the codebase (required)
- `--provider, -p TEXT` - LLM provider name (e.g. 'openai', 'ollama', 'groq')
- `--model, -m TEXT` - Model name to use for the LLM
- `--k INTEGER` - Number of code chunks to retrieve (default: 8)
- `--exact / --semantic` - Prefer exact search over semantic search (default: False)
- `--no-llm` - Only retrieve chunks, don't call LLM (show retrieved context)
- `--stream` - Stream LLM output

**Example:**
```bash
coding-agent ask "How does authentication work?"
coding-agent ask "Where is the main entry point?" --exact
coding-agent ask "What tools are available?" --k 5
coding-agent ask "Show me the config loader" --no-llm
```

---

### 8. `index`
Index the codebase for RAG (Retrieval-Augmented Generation).

**Usage:**
```bash
coding-agent index [OPTIONS]
```

**Options:**
- `--force` - Force reindexing even if index exists
- `--changed` - Index only changed files (requires git)
- `--repo TEXT` - Repository root directory (default: current directory)

**Example:**
```bash
coding-agent index
coding-agent index --force
coding-agent index --changed
```

---

### 9. `select`
Show provider/model selection info. If provider list supplied, shows first available provider from the list.

**Usage:**
```bash
coding-agent select [OPTIONS]
```

**Options:**
- `--provider, -p TEXT` - Provider preference (comma separated) or blank to show defaults
- `--model, -m TEXT` - Model name override to display

**Example:**
```bash
coding-agent select
coding-agent select -p "groq,openai,ollama"
```

---

## Tool Commands

These are the tools that the agent can use internally. Each tool has a `run()` method that accepts specific input formats.

### 1. `shell` (ShellTool)
Execute shell commands with user confirmation.

**Tool Name:** `shell`

**Input Format:**
- String: Direct command to execute
- Dict:
  ```python
  {
    "command": "ls -la",  # or "cmd"
    "background": False,  # Optional: run in background
    "timeout": 60         # Optional: timeout in seconds
  }
  ```

**Features:**
- User confirmation prompts
- Long-running process detection
- Interactive command detection
- Background execution support
- Command blocking (configurable)

**Example Usage:**
```python
tool_manager.run("shell", input="ls -la")
tool_manager.run("shell", input={"command": "npm start", "background": True})
```

---

### 2. `file` / `file_editor` (FileTool / FileEditorTool)
File operations tool for reading, writing, and managing files.

**Tool Name:** `file` or `file_editor`

**Supported Actions:**
- `read` - Read file contents
- `write` - Write content to file
- `append` - Append content to file
- `delete` - Delete a file
- `exists` - Check if file exists
- `list` - List directory contents
- `search` - Search for keywords in file

**Input Format:**
```python
{
  "action": "read",  # Required: one of the actions above
  "path": "file.txt",  # Required for: read, write, append, delete, search
  "content": "...",  # Required for: write, append
  "keyword": "...",  # Required for: search
  "directory": "."  # Optional for: list (default: ".")
}
```

**Example Usage:**
```python
# Read file
tool_manager.run("file", input={"action": "read", "path": "test.txt"})

# Write file
tool_manager.run("file", input={"action": "write", "path": "test.txt", "content": "Hello"})

# List directory
tool_manager.run("file", input={"action": "list", "directory": "/path/to/dir"})
```

---

### 3. `python` (PythonTool)
Execute Python code safely in an isolated scope.

**Tool Name:** `python`

**Input Format:**
- String: Python code to execute

**Example Usage:**
```python
tool_manager.run("python", input="x = 5\nresult = x * 10")
```

**Returns:**
```python
{
  "output": "x = 5\nresult = 50",
  "success": True,
  "locals": {"x": 5, "result": 50}
}
```

---

### 4. `git` (GitTool)
Git operations tool.

**Tool Name:** `git`

**Available Methods:**
- `run_git_command(args, cwd=None)` - Run arbitrary git command
- `clone(url, dest=None)` - Clone a repository
- `pull(cwd=None)` - Pull latest changes
- `checkout(branch, cwd=None)` - Checkout a branch
- `get_status(cwd=None)` - Get git status
- `init_repo(cwd=None)` - Initialize a git repository

**Input Format:**
The tool accepts git command arguments as input.

**Example Usage:**
```python
# Clone repository
GitTool.clone("https://github.com/user/repo.git", dest="./repo")

# Get status
GitTool.get_status(cwd="./repo")

# Run custom git command
GitTool.run_git_command(["log", "--oneline", "-5"])
```

---

### 5. `ci` (CITool)
Run linters and tests; optionally rollback and request LLM patches and apply them.

**Tool Name:** `ci`

**Input Format:**
```python
{
  "paths": ".",  # Path(s) to check (default: repo root)
  "use_pylint": False,  # Use pylint (default: False)
  "pytest_args": None,  # Optional pytest args string
  "attempt_fix": False,  # Attempt to call LLM to fix issues
  "llm": llm_instance,  # LLM adapter instance (required if attempt_fix=True)
  "auto_rollback": True,  # Auto rollback on failure (default: True)
  "commit_message": "ci: apply linter/test fixes (LLM)",  # Commit message for auto-fix
  "max_fix_attempts": 2,  # Maximum fix attempts
  "repo_path": ".",  # Repository root path
  "python_bin": "python"  # Python binary to use
}
```

**Example Usage:**
```python
tool_manager.run("ci", input={
  "paths": ".",
  "attempt_fix": True,
  "use_pylint": False,
  "apply_patch": True
})
```

---

### 6. `install` (InstallerTool)
Install packages using system package managers.

**Tool Name:** `install`

**Available Methods:**
- `run_install(cmd_list)` - Run installation command list
- `install_package(pkg)` - Install a package (auto-detects OS)

**OS-Specific Commands:**
- **Windows:** `choco install <pkg> -y`
- **macOS:** `brew install <pkg>`
- **Linux:** `sudo apt install <pkg> -y`

**Example Usage:**
```python
# Install package (auto-detects OS)
InstallerTool.install_package("python3")

# Run custom install command
InstallerTool.run_install(["pip", "install", "requests"])
```

---

### 7. `web_search` (WebSearchTool)
Perform web searches using Selenium.

**Tool Name:** `web_search`

**Available Methods:**
- `google_search(query)` - Perform Google search and return results

**Example Usage:**
```python
result = WebSearchTool.google_search("python programming")
# Returns: {"success": True, "results": [{"title": "...", "url": "..."}, ...]}
```

---

### 8. `os` (OSTool)
OS-level operations.

**Tool Name:** `os`

**Available Methods:**
- `get_os()` - Get operating system name
- `home_dir()` - Get home directory path
- `list_dir(path)` - List directory contents
- `file_exists(path)` - Check if file exists
- `env_var(key)` - Get environment variable
- `set_env_var(key, value)` - Set environment variable

**Example Usage:**
```python
OSTool.get_os()  # Returns: "linux", "windows", "darwin", etc.
OSTool.home_dir()  # Returns: "/home/user"
OSTool.list_dir("/path/to/dir")  # Returns: {"success": True, "files": [...]}
```

---

## MCP Tools

MCP (Model Context Protocol) tools are dynamically loaded from configured MCP servers. These tools are prefixed with the server name to avoid collisions.

**Tool Naming Convention:**
```
mcp_{server_name}_{tool_name}
```

**Listing MCP Tools:**
```bash
coding-agent list-mcp
coding-agent list-tools
```

**Example:**
If an MCP server named "filesystem" provides a tool called "read_file", it would be registered as:
```
mcp_filesystem_read_file
```

**Configuration:**
MCP servers are configured in `configs/config.yaml` under the `mcp.servers` section.

---

## Internal Agent Commands

These are methods available within the agent core that can be called programmatically.

### AgentLoop Methods
- `run_once(prompt: str)` - Run a single prompt through the agent loop
- `run_interactive()` - Start interactive REPL loop
- `run_todo(prompt: str, *, confirm: bool = True, retry_on_fail: int = 1, interactive_confirm: Callable = None)` - Run with TODO list support

### ToolManager Methods
- `run(name: str, *, input: Any = None, timeout: Optional[float] = None, **kwargs)` - Execute a tool
- `run_all(tasks: List[Dict[str, Any]])` - Run multiple tools in parallel
- `list_tools()` - List all registered tools
- `get_tool(name: str)` - Get a tool instance
- `register_tool(name: str, instance: Any, ...)` - Register a new tool
- `unregister_tool(name: str)` - Remove a tool

### CodeValidator Methods
- `run_linters(paths: Optional[str] = None, use_pylint: bool = False)` - Run linters
- `run_tests(pytest_args: Optional[str] = None)` - Run tests
- `validate_and_fix(...)` - Validate code and optionally fix issues

---

## Summary

### CLI Commands (9 total)
1. `run` - Single-shot prompt execution
2. `repl` - Interactive REPL session
3. `probe` - Probe LLM providers
4. `list-tools` - List available tools
5. `list-mcp` - List MCP servers and tools
6. `memory` - Inspect/clear memory
7. `ask` - RAG-based codebase questions
8. `index` - Index codebase for RAG
9. `select` - Show/select LLM provider/model

### Native Tools (8 total)
1. `shell` - Shell command execution
2. `file` / `file_editor` - File operations
3. `python` - Python code execution
4. `git` - Git operations
5. `ci` - CI/linting/testing
6. `install` - Package installation
7. `web_search` - Web search
8. `os` - OS operations

### MCP Tools
- Dynamic (depends on configured MCP servers)
- Prefixed with `mcp_{server_name}_`

---

*Last updated: Generated from codebase analysis*

