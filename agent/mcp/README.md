# MCP (Model Context Protocol) Integration

This agent supports the Model Context Protocol (MCP), allowing it to connect to external MCP servers and use their tools.

## Configuration

MCP is configured in `configs/config.yaml`:

```yaml
mcp:
  enabled: true
  servers:
    - name: "filesystem"
      command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
      env:
        MCP_API_KEY: "your-key"
      timeout: 30
    - name: "github"
      command: ["npx", "-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: "your-token"
      timeout: 30
```

### Server Configuration Options

- `name`: Unique name for the server
- `command`: List of command arguments to start the server (stdio transport)
- `url`: HTTP URL for the server (HTTP transport - not yet fully implemented)
- `env`: Environment variables to pass to the server process
- `timeout`: Request timeout in seconds (default: 30)

## Available MCP Servers

Popular MCP servers you can use:

1. **Filesystem Server**: Access local filesystem
   ```yaml
   - name: "filesystem"
     command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
   ```

2. **GitHub Server**: Interact with GitHub repositories
   ```yaml
   - name: "github"
     command: ["npx", "-y", "@modelcontextprotocol/server-github"]
     env:
       GITHUB_PERSONAL_ACCESS_TOKEN: "ghp_..."
   ```

3. **SQLite Server**: Query SQLite databases
   ```yaml
   - name: "sqlite"
     command: ["npx", "-y", "@modelcontextprotocol/server-sqlite", "--db-path", "/path/to/db.sqlite"]
   ```

4. **PostgreSQL Server**: Query PostgreSQL databases
   ```yaml
   - name: "postgres"
     command: ["npx", "-y", "@modelcontextprotocol/server-postgres"]
     env:
       POSTGRES_CONNECTION_STRING: "postgresql://user:pass@localhost/db"
   ```

## Usage

### List MCP Tools

```bash
# List all tools including MCP tools
agent list-tools

# List only native tools (exclude MCP)
agent list-tools --no-mcp

# List MCP servers and their tools
agent list-mcp
```

### Using MCP Tools

MCP tools are automatically registered with the tool manager and can be used like any other tool. They are prefixed with `mcp_<server_name>_<tool_name>` to avoid naming conflicts.

Example:
- If a GitHub server provides a tool called `search_repositories`, it will be available as `mcp_github_search_repositories`

The agent can use these tools in its planning and execution:

```bash
agent run "Search for Python repositories on GitHub" --provider groq
```

## Troubleshooting

### Server Not Connecting

1. Check that the server command is correct and the executable is available
2. Verify environment variables are set correctly
3. Check logs for connection errors
4. Ensure the server supports stdio transport (most MCP servers do)

### Tools Not Appearing

1. Verify the server connected successfully (`agent list-mcp`)
2. Check that the server actually provides tools (some servers may not expose tools)
3. Look for errors in the agent logs

### Timeout Issues

Increase the timeout in the server configuration:

```yaml
- name: "slow-server"
  command: ["python", "-m", "slow_mcp_server"]
  timeout: 60  # Increase timeout
```

## Implementation Details

- **Transport**: Currently supports stdio (subprocess) transport
- **Protocol**: JSON-RPC 2.0 over newline-delimited JSON
- **Tool Wrapping**: MCP tools are wrapped to match the agent's tool interface
- **Error Handling**: Failures are logged but don't crash the agent

## Future Enhancements

- HTTP transport support
- SSE (Server-Sent Events) transport
- Resource access (not just tools)
- Prompts from MCP servers
- Better error recovery and reconnection


