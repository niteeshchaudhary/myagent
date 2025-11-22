# agent/mcp/manager.py
"""
MCP Manager for managing multiple MCP server connections and tools.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

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

from agent.mcp.client import MCPClient, MCPServerConfig, MCPToolWrapper


class MCPManager:
    """
    Manages multiple MCP server connections and their tools.
    """
    
    def __init__(self, server_configs: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize MCP manager.
        
        Args:
            server_configs: List of server configuration dicts, each with:
                - name: Server name
                - command: List of command args for stdio transport
                - url: URL for HTTP transport
                - env: Optional environment variables
                - timeout: Request timeout in seconds
        """
        self._clients: Dict[str, MCPClient] = {}
        self._tools: Dict[str, MCPToolWrapper] = {}
        self._server_configs = server_configs or []
    
    def connect_all(self) -> int:
        """
        Connect to all configured MCP servers.
        
        Returns:
            Number of successfully connected servers
        """
        connected = 0
        for config_dict in self._server_configs:
            try:
                config = MCPServerConfig(
                    name=config_dict.get("name", "unknown"),
                    command=config_dict.get("command"),
                    url=config_dict.get("url"),
                    env=config_dict.get("env"),
                    timeout=config_dict.get("timeout", 30)
                )
                
                client = MCPClient(config)
                if client.connect():
                    self._clients[config.name] = client
                    # Register tools from this server
                    self._register_tools(client)
                    connected += 1
                else:
                    logger.warning("Failed to connect to MCP server: %s", config.name)
            except Exception as e:
                logger.exception("Error connecting to MCP server %s: %s", config_dict.get("name"), e)
        
        logger.info("Connected to %d/%d MCP servers", connected, len(self._server_configs))
        return connected
    
    def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for client in self._clients.values():
            try:
                client.disconnect()
            except Exception:
                pass
        self._clients.clear()
        self._tools.clear()
    
    def _register_tools(self, client: MCPClient) -> None:
        """Register tools from an MCP client."""
        tools = client.list_tools()
        for tool_spec in tools:
            tool_name = tool_spec.get("name")
            if not tool_name:
                continue
            
            # Prefix tool name with server name to avoid collisions
            prefixed_name = f"mcp_{client.config.name}_{tool_name}"
            
            wrapper = MCPToolWrapper(client, tool_name, tool_spec)
            self._tools[prefixed_name] = wrapper
            logger.info("Registered MCP tool: %s (from server %s)", prefixed_name, client.config.name)
    
    def get_tool(self, name: str) -> Optional[MCPToolWrapper]:
        """Get an MCP tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[MCPToolWrapper]:
        """List all registered MCP tools."""
        return list(self._tools.values())
    
    def list_servers(self) -> List[str]:
        """List names of connected servers."""
        return list(self._clients.keys())
    
    def __enter__(self):
        """Context manager entry."""
        self.connect_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect_all()


