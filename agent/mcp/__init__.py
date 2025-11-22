# agent/mcp/__init__.py
"""
Model Context Protocol (MCP) integration for the agent.

MCP allows the agent to connect to external MCP servers and use their tools.
"""

from agent.mcp.client import MCPClient, MCPToolWrapper, MCPServerConfig
from agent.mcp.manager import MCPManager

__all__ = ["MCPClient", "MCPToolWrapper", "MCPServerConfig", "MCPManager"]

