# agent/mcp/client.py
"""
MCP (Model Context Protocol) client implementation.

Connects to MCP servers via stdio or HTTP and exposes their tools.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
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


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: Optional[List[str]] = None  # For stdio servers: ["python", "-m", "mcp_server"]
    url: Optional[str] = None  # For HTTP servers: "http://localhost:8000"
    env: Optional[Dict[str, str]] = None
    timeout: int = 30


class MCPClient:
    """
    Client for connecting to MCP servers.
    
    Supports both stdio (subprocess) and HTTP transport.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to the MCP server. Returns True if successful."""
        try:
            if self.config.command:
                # stdio transport
                env = dict(os.environ)
                if self.config.env:
                    env.update(self.config.env)
                
                self.process = subprocess.Popen(
                    self.config.command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    bufsize=0  # Unbuffered
                )
                
                # Send initialize request
                init_request = {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "clientInfo": {
                            "name": "myagent",
                            "version": "1.0.0"
                        }
                    }
                }
                
                self._send_request(init_request)
                response = self._read_response(timeout=10.0)
                
                if response and response.get("result"):
                    # Send initialized notification
                    initialized = {
                        "jsonrpc": "2.0",
                        "method": "notifications/initialized"
                    }
                    self._send_request(initialized)
                    
                    # Wait a bit for server to be ready
                    time.sleep(0.5)
                    
                    # List available tools
                    self._list_tools()
                    self._connected = True
                    logger.info("Connected to MCP server: %s", self.config.name)
                    return True
                else:
                    error_msg = response.get("error", {}).get("message", "Unknown error") if response else "No response"
                    logger.error("MCP server %s initialization failed: %s", self.config.name, error_msg)
                    return False
            elif self.config.url:
                # HTTP transport (simplified - would need requests library)
                logger.warning("HTTP transport not yet implemented for MCP")
                return False
            else:
                logger.error("MCP server config missing both command and url")
                return False
        except Exception as e:
            logger.exception("Failed to connect to MCP server %s: %s", self.config.name, e)
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            finally:
                self.process = None
        self._connected = False
    
    def _next_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id
    
    def _send_request(self, request: Dict[str, Any]) -> None:
        """Send a JSON-RPC request to the server."""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Not connected to MCP server")
        
        line = json.dumps(request) + "\n"
        self.process.stdin.write(line)
        self.process.stdin.flush()
        logger.debug("Sent MCP request: %s", request.get("method"))
    
    def _read_response(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Read a JSON-RPC response from the server."""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Not connected to MCP server")
        
        # Check if process is still alive
        if self.process.poll() is not None:
            stderr_output = ""
            if self.process.stderr:
                try:
                    stderr_output = self.process.stderr.read()
                except Exception:
                    pass
            logger.error("MCP server process exited with code %d. stderr: %s", self.process.returncode, stderr_output[:200])
            return None
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if process died
            if self.process.poll() is not None:
                logger.error("MCP server process died while waiting for response")
                return None
            
            line = self.process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            line = line.strip()
            if not line:
                continue
            
            try:
                response = json.loads(line)
                logger.debug("Received MCP response: %s", response.get("method") or response.get("id") or "response")
                return response
            except json.JSONDecodeError as e:
                logger.debug("Failed to parse MCP response: %s (error: %s)", line[:100], e)
                continue
        
        logger.warning("Timeout waiting for MCP response after %.1fs", timeout)
        return None
    
    def _list_tools(self) -> None:
        """List available tools from the MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list"
        }
        
        self._send_request(request)
        response = self._read_response()
        
        if response and response.get("result"):
            tools = response["result"].get("tools", [])
            for tool in tools:
                name = tool.get("name")
                if name:
                    self._tools[name] = tool
                    logger.info("Discovered MCP tool: %s", name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return list(self._tools.values())
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Dict with 'content' (list of text/image items) and 'isError' flag
        """
        if not self._connected:
            raise RuntimeError("Not connected to MCP server")
        
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found on MCP server")
        
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments or {}
            }
        }
        
        self._send_request(request)
        response = self._read_response(timeout=self.config.timeout)
        
        if not response:
            return {"content": [{"type": "text", "text": "No response from MCP server"}], "isError": True}
        
        if "error" in response:
            error = response["error"]
            return {
                "content": [{"type": "text", "text": f"MCP error: {error.get('message', 'Unknown error')}"}],
                "isError": True
            }
        
        result = response.get("result", {})
        return result
    
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        return self._connected and self.process and self.process.poll() is None


class MCPToolWrapper:
    """
    Wrapper that makes MCP tools compatible with the agent's tool interface.
    """
    
    def __init__(self, client: MCPClient, tool_name: str, tool_spec: Dict[str, Any]):
        self.client = client
        self.name = tool_name
        self.spec = tool_spec
        self.description = tool_spec.get("description", "")
    
    def run(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the MCP tool.
        
        Args:
            input_data: Can be a dict (tool arguments) or string (will try to parse as JSON)
            **kwargs: Additional arguments merged with input_data if it's a dict
        """
        # Parse input
        if isinstance(input_data, dict):
            arguments = {**input_data, **kwargs}
        elif isinstance(input_data, str):
            try:
                arguments = json.loads(input_data)
                if isinstance(arguments, dict):
                    arguments.update(kwargs)
                else:
                    arguments = {"input": input_data, **kwargs}
            except json.JSONDecodeError:
                arguments = {"input": input_data, **kwargs}
        else:
            arguments = {"input": str(input_data), **kwargs}
        
        # Call MCP tool
        try:
            result = self.client.call_tool(self.name, arguments)
            
            # Extract text content from result
            content_items = result.get("content", [])
            text_parts = []
            for item in content_items:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        text_parts.append(f"[Image: {item.get('data', '')[:50]}...]")
                elif isinstance(item, str):
                    text_parts.append(item)
            
            output_text = "\n".join(text_parts) if text_parts else str(result)
            is_error = result.get("isError", False)
            
            return {
                "output": output_text,
                "success": not is_error,
                "error": result.get("error") if is_error else None,
                "raw": result
            }
        except Exception as e:
            logger.exception("MCP tool '%s' execution failed: %s", self.name, e)
            return {
                "output": "",
                "success": False,
                "error": str(e)
            }

