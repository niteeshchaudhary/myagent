import traceback
from typing import Callable, Dict, Any
from agent.utils.logger import log


class ToolError(Exception):
    """Custom error for tool failures."""
    pass


class ToolManager:
    """
    Manages all agent tools.
    
    Features:
    - Register tools dynamically
    - Execute tools safely
    - Validate inputs
    - Logs each tool call
    """

    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    # ------------------------------------------------------
    # Tool Registration
    # ------------------------------------------------------
    def register(self, name: str, func: Callable):
        """
        Register a tool with a name.
        Example:
            tm.register("shell", run_shell)
        """
        if name in self.tools:
            log.warning(f"Tool '{name}' is already registered. Overwriting.")

        self.tools[name] = func
        log.info(f"Registered tool: {name}")

    # ------------------------------------------------------
    # Tool Execution
    # ------------------------------------------------------
    def execute(self, tool_name: str, tool_input: Any):
        """
        Execute a tool with structured error handling.
        Returns:
            {
                "success": bool,
                "output": str or data,
                "error": str (if any)
            }
        """

        log.info(f"Executing tool: {tool_name}")
        log.debug(f"Input: {tool_input}")

        if tool_name not in self.tools:
            error_msg = f"Tool '{tool_name}' not found."
            log.error(error_msg)
            return {
                "success": False,
                "output": None,
                "error": error_msg
            }

        try:
            result = self.tools[tool_name](tool_input)

            log.debug(f"Output from '{tool_name}': {result}")

            return {
                "success": True,
                "output": result,
                "error": None
            }

        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Error executing tool '{tool_name}': {e}")
            log.debug(tb)

            return {
                "success": False,
                "output": None,
                "error": str(e)
            }

    # ------------------------------------------------------
    # Utility Functions
    # ------------------------------------------------------
    def list_tools(self):
        """Return list of all registered tool names."""
        return list(self.tools.keys())

    def describe_tools(self):
        """Return a dictionary of tool names & docstrings."""
        desc = {}
        for name, func in self.tools.items():
            desc[name] = func.__doc__ or "No description."
        return desc


# Singleton Instance (Used everywhere)
tool_manager = ToolManager()
