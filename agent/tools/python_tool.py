from agent.utils.logger import log


class PythonTool:
    """
    Python code execution tool.
    
    Executes Python code safely inside an isolated scope.
    """
    
    name = "python"
    description = "Execute Python code safely in an isolated scope"
    
    def run(self, input: str, **kwargs) -> dict:
        """
        Execute Python code safely inside an isolated scope.
        
        Input:
            "x = 5\nresult = x * 10"
        
        Returns:
            Dict with 'output' key containing the result
        """
        if not isinstance(input, str):
            raise ValueError("Python tool expects a string of code.")

        log.info("[python] Executing code...")
        log.debug(input)

        # Safe empty environment
        local_vars = {}

        try:
            exec(input, {}, local_vars)
            # Format output - show all variables that were set
            if local_vars:
                output_lines = []
                for var_name, var_value in local_vars.items():
                    output_lines.append(f"{var_name} = {repr(var_value)}")
                output = "\n".join(output_lines)
            else:
                output = "Code executed successfully (no variables set)"
            
            return {"output": output, "success": True, "locals": local_vars}
        except Exception as e:
            error_msg = f"Python execution error: {e}"
            log.error(error_msg)
            return {"output": "", "success": False, "error": error_msg}
