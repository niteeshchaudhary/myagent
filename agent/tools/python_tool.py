from agent.utils.logger import log


def run_python(code: str):
    """
    Execute Python code safely inside an isolated scope.
    
    Input:
        "x = 5\nresult = x * 10"
    
    Returns:
        Locals dictionary containing all variables after execution
    
    Raises:
        Exception on syntax/runtime errors.
    """

    if not isinstance(code, str):
        raise ValueError("Python tool expects a string of code.")

    log.info("[python] Executing code...")
    log.debug(code)

    # Safe empty environment
    local_vars = {}

    try:
        exec(code, {}, local_vars)
    except Exception as e:
        raise Exception(f"Python execution error: {e}")

    return local_vars
