import subprocess
import sys
import threading
import time
import queue
from typing import Optional, Dict, Any
from agent.utils.logger import log


class ShellTool:
    """
    Shell command execution tool with confirmation prompts.
    
    Supports executing shell commands with user confirmation.
    Handles long-running processes (like servers) by running them in background
    and monitoring output for success/failure indicators.
    """
    
    name = "shell"
    description = "Execute shell commands with user confirmation"
    
    # Commands that are typically long-running servers/processes
    LONG_RUNNING_KEYWORDS = [
        "npm start", "npm run dev", "npm run serve",
        "python -m http.server", "python server.py", "python app.py",
        "node server.js", "node app.js",
        "flask run", "gunicorn", "uvicorn",
        "rails server", "rails s",
        "php -S", "php artisan serve",
        "java -jar", "mvn spring-boot:run",
        "go run", "cargo run",
        "docker run", "docker-compose up",
        "serve", "http-server", "live-server"
    ]
    
    # Success indicators for long-running processes
    SUCCESS_INDICATORS = [
        "server running", "listening on", "listening at",
        "started on", "running on", "serving on",
        "ready", "compiled successfully", "compiled with",
        "webpack compiled", "vite", "development server",
        "application startup", "uvicorn running", "fastapi",
        "server is running", "server started"
    ]
    
    # Error indicators that suggest immediate failure
    ERROR_INDICATORS = [
        "error:", "failed", "cannot", "unable to",
        "missing", "not found", "eaddrinuse", "port already in use",
        "eacces", "permission denied", "enoent",
        "npm err!", "npm error", "command not found",
        "script:", "missing script"
    ]
    
    def _is_long_running(self, cmd: str) -> bool:
        """Check if a command is likely to be long-running."""
        cmd_lower = cmd.lower()
        return any(keyword in cmd_lower for keyword in self.LONG_RUNNING_KEYWORDS)
    
    def run(self, input_data, **kwargs):
        """
        Execute shell command with confirmation.
        
        Input can be:
        - A string command
        - A dict with 'command' key and optional 'background' or 'timeout' keys
        """
        # Extract command from input
        if isinstance(input_data, dict):
            cmd = input_data.get("command") or input_data.get("cmd") or str(input_data.get("input", ""))
            background = input_data.get("background", False)
            timeout = input_data.get("timeout", 10.0)  # Default 10 seconds for monitoring
        else:
            cmd = str(input_data)
            background = False
            timeout = 10.0
        
        if not cmd or not cmd.strip():
            raise ValueError("Shell tool expects a non-empty command.")
        
        # Ask for confirmation if running interactively
        if sys.stdin and sys.stdin.isatty():
            print(f"\n[Shell] Command: {cmd}")
            response = input("Would you like to execute this command? (Y/n): ").strip().lower()
            if response and response not in ('y', 'yes', ''):
                return {"output": "Command execution cancelled by user.", "cancelled": True}
        
        # Check if this is a long-running command
        is_long_running = background or self._is_long_running(cmd)
        
        # Execute the command
        try:
            if is_long_running:
                result = run_shell_background(cmd, timeout=timeout, 
                                            success_indicators=self.SUCCESS_INDICATORS,
                                            error_indicators=self.ERROR_INDICATORS)
            else:
                result = run_shell(cmd)
            
            # Return as dict for consistency
            if isinstance(result, dict):
                return result
            return {"output": result, "success": True}
        except Exception as e:
            return {"output": "", "success": False, "error": str(e)}


def run_shell(cmd: str) -> str:
    """
    Execute shell command using subprocess (for short-running commands).
    
    Input:
        "ls -la"
        
    Returns:
        Command output (stdout + stderr)
    
    Raises:
        Exception if command fails.
    """

    if not isinstance(cmd, str):
        raise ValueError("Shell tool expects a string command.")

    log.info(f"[shell] Running: {cmd}")

    # Always use shell=True for shell commands to support:
    # - Commands with arguments (e.g., "npm install react")
    # - Commands with pipes, redirects, etc.
    # - Commands that need PATH resolution
    # On Windows, shell=True is required for commands like dir, etc.
    # On Linux/Unix, shell=True is needed for commands with spaces/arguments
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise Exception(f"Shell command failed:\n{stderr.strip()}")

    return stdout.strip() or "(no output)"


def run_shell_background(cmd: str, timeout: float = 10.0, 
                        success_indicators: Optional[list] = None,
                        error_indicators: Optional[list] = None) -> Dict[str, Any]:
    """
    Execute a long-running shell command in the background and monitor output.
    
    This function:
    1. Starts the process in the background
    2. Monitors stdout/stderr for a short period (timeout seconds)
    3. Looks for success/error indicators
    4. Returns early if success is detected
    5. Returns failure if errors are detected quickly
    6. Otherwise returns success (assuming process started correctly)
    
    Args:
        cmd: Command to execute
        timeout: How long to monitor output (seconds)
        success_indicators: List of strings that indicate success
        error_indicators: List of strings that indicate failure
    
    Returns:
        Dict with 'output', 'success', 'process_id', and optional 'error'
    """
    if not isinstance(cmd, str):
        raise ValueError("Shell tool expects a string command.")
    
    if success_indicators is None:
        success_indicators = []
    if error_indicators is None:
        error_indicators = []
    
    log.info(f"[shell] Running long-running command in background: {cmd}")
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True
    )
    
    output_lines = []
    output_queue = queue.Queue()
    error_detected = False
    success_detected = False
    
    def read_output():
        """Read output from process in a separate thread."""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    output_lines.append(line)
                    output_queue.put(line)
                    log.debug(f"[shell output] {line}")
        except Exception as e:
            log.debug(f"[shell] Error reading output: {e}")
        finally:
            output_queue.put(None)  # Signal end of output
    
    # Start reading output in background thread
    reader_thread = threading.Thread(target=read_output, daemon=True)
    reader_thread.start()
    
    # Monitor output for the timeout period
    start_time = time.time()
    collected_output = []
    
    try:
        while time.time() - start_time < timeout:
            # Check if process has exited
            return_code = process.poll()
            if return_code is not None:
                # Process ended - collect any remaining output
                time.sleep(0.5)  # Give a moment for final output
                while not output_queue.empty():
                    try:
                        line = output_queue.get_nowait()
                        if line is not None:
                            collected_output.append(line)
                    except queue.Empty:
                        break
                
                if return_code != 0:
                    # Process exited with error
                    error_msg = "\n".join(collected_output) if collected_output else "Process exited with error"
                    raise Exception(f"Shell command failed:\n{error_msg}")
                else:
                    # Process exited successfully (unusual for long-running, but possible)
                    output_text = "\n".join(collected_output) if collected_output else "Process completed"
                    return {
                        "output": output_text,
                        "success": True,
                        "process_id": process.pid,
                        "background": False,
                        "message": "Process completed successfully"
                    }
            
            try:
                line = output_queue.get(timeout=0.5)
                if line is None:
                    # Reader thread ended, but process might still be running
                    # Wait a bit and check process status
                    time.sleep(0.5)
                    return_code = process.poll()
                    if return_code is not None and return_code != 0:
                        error_msg = "\n".join(collected_output) if collected_output else "Process exited with error"
                        raise Exception(f"Shell command failed:\n{error_msg}")
                    break
                
                collected_output.append(line)
                line_lower = line.lower()
                
                # Check for error indicators
                if any(indicator in line_lower for indicator in error_indicators):
                    error_detected = True
                    log.warning(f"[shell] Error indicator detected: {line}")
                    # Give it a moment to see if it's a fatal error
                    time.sleep(1.0)
                    # Check if process is still running
                    return_code = process.poll()
                    if return_code is not None:
                        # Process died, definitely an error
                        error_msg = "\n".join(collected_output) if collected_output else "Process failed"
                        raise Exception(f"Shell command failed:\n{error_msg}")
                
                # Check for success indicators
                if any(indicator in line_lower for indicator in success_indicators):
                    success_detected = True
                    log.info(f"[shell] Success indicator detected: {line}")
                    # Wait a bit more to see if any errors follow
                    time.sleep(2.0)
                    # Check one more time if process is still running
                    return_code = process.poll()
                    if return_code is not None and return_code != 0:
                        error_msg = "\n".join(collected_output) if collected_output else "Process failed after success indicator"
                        raise Exception(f"Shell command failed:\n{error_msg}")
                    break
                    
            except queue.Empty:
                # No output yet, continue monitoring
                continue
        
        # Final check: if process exited during monitoring, handle it
        return_code = process.poll()
        if return_code is not None and return_code != 0:
            error_msg = "\n".join(collected_output) if collected_output else "Process exited with error"
            raise Exception(f"Shell command failed:\n{error_msg}")
        
        # If we detected success or process is still running, consider it successful
        output_text = "\n".join(collected_output) if collected_output else "Process started successfully"
        
        if success_detected:
            output_text += "\n[Process is running in background]"
        elif return_code is None:
            # Process is still running but no success indicator - still consider it started
            output_text += "\n[Process is running in background - no success indicator detected]"
        
        return {
            "output": output_text,
            "success": True,
            "process_id": process.pid,
            "background": True,
            "message": "Process started successfully and is running in background"
        }
        
    except Exception as e:
        # Try to terminate the process if we're returning an error
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            try:
                process.kill()
            except:
                pass
        raise
