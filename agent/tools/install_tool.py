# tools/installer_tool.py

import subprocess
from agent.utils.logger import get_logger
from agent.utils.os_detect import get_os

logger = get_logger(__name__)

class InstallerTool:

    @staticmethod
    def run_install(cmd_list):
        """Runs installation command list."""
        try:
            result = subprocess.run(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "exit_code": result.returncode
            }
        except Exception as e:
            logger.error(f"[InstallerTool] Installation error: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def install_package(pkg):
        os_type = get_os()

        if os_type == "windows":
            cmd = ["choco", "install", pkg, "-y"]

        elif os_type == "macos":
            cmd = ["brew", "install", pkg]

        elif os_type == "linux":
            cmd = ["sudo", "apt", "install", pkg, "-y"]

        else:
            return {"success": False, "error": "Unsupported OS"}

        logger.info(f"[InstallerTool] Installing {pkg} using: {cmd}")
        return InstallerTool.run_install(cmd)
