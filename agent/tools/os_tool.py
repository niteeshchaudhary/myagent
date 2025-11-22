# tools/os_tool.py

import os
import platform
from agent.utils.logger import get_logger

logger = get_logger(__name__)

class OSTool:

    @staticmethod
    def get_os():
        return platform.system().lower()

    @staticmethod
    def home_dir():
        return os.path.expanduser("~")

    @staticmethod
    def list_dir(path):
        try:
            return {
                "success": True,
                "files": os.listdir(path)
            }
        except Exception as e:
            logger.error(f"[OSTool] Error listing directory: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def file_exists(path):
        return os.path.exists(path)

    @staticmethod
    def env_var(key):
        return os.getenv(key)

    @staticmethod
    def set_env_var(key, value):
        os.environ[key] = value
        return {"success": True}
