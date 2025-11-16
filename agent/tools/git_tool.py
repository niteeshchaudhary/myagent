# tools/git_tool.py

import subprocess
from utils.logger import logger

class GitTool:

    @staticmethod
    def run_git_command(args, cwd=None):
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "exit_code": result.returncode
            }
        except Exception as e:
            logger.error(f"[GitTool] Error running git command: {e}")
            return {"success": False, "error": str(e)}

    # -------------------------------
    # Git high-level helper functions
    # -------------------------------

    @staticmethod
    def clone(url, dest=None):
        args = ["clone", url]
        if dest:
            args.append(dest)
        return GitTool.run_git_command(args)

    @staticmethod
    def pull(cwd=None):
        return GitTool.run_git_command(["pull"], cwd=cwd)

    @staticmethod
    def checkout(branch, cwd=None):
        return GitTool.run_git_command(["checkout", branch], cwd=cwd)

    @staticmethod
    def get_status(cwd=None):
        return GitTool.run_git_command(["status", "--porcelain"], cwd=cwd)

    @staticmethod
    def init_repo(cwd=None):
        return GitTool.run_git_command(["init"], cwd=cwd)
