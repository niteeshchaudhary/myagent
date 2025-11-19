import platform
import shutil


def get_os():
    """
    Return 'windows', 'mac', or 'linux'.
    """
    os_name = platform.system().lower()
    if "windows" in os_name:
        return "windows"
    if "darwin" in os_name:
        return "mac"
    if "linux" in os_name:
        return "linux"
    return "unknown"


def detect_package_manager():
    """
    Detect best available package manager for current OS.
    """
    os_type = get_os()

    if os_type == "windows":
        return "choco" if shutil.which("choco") else None

    if os_type == "mac":
        return "brew" if shutil.which("brew") else None

    if os_type == "linux":
        # Check order of preference
        if shutil.which("apt"):
            return "apt"
        if shutil.which("yum"):
            return "yum"
        if shutil.which("dnf"):
            return "dnf"
        if shutil.which("pacman"):
            return "pacman"
        return None

    return None


def get_shell():
    """
    Return default terminal shell.
    """
    os_type = get_os()

    if os_type == "windows":
        return "powershell" if shutil.which("powershell") else "cmd"

    # mac/linux:
    return os.environ.get("SHELL", "/bin/bash")
