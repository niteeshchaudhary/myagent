#include <iostream>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <limits.h>

using namespace std;

int main(int argc, char* argv[]) {
    // Get the executable path on Linux
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    
    if (len == -1) {
        cerr << "Error: Could not determine executable path" << endl;
        return 1;
    }
    
    exePath[len] = '\0';
    string path = exePath;

    // Remove the executable name to get just the directory path
    size_t lastSlash = path.find_last_of("/");
    if (lastSlash != string::npos) {
        path = path.substr(0, lastSlash + 1);  // Keep the trailing slash
    }

    // Path to the Python virtual environment's activation script
    string venvActivate = path + ".venv/bin/activate";
    string pythonScript = path + "cli/main.py";

    // Get the current working directory (where user is running the executable from)
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        cerr << "Error: Could not determine current working directory" << endl;
        return 1;
    }
    string userCwd = cwd;

    // Build command arguments string
    string args = "";
    for (int i = 1; i < argc; i++) {
        args += " '";
        args += argv[i];
        args += "'";
    }

    // Full command to activate the virtual environment, set PYTHONPATH, and run the Python script
    // Using bash -c to properly handle source and command chaining
    // Set PYTHONPATH to project root so 'agent' module can be found
    // Change to executable directory only to activate venv, then change back to user's CWD
    string command = "bash -c \"cd '" + path + "' && export PYTHONPATH='" + path + "' && source '" + venvActivate + "' && cd '" + userCwd + "' && python '" + pythonScript + "'" + args + "\"";

    // Execute the command
    int result = system(command.c_str());

    // Return the exit code from the Python script
    return result;
}
