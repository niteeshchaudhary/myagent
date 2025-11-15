# MiAgent - CLI Code Generation Agent

A powerful CLI agent that uses LLM API calls to write code, create projects, and implement features - similar to Cursor/Windsurf functionality.

## Features

- 🤖 **LLM Integration**: Supports OpenAI (GPT-4) and Anthropic (Claude) APIs
- 📁 **File Management**: Automatically creates and manages project files
- 💬 **Interactive Mode**: Have conversations with the agent for iterative development
- 🎯 **Context Awareness**: Understands your project structure and existing code
- 🔧 **Project Generation**: Creates complete project structures with all necessary files

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key:
```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# For Anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

### Interactive Mode (Recommended)

Start an interactive session where you can have a conversation with the agent:

```bash
python MiAgent.py --interactive
```

In interactive mode, you can:
- Type tasks and get code generated
- Have back-and-forth conversations
- Type `clear` to reset conversation history
- Type `context` to see current project structure
- Type `exit` or `quit` to exit

### Single Task Mode

Execute a single task:

```bash
python MiAgent.py "Create a Python web scraper for news articles"
```

### Advanced Options

```bash
# Use Anthropic Claude instead of OpenAI
python MiAgent.py --provider anthropic "Create a REST API with FastAPI"

# Specify a custom model
python MiAgent.py --provider openai --model gpt-4-turbo "Build a React component"

# Work in a specific directory
python MiAgent.py --path ./myproject "Initialize a Node.js project"

# Use API key from command line
python MiAgent.py --api-key sk-... "Create a Python package"
```

## Examples

### Example 1: Create a Simple Project

```bash
python MiAgent.py "Create a Python Flask web application with a hello world route"
```

### Example 2: Build a Complete Project

```bash
python MiAgent.py --interactive
# Then in the interactive session:
💬 You: Create a todo list application with React
💬 You: Add authentication functionality
💬 You: Add database integration with PostgreSQL
```

### Example 3: Add Features to Existing Project

```bash
cd my-existing-project
python MiAgent.py --interactive
💬 You: Add error handling to all API endpoints
💬 You: Create unit tests for the main module
```

## How It Works

1. **Task Input**: You provide a task description (what you want to build/create)
2. **Context Gathering**: The agent reads your current project structure
3. **LLM Processing**: Your task is sent to the LLM with project context
4. **Code Generation**: The LLM generates code and file structures
5. **File Creation**: The agent automatically creates the files in your project

## Supported Providers

- **OpenAI**: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku

## Tips

1. **Be Specific**: The more details you provide, the better the output
   - ✅ Good: "Create a REST API with FastAPI that handles user authentication using JWT tokens"
   - ❌ Less Good: "Make an API"

2. **Use Interactive Mode**: For complex projects, use interactive mode to iterate
3. **Review Generated Code**: Always review and test the generated code
4. **Provide Context**: Mention frameworks, libraries, or patterns you want to use

## Troubleshooting

### API Key Issues
- Make sure your API key is set correctly
- Check that you have credits/quota available
- Verify the API key has the correct permissions

### File Creation Issues
- Ensure you have write permissions in the target directory
- Check that file paths don't conflict with existing files

### LLM Response Issues
- Try rephrasing your request
- Break complex tasks into smaller steps
- Use interactive mode for clarification

## License

This project is part of your MyIdea workspace.

