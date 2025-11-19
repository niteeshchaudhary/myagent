coding-agent/
├── agent/
│   ├── core/
│   │   ├── agent_loop.py
│   │   ├── planner.py
│   │   ├── memory.py
│   │   └── tool_manager.py
│   │
│   ├── tools/
│   │   ├── shell_tool.py
│   │   ├── python_tool.py
│   │   ├── file_tool.py
│   │   ├── git_tool.py
│   │   ├── installer_tool.py
│   │   ├── web_search_tool.py
│   │   └── os_tool.py
│   │
│   ├── llm/
│   │   ├── openai_llm.py
│   │   ├── groq_llm.py
│   │   ├── local_llm.py
│   │   └── model_selector.py
│   │
│   ├── rag/
│   │   ├── rg_search.py
│   │   ├── tags_client.py
│   │   ├── indexer.py
│   │   ├── retriever.py
│   │   ├── templates.py
│   │   └── rag_query.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── json_parser.py
│       ├── file_ref.py      ← handles @file references
│       └── os_detect.py
│
├── cli/
|   ├── main.py
|   └── __init__.py
|
├── configs/
│   ├── config.yaml
│   ├── tools.yaml
│   └── models.yaml
│
├── logs/
│   └── agent.log
│
├── tests/
│   ├── test_tools.py
│   ├── test_agent.py
│   └── test_llm.py
│
├── requirements.txt
├── README.md
└── setup.py OR pyproject.toml


📘 Coding Agent – README

A lightweight, extensible terminal-based coding agent that uses planning, memory, tools, and multiple LLM providers (OpenAI, Groq, Ollama, or local models) to execute tasks.

This project provides:

🔄 Agent loop (Planner → Tools → LLM → Memory)

🧠 Memory system (auto switch between Redis or in-memory backend)

🛠️ Tool Manager (auto-load tools from agent/tools/)

🤖 Multi-LLM support (OpenAI / Groq / Ollama / local)

🖥️ CLI interface (Typer-based, easy to run)

📦 Installation
1. Clone the repository
git clone <your-repo-url>
cd coding-agent

2. Install dependencies
pip install -r requirements.txt


Or if using poetry:

poetry install

3. (Optional) Install Redis for memory backend

If Redis is running at redis://localhost:6379/0, the agent automatically uses Redis-based memory.

Otherwise, it falls back to in-memory storage.

⚙️ Environment Configuration

The following environment variables control LLM providers:

Provider	Variable	Example
OpenAI API	OPENAI_API_KEY	export OPENAI_API_KEY="sk-..."
Groq API	GROQ_API_KEY	export GROQ_API_KEY="gsk_..."
Ollama local models	No key needed	ensure ollama is installed
Redis memory	REDIS_URL	export REDIS_URL="redis://localhost:6379/0"

Example:

export OPENAI_API_KEY="sk-xxxx"
export GROQ_API_KEY="gsk-xxxx"
export REDIS_URL="redis://localhost:6379/0"


You may also configure defaults in:

configs/config.yaml
configs/models.yaml
configs/tools.yaml

🚀 Running the CLI

The main CLI entrypoint lives in:

cli/main.py


Run it directly:

python -m cli.main run "Write a python script that prints prime numbers."


Or if installed via pip install -e ., run:

coding-agent run "Generate a hello world program."

🧰 CLI Commands
1️⃣ Run a single prompt
coding-agent run --provider openai --model gpt-4o "Build a flask server with 2 endpoints"


Options:

--provider, -p        (openai | groq | ollama | local)
--model, -m           Model name
--persist             Enable memory persistence (local JSON)
--persist-path        File path for memory storage
--max-memory          Max memory entries
--verbose/--no-verbose


Example:

coding-agent run -p groq -m mixtral-8x7b "Optimize this SQL query"

2️⃣ Interactive REPL mode

Start a persistent session with memory, tools, and streaming:

coding-agent repl -p ollama -m llama3


Inside REPL, you can talk to your agent continuously.

3️⃣ List all available tools
coding-agent list-tools


Example output:

[
  {"name": "shell", "module": "agent.tools.shell_tool", "class": "ShellTool"},
  {"name": "python", "module": "agent.tools.python_tool", "class": "PythonTool"},
  {"name": "file", "module": "agent.tools.file_tool", "class": "FileTool"}
]

4️⃣ Probe available LLM providers
coding-agent probe


Or probe a specific provider:

coding-agent probe -p groq


Example output:

{
  "openai": {"available": true, "msg": "API key OK"},
  "groq":   {"available": true, "msg": "API key OK"},
  "ollama": {"available": true, "msg": "Running locally"}
}

5️⃣ Inspect or clear memory

Show stored memory items:

coding-agent memory --show


Clear memory:

coding-agent memory --clear


Limit output:

coding-agent memory --show --limit 10

6️⃣ Show or select provider/model
coding-agent select


Select from preference order:

coding-agent select -p "groq,openai,ollama"

🧠 Memory Backend
Automatic selection:
Condition	Memory Used
Redis installed & reachable	RedisMemory
Else	InMemory (local)

No config changes are needed — it's automatic.

To force Redis:

export REDIS_URL="redis://localhost:6379/0"

🧱 Project Structure
coding-agent/
│
├── agent/
│   ├── core/ (agent loop, planner, memory, tool manager)
│   ├── tools/ (shell, python, git, file, installer, web search)
│   ├── llm/ (openai, ollama, groq, local)
│   └── utils/
│
├── cli/
│   └── main.py
│
├── configs/
├── logs/
├── tests/
└── README.md

🛠️ Development Setup
Install in editable mode:
pip install -e .

Run tests:
pytest -q

📌 Example Usage
Ask agent to generate and apply code changes:
coding-agent run "Add a new CLI option --dry-run to my Python tool"

Use shell + git tools to modify your repo:
coding-agent run "Create feature branch and update README"

Use local LLM via Ollama:
coding-agent run -p ollama -m codellama "Refactor these functions for readability."

🎉 You're all set!

Your coding agent is now ready to run with:

coding-agent repl

::coding-agent <- name depends on the object file you create after compiling c code
