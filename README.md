# LangGraph + Webots MCP

A small web agent with an explicit LangGraph model → tools → model loop.
It starts the existing Webots MCP server as a subprocess and keeps one stdio
session open throughout the conversation. Chat history is kept in memory until exit.

## Setup

```sh
python3 -m venv env  # Skip if env already exists.
source env/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Optionally change `OPENAI_MODEL`, `MCP_SERVER_PATH`,
or `MCP_PYTHON`. The default server is the sibling
`../webots-mcp/webots_mcp_server.py`, using the agent's Python interpreter.

## Run the web UI

```sh
langgraph dev
# Open http://127.0.0.1:8000
```

The browser UI keeps a chat transcript and supports multiple turns. The CLI remains
available with `python agent.py` (and `python agent.py --check` for an API-free check).

`--check` discovers tools and calls `list_robots` without an API key or robot movement.
Run Webots with its matching robot controller for live data and movement commands.
An MCP connection alone does not mean the simulator is running.

The local launcher redirects the server's startup debug prints to stderr so stdout
contains only MCP messages. The original server is not edited. Its log is written
to this project, and robot data remains in the server's existing `data` directory.

The server currently hardcodes one connection flag and returns camera file paths
rather than image content. The agent prompt accounts for these limitations; it does
not load camera images. Commands are requested one at a time because the server
uses a single command file per robot.

MCP integration follows the [LangChain MCP adapter documentation](https://github.com/langchain-ai/langchain-mcp-adapters).
