# MCPMess

MCPMess is a Model Context Protocol (MCP) server that provides access to chat message history stored in the Recall API.

## What it does

This server exposes two tools that can be used by MCP clients:

- `get_chat_history()` - Retrieve chat messages with filtering options
- `search_messages()` - Search through message content for specific text

## Requirements

- The Recall API server must be running
- Python dependencies are managed by the parent project (uv sync)

## Usage

MCP clients spawn this server automatically when needed.

### Tools

**get_chat_history()**
- Filter by author, server, channel, or tag
- Time-based filtering (hours_ago, since/until dates)
- Pagination support

**search_messages()**
- Full-text search through message content and summaries
- Case-insensitive matching
- Optional filtering by author, server, channel, time

## Configuration

The server reads connection settings from the Recall API's settings file. Make sure the Recall server is running before using this MCP server.

## Testing

You can test the functions directly:

```bash
cd history/mcpmess
uv run python -c "
import asyncio
from main import search_messages
result = asyncio.run(search_messages('foo'))
print(result)
"
```

## MCP Client Setup

Configure your MCP client to run:
```bash
cd /path/to/sandy/history/mcpmess && uv run python main.py
```