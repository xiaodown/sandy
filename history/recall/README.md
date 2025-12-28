# Recall API Server

This directory contains a FastAPI-based server for storing and retrieving chat messages.

## Features

- **RESTful API** with automatic documentation (Swagger UI)
- **SQLite database** for reliable storage
- **Data validation** using Pydantic models
- **Filtering and pagination** for message retrieval
- **CORS support** for web clients

## Requirements

Install the required packages:

```bash
uv pip install fastapi uvicorn pydantic
```

## Running the Server

From the server directory:

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start on http://localhost:8000

## API Endpoints

- `GET /` - API information
- `GET /docs` - Swagger UI documentation
- `POST /messages/` - Create a new message
- `GET /messages/` - Get messages (with filtering and pagination)
  - `?author=username` - Filter by author
  - `?tag=tagname` - Filter by tag
  - `?hours_ago=N` - Messages from last N hours
  - `?minutes_ago=N` - Messages from last N minutes  
  - `?since=ISO_DATE` - Messages since specific datetime
  - `?until=ISO_DATE` - Messages until specific datetime
  - `?limit=N` - Limit number of results (default: 100)
  - `?offset=N` - Skip N messages for pagination
- `GET /messages/{id}` - Get a specific message
- `DELETE /messages/{id}` - Delete a message
- `GET /stats/` - Get database statistics

## Example Usage

### Creating a message:
```bash
curl -X POST "http://localhost:8000/messages/" \
  -H "Content-Type: application/json" \
  -d '{
    "author": "alice",
    "content": "Hello world!",
    "tags": ["greeting", "test"],
    "summary": "A simple greeting"
  }'
```

### Getting messages:
```bash
curl "http://localhost:8000/messages/"
```

### Filtering messages:
```bash
curl "http://localhost:8000/messages/?author=alice&limit=10"
```

### Time range filtering:
```bash
# Get messages from the last 24 hours
curl "http://localhost:8000/messages/?hours_ago=24"

# Get messages from the last 30 minutes
curl "http://localhost:8000/messages/?minutes_ago=30"

# Get messages between specific dates
curl "http://localhost:8000/messages/?since=2023-12-28T10:00:00&until=2023-12-28T18:00:00"

# Combine filters (e.g., Alice's messages from last hour)
curl "http://localhost:8000/messages/?author=alice&hours_ago=1"
```

## Database Schema

The SQLite database stores messages with the following fields:
- `id`: Auto-incrementing primary key
- `author`: Message author (required)
- `content`: Message content (required)
- `timestamp`: When the message was stored
- `tags`: JSON array of tags (optional)
- `summary`: Message summary (optional)