#!/bin/bash
# Start the Chat Messages API server

# Extract port and host from settings.py
PORT=$(cd "$(dirname "$0")/history/server" && python3 -c "import settings; print(settings.port)")
HOST=$(cd "$(dirname "$0")/history/server" && python3 -c "import settings; print(settings.host)")

echo "Starting Chat History server..."
echo "The server will be available at: http://$HOST:$PORT"
echo "API documentation: http://$HOST:$PORT/docs"
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")/history/server"
uv run uvicorn main:app --reload --host $HOST --port $PORT &