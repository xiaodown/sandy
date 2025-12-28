#!/bin/bash
# Sandy's multi-server management script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Recall Server Configuration
RECALL_SERVER_DIR="$SCRIPT_DIR/history/recall"
RECALL_PID_FILE="$RECALL_SERVER_DIR/server.pid"
RECALL_LOG_FILE="$RECALL_SERVER_DIR/server.log"

# Extract recall server config
get_recall_config() {
    RECALL_PORT=$(cd "$RECALL_SERVER_DIR" && uv run python -c "import settings; print(settings.port)" 2>/dev/null)
    RECALL_HOST=$(cd "$RECALL_SERVER_DIR" && uv run python -c "import settings; print(settings.host)" 2>/dev/null)
    
    if [ -z "$RECALL_PORT" ] || [ -z "$RECALL_HOST" ]; then
        echo "Error: Could not read recall server settings"
        exit 1
    fi
}

start_recall() {
    if [ -f "$RECALL_PID_FILE" ] && kill -0 $(cat "$RECALL_PID_FILE") 2>/dev/null; then
        echo "Recall server is already running (PID: $(cat "$RECALL_PID_FILE"))"
        return 1
    fi
    
    get_recall_config
    
    echo "Starting Recall server..."
    echo "The server will be available at: http://$RECALL_HOST:$RECALL_PORT"
    echo "API documentation: http://$RECALL_HOST:$RECALL_PORT/docs"
    echo ""
    
    cd "$RECALL_SERVER_DIR"
    nohup uv run uvicorn main:app --reload --host "$RECALL_HOST" --port "$RECALL_PORT" > "$RECALL_LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$RECALL_PID_FILE"
    
    # Give it a moment to start
    sleep 2
    
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "✓ Recall server started successfully (PID: $SERVER_PID)"
        echo "  Logs: $RECALL_LOG_FILE"
    else
        echo "✗ Failed to start Recall server"
        rm -f "$RECALL_PID_FILE"
        return 1
    fi
}

stop_recall() {
    if [ ! -f "$RECALL_PID_FILE" ]; then
        echo "No Recall PID file found. Attempting to kill any recall uvicorn processes..."
        pkill -f "uvicorn main:app" && echo "✓ Killed recall uvicorn processes" || echo "No recall uvicorn processes found"
        return 0
    fi
    
    PID=$(cat "$RECALL_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Recall server (PID: $PID)..."
        kill "$PID"
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "✓ Recall server stopped"
                rm -f "$RECALL_PID_FILE"
                return 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        echo "Force killing Recall server..."
        kill -9 "$PID" 2>/dev/null
        rm -f "$RECALL_PID_FILE"
        echo "✓ Recall server force stopped"
    else
        echo "Recall server not running (stale PID file removed)"
        rm -f "$RECALL_PID_FILE"
    fi
}

recall_status() {
    get_recall_config
    
    if [ -f "$RECALL_PID_FILE" ] && kill -0 $(cat "$RECALL_PID_FILE") 2>/dev/null; then
        PID=$(cat "$RECALL_PID_FILE")
        echo "✓ Recall server process is running (PID: $PID)"
        echo "  URL: http://$RECALL_HOST:$RECALL_PORT"
        echo "  Docs: http://$RECALL_HOST:$RECALL_PORT/docs"
        echo "  Logs: $RECALL_LOG_FILE"
        
        # Test if server is actually responding
        echo -n "  Web server status: "
        if curl -s --connect-timeout 3 "http://$RECALL_HOST:$RECALL_PORT/" > /dev/null 2>&1; then
            echo "✓ Responding"
        else
            echo "✗ Not responding (process exists but web server may have failed)"
            echo "  Check logs with: $0 recall logs"
            return 1
        fi
    else
        echo "✗ Recall server is not running"
        [ -f "$RECALL_PID_FILE" ] && rm -f "$RECALL_PID_FILE"  # Clean up stale PID file
        return 1
    fi
}

restart_recall() {
    echo "Restarting Recall server..."
    stop_recall
    sleep 1
    start_recall
}

show_recall_logs() {
    if [ -f "$RECALL_LOG_FILE" ]; then
        echo "Recent Recall server logs:"
        echo "=========================="
        tail -n 20 "$RECALL_LOG_FILE"
    else
        echo "No Recall log file found"
    fi
}

# TODO: Add other server functions here (ollama, discord bot, etc.)

show_help() {
    echo "Multi-Server Management"
    echo ""
    echo "Usage: $0 {service} {command}"
    echo ""
    echo "Services:"
    echo "  recall       - Recall API server (chat message storage)"
    echo "  # TODO: ollama, discord-bot, etc."
    echo ""
    echo "Commands:"
    echo "  start    - Start the service"
    echo "  stop     - Stop the service" 
    echo "  restart  - Restart the service"
    echo "  status   - Show service status"
    echo "  logs     - Show recent service logs"
    echo ""
    echo "Examples:"
    echo "  $0 recall start"
    echo "  $0 recall status"
    echo ""
}

case "$1" in
    recall)
        case "$2" in
            start)
                start_recall
                ;;
            stop)
                stop_recall
                ;;
            restart)
                restart_recall
                ;;
            status)
                recall_status
                ;;
            logs)
                show_recall_logs
                ;;
            *)
                echo "Unknown command for recall: $2"
                show_help
                exit 1
                ;;
        esac
        ;;
    *)
        show_help
        exit 1
        ;;
esac