#!/bin/bash
# Sandy's multi-server management script

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Recall Server Configuration
RECALL_SERVER_DIR="$SCRIPT_DIR/history/recall"
RECALL_PID_FILE="$RECALL_SERVER_DIR/server.pid"
RECALL_LOG_FILE="$RECALL_SERVER_DIR/server.log"

# Discord Bot Configuration
BOT_DIR="$SCRIPT_DIR"
BOT_PID_FILE="$SCRIPT_DIR/server.pid"
BOT_LOG_FILE="$SCRIPT_DIR/bot.log"
BOT_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Extract recall server config
get_recall_config() {
    source $RECALL_SERVER_DIR/.env
    
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

# ---------------------------------------------------------------------------
# Discord Bot
# ---------------------------------------------------------------------------

start_bot() {
    if [ -f "$BOT_PID_FILE" ] && kill -0 $(cat "$BOT_PID_FILE") 2>/dev/null; then
        echo "Discord bot is already running (PID: $(cat "$BOT_PID_FILE"))"
        return 1
    fi

    if [ ! -f "$BOT_PYTHON" ]; then
        echo "Error: venv not found at $BOT_PYTHON"
        echo "Run: python -m venv .venv && .venv/bin/pip install -e ."
        return 1
    fi

    echo "Starting Discord bot..."

    cd "$BOT_DIR"
    nohup "$BOT_PYTHON" discord_handler.py > "$BOT_LOG_FILE" 2>&1 &
    BOT_PID=$!
    echo $BOT_PID > "$BOT_PID_FILE"

    sleep 2

    if kill -0 $BOT_PID 2>/dev/null; then
        echo "✓ Discord bot started successfully (PID: $BOT_PID)"
        echo "  Logs: $BOT_LOG_FILE"
    else
        echo "✗ Failed to start Discord bot"
        rm -f "$BOT_PID_FILE"
        return 1
    fi
}

stop_bot() {
    if [ ! -f "$BOT_PID_FILE" ]; then
        echo "No bot PID file found. Attempting to kill any discord_handler processes..."
        pkill -f "discord_handler.py" && echo "✓ Killed discord_handler processes" || echo "No discord_handler processes found"
        return 0
    fi

    PID=$(cat "$BOT_PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Discord bot (PID: $PID)..."
        kill "$PID"

        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "✓ Discord bot stopped"
                rm -f "$BOT_PID_FILE"
                return 0
            fi
            sleep 1
        done

        echo "Force killing Discord bot..."
        kill -9 "$PID" 2>/dev/null
        rm -f "$BOT_PID_FILE"
        echo "✓ Discord bot force stopped"
    else
        echo "Discord bot not running (stale PID file removed)"
        rm -f "$BOT_PID_FILE"
    fi
}

bot_status() {
    if [ -f "$BOT_PID_FILE" ] && kill -0 $(cat "$BOT_PID_FILE") 2>/dev/null; then
        PID=$(cat "$BOT_PID_FILE")
        echo "✓ Discord bot is running (PID: $PID)"
        echo "  Logs: $BOT_LOG_FILE"
    else
        echo "✗ Discord bot is not running"
        [ -f "$BOT_PID_FILE" ] && rm -f "$BOT_PID_FILE"
        return 1
    fi
}

restart_bot() {
    echo "Restarting Discord bot..."
    stop_bot
    sleep 1
    start_bot
}

show_bot_logs() {
    if [ -f "$BOT_LOG_FILE" ]; then
        echo "Recent Discord bot logs:"
        echo "========================"
        tail -n 20 "$BOT_LOG_FILE"
    else
        echo "No bot log file found"
    fi
}

show_status_all() {
    echo "=== Sandy Services ==="
    echo ""
    echo "[ Recall ]"
    recall_status
    echo ""
    echo "[ Discord Bot ]"
    bot_status
}

show_help() {
    echo "Sandy — Multi-Service Management"
    echo ""
    echo "Usage: $0 {service} {command}"
    echo ""
    echo "Services:"
    echo "  recall   - Recall API server (chat message storage)"
    echo "  bot      - Discord bot"
    echo "  all      - Status of all services"
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
    echo "  $0 bot start"
    echo "  $0 all status"
    echo ""
}

case "$1" in
    recall)
        case "$2" in
            start)   start_recall ;;
            stop)    stop_recall ;;
            restart) restart_recall ;;
            status)  recall_status ;;
            logs)    show_recall_logs ;;
            *) echo "Unknown command for recall: $2"; show_help; exit 1 ;;
        esac
        ;;
    bot)
        case "$2" in
            start)   start_bot ;;
            stop)    stop_bot ;;
            restart) restart_bot ;;
            status)  bot_status ;;
            logs)    show_bot_logs ;;
            *) echo "Unknown command for bot: $2"; show_help; exit 1 ;;
        esac
        ;;
    all)
        case "$2" in
            status)  show_status_all ;;
            *) echo "Unknown command for all: $2"; show_help; exit 1 ;;
        esac
        ;;
    *)
        show_help
        exit 1
        ;;
esac
