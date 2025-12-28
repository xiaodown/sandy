from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# Import recall server settings
# This should be the final import because of the path munge jank:
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "recall"))
import settings as recall_settings
# end of jank (heh, path munge import jank at least)

# Initialize FastMCP server
mcp = FastMCP("recall")

# Constants
RECALL_URL = f"http://{recall_settings.host}:{recall_settings.port}"
USER_AGENT = "recall-mcp-server/1.0"


async def make_recall_request(url: str) -> dict[str, Any] | None:
    """Make a request to the recall API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


@mcp.tool()
async def get_chat_history(
    author: str | None = None,
    server: str | None = None, 
    channel: str | None = None,
    tag: str | None = None,
    hours_ago: int | None = None,
    minutes_ago: int | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100
) -> str:
    """Get chat message history from the recall API with optional filtering.
    
    Args:
        author: Filter by message author
        server: Filter by server name  
        channel: Filter by channel name
        tag: Filter by tag
        hours_ago: Get messages from last N hours
        minutes_ago: Get messages from last N minutes  
        since: Get messages since specific time (ISO format)
        until: Get messages until specific time (ISO format)
        limit: Maximum number of messages to return (1-1000)
    
    Returns:
        Formatted string of chat messages or error message
    """
    # Build query parameters
    params = []
    if author:
        params.append(f"author={author}")
    if server:
        params.append(f"server={server}")
    if channel:
        params.append(f"channel={channel}")
    if tag:
        params.append(f"tag={tag}")
    if hours_ago:
        params.append(f"hours_ago={hours_ago}")
    if minutes_ago:
        params.append(f"minutes_ago={minutes_ago}")
    if since:
        params.append(f"since={since}")
    if until:
        params.append(f"until={until}")
    if limit != 100:
        params.append(f"limit={limit}")
    
    query_string = "?" + "&".join(params) if params else ""
    url = f"{RECALL_URL}/messages/{query_string}"
    
    data = await make_recall_request(url)
    if data is None:
        return "Error: Could not retrieve chat history from recall API"
    
    if not data:
        return "No messages found matching the specified criteria."
    
    # Format the messages for display
    formatted_messages = []
    for msg in data:
        timestamp = msg.get('timestamp', 'Unknown time')
        author = msg.get('author', 'Unknown author')
        server = msg.get('server', 'Unknown server')
        channel = msg.get('channel', 'Unknown channel')
        content = msg.get('content', '')
        tags = msg.get('tags', [])
        summary = msg.get('summary', '')
        
        # Format timestamp to be more readable
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass  # Keep original timestamp if parsing fails
            
        message_str = f"[{timestamp}] {server}#{channel} <{author}>: {content}"
        if tags:
            message_str += f" [tags: {', '.join(tags)}]"
        if summary:
            message_str += f" (summary: {summary})"
            
        formatted_messages.append(message_str)
    
    return "\n".join(formatted_messages)


@mcp.tool()
async def search_messages(
    query: str,
    author: str | None = None,
    server: str | None = None,
    channel: str | None = None,
    hours_ago: int | None = None,
    limit: int = 50
) -> str:
    """Search through chat message content for specific text or topics.
    
    Perfect for finding past conversations about specific topics, remembering what
    someone said about something, or finding relevant context from chat history.
    This performs a case-insensitive search through message content.
    
    Args:
        query: Text to search for in message content (case-insensitive)
        author: Optionally limit search to specific author
        server: Optionally limit search to specific server
        channel: Optionally limit search to specific channel  
        hours_ago: Optionally limit search to last N hours
        limit: Maximum number of messages to return (default 50)
    
    Returns:
        Formatted string of matching chat messages or error message
    """
    # Get messages with basic filters first
    params = []
    if author:
        params.append(f"author={author}")
    if server:
        params.append(f"server={server}")
    if channel:
        params.append(f"channel={channel}")
    if hours_ago:
        params.append(f"hours_ago={hours_ago}")
    # Use a higher limit for search since we'll filter client-side
    params.append(f"limit={min(limit * 3, 1000)}")  
    
    query_string = "?" + "&".join(params) if params else f"?limit={min(limit * 3, 1000)}"
    url = f"{RECALL_URL}/messages/{query_string}"
    
    data = await make_recall_request(url)
    if data is None:
        return "Error: Could not search chat history from recall API"
    
    if not data:
        return f"No messages found matching search criteria."
    
    # Filter messages that contain the search query (case-insensitive)
    query_lower = query.lower()
    matching_messages = []
    
    for msg in data:
        content = msg.get('content', '').lower()
        summary = msg.get('summary', '').lower()
        
        # Search in content and summary
        if query_lower in content or query_lower in summary:
            matching_messages.append(msg)
            
        # Stop when we have enough results
        if len(matching_messages) >= limit:
            break
    
    if not matching_messages:
        return f"No messages found containing '{query}'."
    
    result = f"Found {len(matching_messages)} messages containing '{query}':\n\n"
    
    # Format the messages for display (reuse the same logic)
    formatted_messages = []
    for msg in matching_messages:
        timestamp = msg.get('timestamp', 'Unknown time')
        author = msg.get('author', 'Unknown author')
        server = msg.get('server', 'Unknown server')
        channel = msg.get('channel', 'Unknown channel')
        content = msg.get('content', '')
        tags = msg.get('tags', [])
        summary = msg.get('summary', '')
        
        # Format timestamp to be more readable
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass  # Keep original timestamp if parsing fails
            
        message_str = f"[{timestamp}] {server}#{channel} <{author}>: {content}"
        if tags:
            message_str += f" [tags: {', '.join(tags)}]"
        if summary:
            message_str += f" (summary: {summary})"
            
        formatted_messages.append(message_str)
    
    result += "\n".join(formatted_messages)
    return result


if __name__ == "__main__":
    mcp.run()