"""
Settings for the chat history server
"""


# The sqlite database for holding chat messages for the MCP server to consume.
# default: ../database/history.db (from the git checkout root, history/database/history.db)
#
# NOTE: the database directory is in the .gitignore; if you change it, remember to also 
# change your git ignore so that you don't check your database in to version control.

db_path="../database/history.db"

# Server configuration
port=8000
host="127.0.0.1"