
"""
Centralized logging configuration for Sandy.

Import this module once (e.g. in discord_handler.py at startup) and it
configures the root logger for the entire process.  Every other module
just does the standard:

    import logging
    logger = logging.getLogger(__name__)

Or use the convenience wrapper:

    from logconf import get_logger
    logger = get_logger(__name__)

Logging is async-safe: all records go through a QueueHandler so the
calling thread/coroutine is never blocked by I/O.  A QueueListener
drains the queue on a background thread and writes to stdout.
atexit cleans up the listener automatically on shutdown.
"""

import atexit
import logging
import logging.handlers
import queue


_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# The actual destination — writes to stdout
_stdout_handler = logging.StreamHandler()
_stdout_handler.setFormatter(_formatter)

# A queue that receives log records from any thread/coroutine without blocking
_log_queue = queue.SimpleQueue()
_queue_handler = logging.handlers.QueueHandler(_log_queue)

# The listener drains the queue on a dedicated background thread
_listener = logging.handlers.QueueListener(_log_queue, _stdout_handler, respect_handler_level=True)
_listener.start()
atexit.register(_listener.stop)

# Configure the root logger — everything in the process inherits this
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_queue_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call as get_logger(__name__) in each module."""
    return logging.getLogger(name)


