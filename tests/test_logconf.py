from __future__ import annotations

import json
import logging

from sandy.logconf import ConsoleFormatter, JsonlFormatter, _HttpxConsoleFilter, emit_forensic_record


def test_jsonl_formatter_marks_trace_records() -> None:
    formatter = JsonlFormatter()
    record = logging.LogRecord(
        name="sandy.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="trace message",
        args=(),
        exc_info=None,
    )
    record.event_payload = {"trace_id": "123", "stage": "message_received"}

    rendered = json.loads(formatter.format(record))

    assert rendered["record_type"] == "trace"
    assert rendered["event"] == {"trace_id": "123", "stage": "message_received"}


def test_jsonl_formatter_marks_forensic_records() -> None:
    formatter = JsonlFormatter()
    record = logging.LogRecord(
        name="sandy.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="forensic message",
        args=(),
        exc_info=None,
    )
    record.forensic_payload = {"trace_id": "123", "artifact": "brain_generation"}

    rendered = json.loads(formatter.format(record))

    assert rendered["record_type"] == "forensic"
    assert rendered["forensic"] == {
        "trace_id": "123",
        "artifact": "brain_generation",
    }


def test_emit_forensic_record_routes_away_from_console_and_trace_store() -> None:
    logger = logging.getLogger("sandy.test.forensic")
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = CaptureHandler()
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    emit_forensic_record(logger, "FORENSIC example", {"trace_id": "123"})

    assert len(records) == 1
    record = records[0]
    assert record.forensic_payload == {"trace_id": "123"}
    assert record.log_to_console is False
    assert record.log_to_trace_store is False


def test_console_formatter_adds_ansi_and_separators() -> None:
    formatter = ConsoleFormatter()
    record = logging.LogRecord(
        name="sandy.bot",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello world",
        args=(),
        exc_info=None,
    )

    rendered = formatter.format(record)

    assert "\033[" in rendered
    assert " | " in rendered
    assert "sandy.bot" in rendered
    assert "hello world" in rendered


def test_httpx_console_filter_suppresses_info_but_keeps_warnings() -> None:
    flt = _HttpxConsoleFilter()
    info_record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="HTTP Request",
        args=(),
        exc_info=None,
    )
    warning_record = logging.LogRecord(
        name="httpx",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="warning",
        args=(),
        exc_info=None,
    )
    other_record = logging.LogRecord(
        name="sandy.bot",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="normal log",
        args=(),
        exc_info=None,
    )

    assert flt.filter(info_record) is False
    assert flt.filter(warning_record) is True
    assert flt.filter(other_record) is True
