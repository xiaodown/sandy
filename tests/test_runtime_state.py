from sandy.runtime_state import RuntimeState
from sandy.trace import TurnTrace


def _trace(trace_id: str = "123") -> TurnTrace:
    return TurnTrace(
        trace_id=trace_id,
        message_id=int(trace_id),
        guild_id=1,
        guild_name="Guild",
        channel_id=2,
        channel_name="general",
        author_id=3,
        author_name="alice",
        started_at=1.0,
    )


def test_runtime_state_tracks_turns_memory_and_bouncer() -> None:
    state = RuntimeState()
    trace = _trace()

    state.set_discord_connected(True, user_name="Sandy")
    state.set_discord_servers(["Guild One", "Guild Two"])
    state.begin_turn(trace, author_is_bot=False)
    state.update_turn_stage(trace, "brain")
    state.memory_enqueued()
    state.memory_processing_started(message_id=trace.message_id)
    state.set_last_bouncer_decision(
        trace_id=trace.trace_id,
        should_respond=True,
        use_tool=False,
        tool_name=None,
    )

    snapshot = state.snapshot()

    assert snapshot["discord"]["connected"] is True
    assert snapshot["discord"]["user_name"] == "Sandy"
    assert snapshot["discord"]["server_count"] == 2
    assert snapshot["discord"]["server_names"] == ["Guild One", "Guild Two"]
    assert snapshot["active_turns"][0]["trace_id"] == trace.trace_id
    assert snapshot["active_turns"][0]["stage"] == "brain"
    assert snapshot["memory_worker"]["busy"] is True
    assert snapshot["memory_worker"]["queue_depth"] == 0
    assert snapshot["last_bouncer_decision"]["should_respond"] is True

    state.memory_processing_finished(message_id=trace.message_id)
    state.end_turn(trace.trace_id)
    snapshot = state.snapshot()

    assert snapshot["active_turns"] == []
    assert snapshot["memory_worker"]["busy"] is False
