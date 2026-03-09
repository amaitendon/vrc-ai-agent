import pytest
from agent.nodes.action import memory_store, end_action, remember, recall
from agent.state import AgentState
from datetime import datetime


@pytest.mark.asyncio
async def test_memory_tools_exist():
    assert memory_store is not None
    assert end_action is not None
    assert remember is not None
    assert recall is not None


def test_end_action_nudge():
    # Test nudge at 10 cycles (threshold depends on MAX_HISTORY, typically 10)
    state = AgentState(
        messages=[],
        current_time=datetime.now(),
        current_date="2026-03-01",
        osc_status=None,
        tool_call_history=[],
        last_spoke_at=None,
        last_memory_saved_at=None,
        unsaved_cycles=10,
        nudge_remember="idle",
        day_summary_context="",
    )
    result = end_action.func(state, tool_call_id="test_id")
    from langgraph.types import Command
    assert isinstance(result, Command)
    assert result.update["nudge_remember"] == "nudge_pending"
    assert "10 cycles have passed" in result.update["messages"][0].content


def test_end_action_no_nudge():
    # Test end action at 0 cycles
    state = AgentState(
        messages=[],
        current_time=datetime.now(),
        current_date="2026-03-01",
        osc_status=None,
        tool_call_history=[],
        last_spoke_at=None,
        last_memory_saved_at=None,
        unsaved_cycles=0,
        nudge_remember="idle",
        day_summary_context="",
    )
    result = end_action.func(state, tool_call_id="test_id")
    from langgraph.types import Command
    assert isinstance(result, Command)
    assert result.update["nudge_remember"] == "idle"
    assert result.update["messages"][0].content == "action_ended"


@pytest.mark.asyncio
async def test_remember_tool_no_image():
    # content and emotion are required/optional
    result = await remember.coroutine(
        content="Test memory content", emotion="happy", tool_call_id="test_id"
    )
    from langgraph.types import Command
    assert isinstance(result, Command)
    assert "Remembered:" in result.update["messages"][0].content
    assert "Test memory content" in result.update["messages"][0].content
    assert result.update["unsaved_cycles"] == 0
    # Check that image_filename is no longer in the tool's arguments schema
    assert "image_filename" not in remember.args
