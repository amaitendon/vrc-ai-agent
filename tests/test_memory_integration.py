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
        prev_was_end_action=False,
        day_summary_context="",
    )
    result = end_action.func(state)
    assert isinstance(result, str)
    assert result.startswith("NUDGE: ")
    assert "10 cycles have passed" in result


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
        prev_was_end_action=False,
        day_summary_context="",
    )
    result = end_action.func(state)
    assert isinstance(result, str)
    assert result == "action_ended"
