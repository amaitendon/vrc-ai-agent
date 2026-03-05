import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, SystemMessage
from agent.memory_utils import generate_day_summary
from prompts.prompts import BASE_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_generate_day_summary_includes_system_prompt():
    # Mock memory_store.get_observations_for_date
    mock_observations = [
        {"time": "10:00", "kind": "chat", "emotion": "happy", "content": "Hello world"},
    ]

    with patch("agent.memory_utils.memory_store") as mock_store:
        mock_store.get_observations_for_date.return_value = mock_observations

        # Mock get_llm
        with patch("agent.memory_utils.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_get_llm.return_value = mock_llm
            mock_llm.ainvoke.return_value = AsyncMock(content="Summary")

            # Execute
            await generate_day_summary("2026-03-05")

            # Verify messages
            args, kwargs = mock_llm.ainvoke.call_args
            messages = args[0]

            assert len(messages) == 2
            assert isinstance(messages[0], SystemMessage)
            assert messages[0].content == BASE_SYSTEM_PROMPT
            assert isinstance(messages[1], HumanMessage)
            assert "Hello world" in messages[1].content
