import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from actuators.chat_box import chat_box


@pytest.fixture
def mock_mcp_session():
    # Setup mock session yielding result
    mock_session = AsyncMock()
    mock_session.initialize = AsyncMock()

    mock_result = MagicMock()
    mock_result.isError = False
    mock_content = MagicMock()
    mock_content.text = "Success"
    mock_result.content = [mock_content]

    mock_session.call_tool = AsyncMock(return_value=mock_result)

    # ClientSession context manager mock
    mock_session_cm = AsyncMock()
    mock_session_cm.__aenter__.return_value = mock_session
    mock_session_cm.__aexit__.return_value = None

    return mock_session, mock_session_cm


@pytest.fixture
def mock_stdio_client():
    # stdio_client context manager mock
    mock_stdio_cm = AsyncMock()
    # yields (read_stream, write_stream)
    mock_stdio_cm.__aenter__.return_value = (AsyncMock(), AsyncMock())
    mock_stdio_cm.__aexit__.return_value = None

    return mock_stdio_cm


@pytest.mark.asyncio
@patch("actuators.chat_box.ClientSession")
@patch("actuators.chat_box.stdio_client")
async def test_chat_box_success(
    mock_stdio_client_func, mock_ClientSession, mock_stdio_client, mock_mcp_session
):
    mock_session, mock_session_cm = mock_mcp_session

    mock_stdio_client_func.return_value = mock_stdio_client
    mock_ClientSession.return_value = mock_session_cm

    result = await chat_box.ainvoke({"message": "Hello VRChat!"})

    assert result == "Success"
    mock_session.initialize.assert_awaited_once()
    mock_session.call_tool.assert_awaited_once_with(
        "send_message", arguments={"message": "Hello VRChat!"}
    )


@pytest.mark.asyncio
@patch("actuators.chat_box.ClientSession")
@patch("actuators.chat_box.stdio_client")
async def test_chat_box_tool_error(
    mock_stdio_client_func, mock_ClientSession, mock_stdio_client, mock_mcp_session
):
    mock_session, mock_session_cm = mock_mcp_session

    # Change result to represent an error
    mock_result = MagicMock()
    mock_result.isError = True
    mock_content = MagicMock()
    mock_content.text = "MCP Error"
    mock_result.content = [mock_content]
    mock_session.call_tool.return_value = mock_result

    mock_stdio_client_func.return_value = mock_stdio_client
    mock_ClientSession.return_value = mock_session_cm

    result = await chat_box.ainvoke({"message": "Trigger Error"})

    assert "Failed to send message: MCP Error" in result


@pytest.mark.asyncio
@patch("actuators.chat_box.stdio_client")
async def test_chat_box_exception(mock_stdio_client_func):
    # Simulate an exception when starting stdio_client
    mock_stdio_client_func.side_effect = Exception("Process failed to start")

    result = await chat_box.ainvoke({"message": "Trigger Exception"})

    assert "failed to send message via chat box: Process failed to start" in result
