import pytest
from unittest.mock import patch, MagicMock

from actuators.chat_box import chat_box


@pytest.fixture
def mock_osc_client():
    with patch("actuators.chat_box.OSCClient") as MockOSCClient:
        mock_instance = MagicMock()
        MockOSCClient.get.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_chat_box_success(mock_osc_client):
    result = await chat_box.ainvoke({"message": "Hello VRChat!"})

    assert result == "Message sent."
    mock_osc_client.send_message.assert_called_once_with(
        "/chatbox/input", ["Hello VRChat!", True, True]
    )


@pytest.mark.asyncio
async def test_chat_box_exception(mock_osc_client):
    # Simulate an exception when sending message
    mock_osc_client.send_message.side_effect = Exception("OSC connection error")

    result = await chat_box.ainvoke({"message": "Trigger Exception"})

    assert "failed to send message via chat box: OSC connection error" in result
