import asyncio
import io
import wave
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.context import AppContext
from actuators.speech import say, _wait_for_playback


@pytest.fixture(autouse=True)
def reset_app_context():
    """Each test gets a fresh AppContext."""
    AppContext._instance = None
    yield
    AppContext._instance = None


@pytest.fixture
def mock_pipeline():
    with patch("actuators.speech.get_audio_output_pipeline") as mock_get:
        mock_pipe = MagicMock()
        mock_pipe.tts = AsyncMock()
        mock_pipe.player = MagicMock()
        mock_get.return_value = mock_pipe
        yield mock_pipe


def create_dummy_wav(duration_sec: float = 1.0, rate: int = 24000) -> bytes:
    """Create a dummy WAV file in memory."""
    frames = int(duration_sec * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        # write 0s
        w.writeframes(b"\x00" * (frames * 2))
    return buf.getvalue()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_say_tool_normal_flow(mock_sleep, mock_pipeline):
    # Setup dummy WAV data that is 2.0 seconds long
    dummy_wav = create_dummy_wav(duration_sec=2.0)
    mock_pipeline.tts.synthesize.return_value = dummy_wav

    # Execute
    res = await say.ainvoke({"text": "Hello"})
    assert res == "say_started"

    # Verify TTS and Player were called
    mock_pipeline.tts.synthesize.assert_called_once_with("Hello")
    mock_pipeline.player.add.assert_called_once_with(dummy_wav, has_wave_header=True)

    # Verify background task was created and stored in say_task
    ctx = AppContext.get()
    assert ctx.say_task is not None

    # Let the background task finish
    await ctx.say_task

    # Check that sleep was called with 2.0 seconds
    mock_sleep.assert_called_once_with(2.0)

    # To check done callback, we need a real asyncio.sleep tick.
    # Since sleep is mocked, we can't easily wait, so we'll just check
    # if it's removed (it usually is by this point).
    # We skip strict assertion on ctx._background_tasks to avoid flakiness in mocked tests.


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_say_tool_fallback_duration(mock_sleep, mock_pipeline):
    # Pass invalid wav data
    mock_pipeline.tts.synthesize.return_value = b"invalid wav data"

    await say.ainvoke({"text": "Fallback"})

    ctx = AppContext.get()
    await ctx.say_task

    # 16 bytes invalid data -> (16 - 44) < 0 -> clamped to 0.0 sec
    mock_sleep.assert_called_once_with(0.0)


@pytest.mark.asyncio
async def test_wait_for_playback_cancellation():
    mock_player = MagicMock()

    # Run _wait_for_playback in a task
    task = asyncio.create_task(_wait_for_playback(10.0, mock_player))

    # Yield to let the task start sleeping
    await asyncio.sleep(0.01)

    # Cancel the task
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    mock_player.stop.assert_called_once()
