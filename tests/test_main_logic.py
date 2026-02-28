import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from main import audio_listener
from core.context import QueueEvent, PRIORITY_VOICE


def test_queue_event_priority():
    ev1 = QueueEvent(priority=10, text="lower")
    ev2 = QueueEvent(priority=0, text="higher")

    # PriorityQueueは値が小さいほど優先度が高い
    assert ev2 < ev1


@pytest.mark.asyncio
async def test_audio_listener_interrupt(clean_app_context):
    ctx = clean_app_context
    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = False
    ctx.say_task = mock_task

    # setup_audio_listener 内の処理として pipeline._on_speech_detected が呼ばれたとする
    # main_logic の audio_listener は現在単なるラップ関数なので、ここでは直接内部処理をモックする
    from inputs.audio import AudioInputPipeline

    with patch.object(AudioInputPipeline, "start_listening", new_callable=AsyncMock):
        # start_listening を呼び出すとすぐに終わるようにモック
        await audio_listener(ctx)

    # テストとして、マイクからの発話検知コールバックを手動で呼び出してみる
    pipeline = AudioInputPipeline(ctx)
    from inputs.stt_faster_whisper import FasterWhisperSpeechRecognizer

    with patch.object(
        FasterWhisperSpeechRecognizer, "transcribe", new_callable=AsyncMock
    ) as mock_transcribe:
        mock_transcribe.return_value = "test input"
        await pipeline._on_speech_detected(b"\x00")

    # say_taskがキャンセルされているか
    mock_task.cancel.assert_called_once()

    # キューにイベントが入っているか
    assert not ctx.priority_queue.empty()
    event = await ctx.priority_queue.get()
    assert event.text == "test input"
    assert event.priority == PRIORITY_VOICE
    assert event.interrupted_action == "say"
