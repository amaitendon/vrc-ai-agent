import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from core.context import AppContext, QueueEvent, PRIORITY_VOICE
from inputs.audio import AudioInputPipeline


@pytest.mark.asyncio
async def test_audio_input_pipeline_voice_interrupt():
    """
    VADが発話を検知した際に、既に実行中の say_task があれば cancel() され、
    interrupted_action='say' の情報が付与された状態でキューに入るかテスト。
    """
    ctx = AppContext.get()

    # sayタスクをモックし、まだ完了していない(False)を返すように設定
    mock_say_task = MagicMock(spec=asyncio.Task)
    mock_say_task.done.return_value = False
    ctx.say_task = mock_say_task

    pipeline = AudioInputPipeline(ctx)

    # STTの transcribe をモックして特定の文字列を即座に返すようにする
    pipeline.stt.transcribe = AsyncMock(return_value="テストの割り込み発言です")

    dummy_audio_data = b"\x00" * 1024

    await pipeline._on_speech_detected(dummy_audio_data)

    # say_task.cancel() が呼ばれたか確認
    mock_say_task.cancel.assert_called_once()

    # キューに正しくイベントが入ったか確認
    event: QueueEvent = await ctx.priority_queue.get()
    assert event.priority == PRIORITY_VOICE
    assert event.text == "テストの割り込み発言です"
    assert event.interrupted_action == "say"
