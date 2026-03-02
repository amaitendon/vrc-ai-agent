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
    session_id = "test_session_123"

    # 1. まず声が検知された瞬間の挙動を確認
    await pipeline._on_voiced(session_id)

    # say_task.cancel() が呼ばれたか確認
    mock_say_task.cancel.assert_called_once()
    # セッションデータに保存されているか確認
    assert pipeline.vad.get_session_data(session_id, "interrupted_action") == "say"

    # 2. 次に発話が完了した際の挙動を確認
    await pipeline._on_speech_detected(dummy_audio_data, session_id=session_id)

    # キューに正しくイベントが入ったか確認
    event: QueueEvent = await ctx.priority_queue.get()
    assert event.priority == PRIORITY_VOICE
    assert event.text.endswith(
        "テストの割り込み発言です"
    )  # タイムスタンプが含まれるため endswith で判定
    assert event.interrupted_action == "say"
