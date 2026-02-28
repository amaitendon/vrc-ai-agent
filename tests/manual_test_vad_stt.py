"""
tests/manual_test_vad_stt.py

マイクのVAD（Silero検知）からSTT（Faster Whisper）への一連のパイプラインが
実機デバイスで正しく動作するか確認するスクリプト。
"""

import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

from dotenv import load_dotenv  # noqa: E402

from core.context import AppContext  # noqa: E402
from inputs.audio import AudioInputPipeline  # noqa: E402


async def print_queue_loop(ctx: AppContext):
    """
    AudioInputPipeline によって AppContext のキューに text がプッシュされるので
    それを取り出してターミナルに表示し続ける。
    """
    print("🎧 Queue Loop started. Waiting for speech...")
    while True:
        event = await ctx.priority_queue.get()
        print(f"\n✅ [STT RESULT] {event.text}\n")
        ctx.priority_queue.task_done()


async def main():
    load_dotenv()

    ctx = AppContext.get()
    pipeline = AudioInputPipeline(ctx)

    from utils.audio import get_device_name_by_index

    device_name = get_device_name_by_index(pipeline.input_device)

    print("✅ AudioInputPipeline generated.")
    print(f"🎤 Input Device Index: {pipeline.input_device} (Name: {device_name})")
    print("🎙️ Please speak into your microphone. Say 'Ctrl+C' to exit.")

    # 2つのタスクを並行実行
    # 1. 音声を監視してQueueに入れる
    # 2. Queueから取り出してPrintする
    try:
        await asyncio.gather(pipeline.start_listening(), print_queue_loop(ctx))
    except asyncio.CancelledError:
        print("🛑 Stopped manual testing.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("🛑 Exited by user.")
